//go:build darwin

package gpu

import (
	"fmt"
	"math"
	"sync"

	"github.com/townsendmerino/aikit/ann"
)

// annbackend.go registers a native-Metal implementation of ann.Backend, so an
// ann.FlatI8 that calls EnableGPU scores its int8 corpus GEMV on the GPU. This is
// the Phase-1 "one proving consumer": it exercises the whole device path — the
// lifted device layer + a minimal W8A8 GEMV kernel — on aikit's highest-fit
// workload (queries × the whole index), parity-gated against the CPU
// linalg.MatmulBTW8A8 (gemv_test.go). The kernel here is intentionally minimal and
// correctness-only; the tuned kernel + batched multi-query GEMM are Phase 2.

// gemvW8A8Src is a minimal, unoptimized W8A8 GEMV: one thread per index row j
// computes the exact int32 dot of the (host-quantized) int8 query against row j,
// then rescales by the query and per-row scales — the same value w8a8Span
// produces on the CPU. The integer dot is exact, so GPU and CPU rank identically.
const gemvW8A8Src = `
#include <metal_stdlib>
using namespace metal;
kernel void gemv_w8a8(
    device const char*  codes  [[buffer(0)]],   // [N*K] int8, row-major
    device const char*  qi8    [[buffer(1)]],   // [K]   int8 query (host-quantized)
    device const float* scales [[buffer(2)]],   // [N]   per-row scale
    constant uint&      K      [[buffer(3)]],
    constant float&     qscale [[buffer(4)]],   // query scale (0 ⇒ all-zero query)
    device float*       out    [[buffer(5)]],   // [N]   scores
    uint j [[thread_position_in_grid]]) {
    device const char* row = codes + (uint)j * K;
    int acc = 0;
    for (uint k = 0; k < K; k++) {
        acc += int(qi8[k]) * int(row[k]);
    }
    out[j] = float(acc) * qscale * scales[j];
}`

type metalBackend struct {
	dev  *Device
	q    Queue
	pipe Pipeline
}

// init reaches Metal and registers the backend. If there is no Metal GPU (or the
// kernel fails to compile), it registers nothing — ann.FlatI8 then stays on the
// CPU and EnableGPU returns "no backend". Building this package is the opt-in.
func init() {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		return
	}
	lib, err := dev.CompileLibrary(gemvW8A8Src, MSL3_1)
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	pipe, err := dev.NewComputePipeline(lib, "gemv_w8a8")
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	ann.RegisterBackend(&metalBackend{dev: dev, q: dev.NewCommandQueue(), pipe: pipe})
}

func (b *metalBackend) Name() string { return "metal" }

// NewI8Index uploads the int8 codes + scales resident on the device and allocates
// the reusable per-query scratch (quantized query, query scale, output).
func (b *metalBackend) NewI8Index(bq []int8, scales []float32, n, dim int) (ann.I8Index, error) {
	if n <= 0 || dim <= 0 {
		return nil, fmt.Errorf("gpu: empty int8 index (n=%d dim=%d)", n, dim)
	}
	if len(bq) != n*dim || len(scales) != n {
		return nil, fmt.Errorf("gpu: int8 index shape mismatch (bq %d, scales %d, n=%d dim=%d)", len(bq), len(scales), n, dim)
	}
	return &metalI8Index{
		b:      b,
		n:      n,
		dim:    dim,
		codes:  b.dev.NewBufferInt8(bq),
		scales: b.dev.NewBufferFloats(scales),
		qi8:    b.dev.NewBufferInt8(make([]int8, dim)),
		qscale: b.dev.NewBufferFloats([]float32{0}),
		kbuf:   b.dev.NewBufferU32(uint32(dim)),
		out:    b.dev.NewBufferLen(n),
	}, nil
}

type metalI8Index struct {
	b      *metalBackend
	n, dim int
	codes  Buffer // [n*dim] int8 (resident)
	scales Buffer // [n] f32
	qi8    Buffer // [dim] int8 (reused per query)
	qscale Buffer // [1] f32 (reused)
	kbuf   Buffer // [1] u32
	out    Buffer // [n] f32 (reused)
	mu     sync.Mutex
}

// Score dynamically quantizes q on the host (byte-identical to linalg's
// quantizeRowInt8), dispatches the GEMV, and copies the scores back. Serialized:
// the reused scratch buffers are per-index, and one command queue runs serially.
func (x *metalI8Index) Score(q []float32, dst []float32) error {
	if len(q) != x.dim {
		return fmt.Errorf("gpu: query dim %d != index dim %d", len(q), x.dim)
	}
	if len(dst) != x.n {
		return fmt.Errorf("gpu: dst len %d != index n %d", len(dst), x.n)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	qscale := quantizeRowInt8(q, x.qi8.Int8s())
	x.qscale.Floats()[0] = qscale
	tg := 256
	if tg > x.n {
		tg = x.n
	}
	x.b.q.Run1D(x.b.pipe, x.n, tg, x.codes, x.qi8, x.scales, x.kbuf, x.qscale, x.out)
	copy(dst, x.out.Floats())
	return nil
}

// Close releases this index's device buffers (the shared device stays alive for
// other indexes; the process holds one Metal device).
func (x *metalI8Index) Close() error {
	for _, b := range []Buffer{x.codes, x.scales, x.qi8, x.qscale, x.kbuf, x.out} {
		x.b.dev.ReleaseBuf(b)
	}
	return nil
}

// quantizeRowInt8 replicates linalg's (unexported) per-row int8 quantizer exactly:
// scale = maxAbs/127, dst[k] = round(q[k]/scale) clamped to ±127; all-zero input →
// scale 0. Kept byte-identical so the GPU int8 dot equals the CPU one.
func quantizeRowInt8(a []float32, dst []int8) (scale float32) {
	var maxAbs float32
	for _, v := range a {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		for k := range dst {
			dst[k] = 0
		}
		return 0
	}
	scale = maxAbs / 127
	inv := 1.0 / scale
	for k, v := range a {
		x := math.Round(float64(v * inv))
		if x > 127 {
			x = 127
		} else if x < -127 {
			x = -127
		}
		dst[k] = int8(x)
	}
	return scale
}

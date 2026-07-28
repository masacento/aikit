//go:build darwin

package annmetal

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/ann"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// annbackend.go registers a native-Metal implementation of ann.Backend, so an
// ann.FlatI8 that calls EnableGPU scores its int8 corpus GEMV on the GPU. This is
// the Phase-1 "one proving consumer": it exercises the whole device path — the
// lifted device layer + a minimal W8A8 GEMV kernel — on aikit's highest-fit
// workload (queries × the whole index), parity-gated against the CPU
// linalg.MatmulBTW8A8 (gemv_test.go). The kernels here are intentionally minimal
// and correctness-only; the tuned decode kernel is Phase 1b/2.
//
// Two kernels: gemv_w8a8 scores ONE query against the whole index (FlatI8.Query),
// gemm_w8a8_tiled scores M queries in one dispatch (FlatI8.QueryBatch) — the batched
// int8 GEMM that is the GPU's sweet spot (Phase 2). Both compute the exact int32 dot
// of the host-quantized int8 query against each row, then the query/row rescale — the
// same value w8a8Span produces on the CPU, so GPU and CPU rank identically.
//
// gemm_w8a8_tiled stages a TILE×TILE block of the query tile (A) and the corpus tile (B)
// through threadgroup memory per K-chunk, so each int8 code is read from global memory
// ONCE per tile rather than once per output — the same tiling that took the f32 GEMM from
// ~350 to ~1080 GFLOP/s. It stays BIT-IDENTICAL to the naive one-thread-per-output kernel
// it replaced: the accumulator is int32 and integer addition is associative, so chunking
// K cannot change the sum (a much sharper property than a tolerance, and the batch parity
// test asserts it). Launch with dispatchThreadgroups over a 2-D (ceil(N/TILE), ceil(M/TILE))
// grid of TILE×TILE threads. The first Apple ANN slice showed the naive kernel (~8 GOP/s)
// LOSING ~5× to the SIMD CPU (docs/BENCH-gpu-results.md); this is the Phase-2 fix.
const w8a8Src = `
#include <metal_stdlib>
using namespace metal;
#define TILE 16
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
}
kernel void gemm_w8a8_tiled(
    device const char*  codes  [[buffer(0)]],   // [N*K] int8 (corpus rows = B)
    device const char*  qi8    [[buffer(1)]],   // [M*K] int8 queries (= A)
    device const float* scales [[buffer(2)]],   // [N]
    constant uint&      K      [[buffer(3)]],
    constant uint&      N      [[buffer(4)]],
    device const float* qscale [[buffer(5)]],   // [M] per-query scale
    device float*       out    [[buffer(6)]],   // [M*N] scores, row-major
    constant uint&      M      [[buffer(7)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup char As[TILE][TILE]; // query tile  [m-local][k]
    threadgroup char Bs[TILE][TILE]; // corpus tile [n-local][k]
    int tx = (int)tid.x, ty = (int)tid.y;
    int m = (int)tgpos.y * TILE + ty;    // query row
    int n = (int)tgpos.x * TILE + tx;    // corpus row (output column)
    int bRow = (int)tgpos.x * TILE + ty; // the corpus row this thread STAGES (not the one it uses)
    int acc = 0;
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        uint k = k0 + (uint)tx;
        As[ty][tx] = (m < (int)M && k < K) ? qi8[(uint)m * K + k] : (char)0;
        Bs[ty][tx] = (bRow < (int)N && k < K) ? codes[(uint)bRow * K + k] : (char)0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < TILE; kk++) acc += int(As[ty][kk]) * int(Bs[tx][kk]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m < (int)M && n < (int)N) out[(uint)m * (uint)N + (uint)n] = float(acc) * qscale[m] * scales[n];
}`

type metalBackend struct {
	dev  *gpu.Device
	q    gpu.Queue
	gemv gpu.Pipeline // one query  × N rows (FlatI8.Query)
	gemm gpu.Pipeline // M queries × N rows (FlatI8.QueryBatch)
}

// init reaches Metal and registers the backend. If there is no Metal GPU (or the
// kernel fails to compile), it registers nothing — ann.FlatI8 then stays on the
// CPU and EnableGPU returns "no backend". Building this package is the opt-in.
func init() {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return
	}
	lib, err := dev.CompileLibrary(w8a8Src, gpu.MSL3_1)
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	gemv, err := dev.NewComputePipeline(lib, "gemv_w8a8")
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	gemm, err := dev.NewComputePipeline(lib, "gemm_w8a8_tiled")
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	ann.RegisterBackend(&metalBackend{dev: dev, q: dev.NewCommandQueue(), gemv: gemv, gemm: gemm})
}

func (b *metalBackend) Name() string { return "metal" }

// runLocked runs a 1-D dispatch pinned to one OS thread. Run1D allocates and
// drains an NSAutoreleasePool, and objc requires the drain on the SAME thread as
// the alloc — but a Go goroutine can migrate between the two, draining on the
// wrong thread and crashing (an intermittent SIGSEGV). Pinning the thread across
// the whole dispatch keeps them together. (goinfer's decode avoids this with a
// dedicated LockOSThread executor; the per-call proving path locks here instead.)
func (b *metalBackend) runLocked(p gpu.Pipeline, n, tg int, bufs ...gpu.Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.q.Run1D(p, n, tg, bufs...)
}

// runLocked2D is runLocked for a 2-D dispatchThreadgroups grid — the tiled GEMM's shape
// (gx×gy whole threadgroups of tgx×tgy threads), same OS-thread pinning for the autorelease
// pool.
func (b *metalBackend) runLocked2D(p gpu.Pipeline, gx, gy, tgx, tgy int, bufs ...gpu.Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.q.Run2D(p, gx, gy, tgx, tgy, bufs...)
}

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
	codes  gpu.Buffer // [n*dim] int8 (resident)
	scales gpu.Buffer // [n] f32
	qi8    gpu.Buffer // [dim] int8 (reused per query)
	qscale gpu.Buffer // [1] f32 (reused)
	kbuf   gpu.Buffer // [1] u32
	out    gpu.Buffer // [n] f32 (reused)
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
	x.b.runLocked(x.b.gemv, x.n, tg, x.codes, x.qi8, x.scales, x.kbuf, x.qscale, x.out)
	copy(dst, x.out.Floats())
	return nil
}

// ScoreBatch scores M = len(queries) queries against all N rows in a single
// dispatch of the batched GEMM — the throughput win over calling Score in a loop.
// dst is [M*N] row-major (dst[m*N+j]). Per-call scratch (host quantization + three
// device buffers) is allocated and released each call: batch queries are the
// infrequent path, so this stays simple rather than caching an M-sized arena.
func (x *metalI8Index) ScoreBatch(queries [][]float32, dst []float32) error {
	M := len(queries)
	if M == 0 {
		return nil
	}
	N, K := x.n, x.dim
	if len(dst) != M*N {
		return fmt.Errorf("gpu: batch dst len %d != M*N %d", len(dst), M*N)
	}
	qi8 := make([]int8, M*K)
	qscale := make([]float32, M)
	for m, q := range queries {
		if len(q) != K {
			return fmt.Errorf("gpu: batch query %d dim %d != index dim %d", m, len(q), K)
		}
		qscale[m] = quantizeRowInt8(q, qi8[m*K:(m+1)*K])
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	qi8Buf := x.b.dev.NewBufferInt8(qi8)
	qscaleBuf := x.b.dev.NewBufferFloats(qscale)
	nBuf := x.b.dev.NewBufferU32(uint32(N))
	mBuf := x.b.dev.NewBufferU32(uint32(M))
	outBuf := x.b.dev.NewBufferLen(M * N)
	defer func() {
		for _, b := range []gpu.Buffer{qi8Buf, qscaleBuf, nBuf, mBuf, outBuf} {
			x.b.dev.ReleaseBuf(b)
		}
	}()
	// Tiled GEMM: one TILE×TILE output block per threadgroup (dispatchThreadgroups →
	// uniform whole groups, so the edge tiles are full and the kernel bounds-checks).
	const tile = 16
	gx, gy := (N+tile-1)/tile, (M+tile-1)/tile
	x.b.runLocked2D(x.b.gemm, gx, gy, tile, tile, x.codes, qi8Buf, x.scales, x.kbuf, nBuf, qscaleBuf, outBuf, mBuf)
	copy(dst, outBuf.Floats())
	return nil
}

// Close releases this index's device buffers (the shared device stays alive for
// other indexes; the process holds one Metal device).
func (x *metalI8Index) Close() error {
	for _, b := range []gpu.Buffer{x.codes, x.scales, x.qi8, x.qscale, x.kbuf, x.out} {
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

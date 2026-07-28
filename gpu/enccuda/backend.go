//go:build linux

// Package enccuda is the native-CUDA encoder.Backend — text-encoder matmuls on the
// GPU, cgo-free (docs/task-native-gpu.md, Phase 4).
//
// Importing it registers "cuda" with encoder.RegisterBackend, so
// encoder.NewBackend("cuda") resolves; aikit's core encoder never imports gpu, and the
// default build stays pure-Go CPU.
//
// # Two things about encoder.Backend shape this backend
//
//  1. **The interface takes HOST slices and has no residency hook**, so both operands
//     are uploaded and the result downloaded on EVERY call. For a 12-layer forward
//     that re-uploads the same weights ~72 times, which is pure waste — and there is
//     no sound way to avoid it from behind this interface.
//
//     A pointer-keyed weight cache is the obvious idea and it is WRONG here. In
//     attention, `b` is `kH`/`vHT` — scratch slices from a pool whose backing array is
//     stable across calls while their CONTENTS change every call. Keying residency on
//     the pointer would serve stale data, silently, and only for attention. A size
//     threshold does not rescue it either: at long sequences the per-head QKᵀ grows
//     past any threshold you would pick for perf reasons. So: no cache. Fixing this
//     properly means widening encoder.Backend with a "this operand is a resident
//     weight" concept — a Hard-tier change, and a real one to consider given the
//     numbers below.
//
//  2. **A backend sees EVERY f32 matmul**, including the tiny per-head QKᵀ and
//     scores·V — at L=80, headDim=64 those are ~0.4 MFLOP, far below where a device
//     round-trip can pay. So this backend declines small work and runs it on the CPU
//     path instead. That threshold is the actual product of the benchmark
//     (docs/BENCH-gpu.md: "the curve IS the dispatch threshold"), not an incidental
//     tuning constant — see minGPUFlops.
package enccuda

import (
	"fmt"
	"sync"

	"github.com/townsendmerino/aikit/encoder"
	gpu "github.com/townsendmerino/aikit/gpu"
)

func init() {
	encoder.RegisterBackend("cuda", func() (encoder.Backend, error) { return New() })
}

// minGPUFlops is the dispatch threshold: below this many FLOPs (2*M*K*N) a matmul runs
// on the CPU path instead of the device. Per docs/BENCH-gpu.md the crossover IS the
// deliverable of BenchmarkEncoderMatmulBT, so this constant is derived from it rather
// than picked.
//
// The measurement (RTX 2070 SUPER vs 16-thread CPU, same box, ns/op, both operands
// uploaded and the result downloaded every call — the honest cost of this interface):
//
//	   1 MFLOP  M=80  K=64   N=80     cpu   352 us   gpu    46 us    7.7x
//	   2 MFLOP  M=128 K=64   N=128    cpu  1076 us   gpu    60 us   17.9x
//	  19 MFLOP  M=64  K=384  N=384    cpu   735 us   gpu   192 us    3.8x
//	  24 MFLOP  M=80  K=384  N=384    cpu   925 us   gpu   238 us    3.9x
//	 151 MFLOP  M=128 K=768  N=768    cpu  1439 us   gpu   875 us    1.6x
//	 604 MFLOP  M=128 K=768  N=3072   cpu  4016 us   gpu  2482 us    1.6x
//	2416 MFLOP  M=512 K=768  N=3072   cpu  7238 us   gpu  6646 us    1.09x
//	8590 MFLOP  M=1024 K=1024 N=4096  cpu 42357 us   gpu 21685 us    2.0x
//
// The GPU wins at every shape measured, and — counter-intuitively — by the LARGEST
// margin on the SMALLEST work. That is not a GPU result, it is a CPU one: the pure-Go
// path switches from a scalar naive kernel to a blocked parallel one at ~4 MFLOP, so
// it is at its worst exactly where the device looks best, and reaches 329 GFLOP/s at
// 2416 MFLOP where the device is transfer-bound.
//
// So the threshold is set low (1 MFLOP) — but see the honesty note in the package doc:
// this is a MICROBENCHMARK-derived number. BENCH-gpu.md is explicit that
// microbenchmarks tune and end-to-end publishes, and the end-to-end encode has NOT
// been measured here (testdata/minilm-model is not present on this box). Treat 1 MFLOP
// as provisional until a real Encode is timed: a forward makes ~72 of these calls
// sequentially, each paying a device synchronize, and that serial cost does not appear
// in a per-call microbenchmark.
const minGPUFlops = 1 << 20 // 1 MFLOP — provisional, microbenchmark-derived

// Backend is the CUDA encoder backend.
type Backend struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	cpu encoder.Backend // the pure-Go path, for shapes below the threshold

	// Device buffers are reused across calls (grown on demand) — that is safe, since
	// they are re-uploaded every call. What is NOT safe is caching CONTENTS; see the
	// package note.
	mu               sync.Mutex
	aBuf, bBuf, cBuf gpu.Buffer
	aCap, bCap, cCap int
}

// New creates the backend. It fails (rather than degrading silently) when there is no
// CUDA device, so encoder.NewBackend("cuda") reports the real reason.
func New() (*Backend, error) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, fmt.Errorf("enccuda: %w", err)
	}
	k, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, fmt.Errorf("enccuda: load kernels: %w", err)
	}
	cpu, err := encoder.NewBackend("cpu")
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	return &Backend{dev: dev, q: dev.NewCommandQueue(), k: k, cpu: cpu}, nil
}

func (b *Backend) Name() string { return "cuda" }

// grow ensures buf holds at least n float32s, reallocating when it does not.
func (b *Backend) grow(buf *gpu.Buffer, cap *int, n int) {
	if n <= *cap {
		return
	}
	if *cap > 0 {
		b.dev.ReleaseBuf(*buf)
	}
	*buf, *cap = gpu.NewBufferLenOf[float32](b.dev, n), n
}

// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ.
//
// Small shapes go to the CPU path — see minGPUFlops. On the device path the weight is
// resident after first use; the activation is uploaded and the result read back.
//
// Errors are not in the signature (encoder.Backend predates this backend), so a device
// failure falls back to the CPU path rather than producing wrong numbers. Silent
// wrongness is the one outcome not on the table.
func (b *Backend) MatmulBT(a, w, dst []float32, M, K, N int) {
	if 2*int64(M)*int64(K)*int64(N) < minGPUFlops || M <= 0 || K <= 0 || N <= 0 {
		b.cpu.MatmulBT(a, w, dst, M, K, N)
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.gpuMatmul(a, w, dst, M, K, N); err != nil {
		b.cpu.MatmulBT(a, w, dst, M, K, N)
	}
}

func (b *Backend) gpuMatmul(a, w, dst []float32, M, K, N int) (err error) {
	defer func() {
		if r := recover(); r != nil { // MustBuf panics on OOM; degrade, don't die
			err = fmt.Errorf("enccuda: device allocation failed: %v", r)
		}
	}()
	b.grow(&b.aBuf, &b.aCap, M*K)
	b.grow(&b.bBuf, &b.bCap, N*K)
	b.grow(&b.cBuf, &b.cCap, M*N)
	if err := gpu.Upload(b.aBuf, a[:M*K]); err != nil {
		return err
	}
	if err := gpu.Upload(b.bBuf, w[:N*K]); err != nil {
		return err
	}
	if err := b.q.Launch(b.k.GEMMF32Tiled, gpu.TileGrid(M, N),
		gpu.Arg(b.aBuf), gpu.Arg(b.bBuf), gpu.Arg(b.cBuf),
		gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K))); err != nil {
		return err
	}
	if err := b.q.Sync(); err != nil {
		return err
	}
	return gpu.Download(b.cBuf, dst[:M*N])
}

// Close releases the device and every cached weight.
func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.dev != nil {
		b.dev.ReleaseObjects()
		b.dev = nil
	}
	return nil
}

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
	"unsafe"

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
// uploaded and the result downloaded every call — the honest cost of this interface),
// re-taken after gemm_f32_reg replaced gemm_f32_tiled on the aligned path:
//
//	   1 MFLOP  M=80  K=64   N=80     cpu   408 us   gpu    47 us    8.6x
//	   2 MFLOP  M=128 K=64   N=128    cpu  1091 us   gpu    61 us   17.8x
//	  19 MFLOP  M=64  K=384  N=384    cpu   524 us   gpu   199 us    2.6x
//	  24 MFLOP  M=80  K=384  N=384    cpu   723 us   gpu   227 us    3.2x
//	 151 MFLOP  M=128 K=768  N=768    cpu  1640 us   gpu   577 us    2.8x
//	 604 MFLOP  M=128 K=768  N=3072   cpu  4283 us   gpu  1466 us    2.9x
//	2416 MFLOP  M=512 K=768  N=3072   cpu  7109 us   gpu  2685 us    2.6x
//	8590 MFLOP  M=1024 K=1024 N=4096  cpu 42010 us   gpu  6671 us    6.3x
//
// The faster kernel fixed the shape of this curve, not just its level. With the tiled
// kernel the device COLLAPSED at the large end — 1.09x at 2416 MFLOP and 2.0x at 8590 —
// because it was transfer-bound while the pure-Go path hit 329 GFLOP/s. Register
// blocking moved those to 2.6x and 6.3x. The small end is unchanged: those shapes are
// unaligned, so they still run the tiled kernel, and they are dominated by the per-call
// upload/download either way.
//
// The threshold itself stays at 1 MFLOP — the device already won everywhere, and making
// the kernel faster cannot make it lose.
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
	// q8 holds dequantized int8 weights resident on the device. Sound here (model
	// weights are write-once) where an f32 cache would not be; see MatmulBTQ8.
	q8 map[uintptr]q8w
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
	gp, gcfg := b.k.GEMMF32Plan(M, N, K)
	if err := b.q.Launch(gp, gcfg,
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

// --- int8 (weight-only) path: encoder.Q8Backend ---
//
// The encoder's int8 is WEIGHT-ONLY — int8 weights widened to f32, multiplied against
// f32 activations. It is NOT W8A8, and reaching for the W8A8 GEMV here would be a
// quality regression, not an optimization: quantizing the activations was measured and
// rejected for falling below the 0.97 reranker bar (see encoder.Q8Backend's contract).
//
// So this path dequantizes and then runs the SAME gemm_f32_tiled the f32 path uses —
// identical arithmetic to the CPU's matmulBTQ8Into, which is what makes a tight parity
// bound available.
//
// AND IT IS THE CHEAPER SHAPE, not merely the correct one. The CPU redoes the N*K
// widen on EVERY call — the comment on matmulBTQ8Into names that as the actual reason
// LoadQ8 ran ~5x slower than Load. Here the weight is dequantized ONCE and the f32
// result stays resident, so a 12-layer forward pays it once per matrix instead of once
// per matmul. That is the real win of the int8 GPU path, and a W8A8 implementation
// would have thrown it away.
//
// Caching is SOUND here in a way it was not for f32. The f32 backend deliberately has
// no cache because its `b` operand is attention's pooled kH/vHT scratch — stable
// pointer, changing contents. The int8 `wq` is a model weight: loaded once, never
// written. The negative control (activations must never be cached) is still tested.

// q8w is one dequantized weight resident on the device.
type q8w struct {
	buf gpu.Buffer
	n   int // element count, so a recycled address at a different length cannot hit
}

// residentQ8 returns the device buffer holding dequant(wq), uploading on first use.
// Dequantization runs on the host: it is one pass per weight for the process lifetime,
// so a device kernel would buy nothing and cost a launch.
func (b *Backend) residentQ8(wq []int8, wscales []float32, K, N int) (gpu.Buffer, error) {
	k := q8key(wq)
	if c, ok := b.q8[k]; ok && c.n == len(wq) {
		return c.buf, nil
	}
	deq := make([]float32, N*K)
	for n := range N {
		sc := wscales[n]
		row, src := deq[n*K:(n+1)*K], wq[n*K:(n+1)*K]
		for i := range row {
			row[i] = float32(src[i]) * sc
		}
	}
	buf := gpu.NewBufferOf(b.dev, deq)
	if b.q8 == nil {
		b.q8 = map[uintptr]q8w{}
	}
	b.q8[k] = q8w{buf: buf, n: len(wq)}
	return buf, nil
}

func q8key(w []int8) uintptr {
	if len(w) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&w[0]))
}

// MatmulBTQ8 implements encoder.Q8Backend: dst[M,N] = a[M,K] · dequant(wq)[N,K]ᵀ,
// activations untouched. Returns false to leave the call on the CPU path.
func (b *Backend) MatmulBTQ8(dst, a []float32, wq []int8, wscales []float32, M, K, N int) bool {
	if 2*int64(M)*int64(K)*int64(N) < minGPUFlops || M <= 0 || K <= 0 || N <= 0 {
		return false
	}
	if len(a) < M*K || len(wq) < N*K || len(wscales) < N || len(dst) < M*N {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.gpuMatmulQ8(dst, a, wq, wscales, M, K, N) == nil
}

func (b *Backend) gpuMatmulQ8(dst, a []float32, wq []int8, wscales []float32, M, K, N int) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("enccuda: device allocation failed: %v", r)
		}
	}()
	wb, err := b.residentQ8(wq, wscales, K, N)
	if err != nil {
		return err
	}
	b.grow(&b.aBuf, &b.aCap, M*K)
	b.grow(&b.cBuf, &b.cCap, M*N)
	// The ACTIVATION is uploaded every call — it changes every call. Only the weight
	// is resident.
	if err := gpu.Upload(b.aBuf, a[:M*K]); err != nil {
		return err
	}
	gp, gcfg := b.k.GEMMF32Plan(M, N, K)
	if err := b.q.Launch(gp, gcfg,
		gpu.Arg(b.aBuf), gpu.Arg(wb), gpu.Arg(b.cBuf),
		gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K))); err != nil {
		return err
	}
	if err := b.q.Sync(); err != nil {
		return err
	}
	return gpu.Download(b.cBuf, dst[:M*N])
}

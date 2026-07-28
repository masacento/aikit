//go:build darwin

// Package encmetal is the native-Metal encoder.Backend — text-encoder matmuls on Apple
// GPUs, cgo-free (docs/task-native-gpu.md, Phase 4). It is the Apple mirror of
// gpu/enccuda.
//
// Importing it registers "metal" with encoder.RegisterBackend, so
// encoder.NewBackend("metal") resolves; aikit's core encoder never imports gpu, and the
// default build stays pure-Go CPU.
//
// # Two things about encoder.Backend shape this backend (identical to enccuda's)
//
//  1. **The interface takes HOST slices and has no residency hook**, so both operands are
//     copied into resident buffers on EVERY call. On Metal that copy is a memcpy into a
//     shared (UMA) MTLBuffer, NOT a PCIe upload — cheap — but it is still a copy, and a
//     12-layer forward re-copies the same weights ~72 times.
//
//     A pointer-keyed weight cache is the obvious idea and it is WRONG here. In attention,
//     `b` is `kH`/`vHT` — scratch slices from a pool whose backing array is stable across
//     calls while their CONTENTS change every call. Keying residency on the pointer would
//     serve stale data, silently, and only for attention. A size threshold does not rescue
//     it either: at long sequences the per-head QKᵀ grows past any threshold you would pick
//     for perf. So: no cache. TestEncMetal_noStaleOperands is the regression gate. Fixing
//     this properly means widening encoder.Backend with a "this operand is a resident
//     weight" concept — a Hard-tier change.
//
//  2. **A backend sees EVERY f32 matmul**, including the tiny per-head QKᵀ and scores·V,
//     far below where a device round-trip pays. So this backend declines small work to the
//     CPU path. That threshold is the product of the benchmark (docs/BENCH-gpu.md: "the
//     curve IS the dispatch threshold") — see minGPUFlops.
//
// The one Metal-vs-CUDA difference that matters here: UMA means the "upload" is a
// `copy()` into the buffer's `Floats()` view and the "download" is a `copy()` back out,
// with no explicit device transfer and no data race of the kind the CUDA upload path had
// to synchronize against (an MTLBuffer is a live Go slice). Each device dispatch runs
// under one runtime.LockOSThread, because Run2D allocs+drains an NSAutoreleasePool that
// objc requires on a single thread.
package encmetal

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/encoder"
	gpu "github.com/townsendmerino/aikit/gpu"
)

func init() {
	encoder.RegisterBackend("metal", func() (encoder.Backend, error) { return New() })
}

// minGPUFlops is the dispatch threshold: below this many FLOPs (2*M*K*N) a matmul runs on
// the CPU path instead of the device. Per docs/BENCH-gpu.md the crossover IS the
// deliverable of BenchmarkEncoderMatmulBT, so this is DERIVED from the sweep, not picked —
// and deliberately NOT copied from enccuda's 1 MFLOP: the CPU, the transfer cost (UMA vs
// PCIe) and the crossover are all different here.
//
// This backend runs the PRODUCTION gemm_f32_sg (simdgroup_matrix), NOT the correctness-first
// gemm_f32_tiled — the kernel is ~2.3–2.8× faster at these shapes (778 vs 339 GFLOP/s at
// 2416 MFLOP, 967 vs 346 at 8590; see BenchmarkMetalGEMMF32), which moved the crossover
// DOWN ~4× from where the tiled kernel put it (~2.1 GFLOP → ~0.5 GFLOP).
//
// Measurement (Apple M1 Pro, GPU vs 10-core pure-Go CPU, same box, best-of ns/op, both
// operands copied into UMA buffers and the result copied back every call — the honest cost
// of this interface). GFLOP/s in parentheses:
//
//	   1 MFLOP  M=80   K=64   N=80    cpu 227us(3.6)   gpu 255us(3.2)   0.89x   CPU
//	   2 MFLOP  M=128  K=64   N=128   cpu 579us(3.6)   gpu 267us(7.9)   2.17x   (CPU-artifact)
//	  19 MFLOP  M=64   K=384  N=384   cpu 269us(70)    gpu 427us(44)    0.63x   CPU
//	  24 MFLOP  M=80   K=384  N=384   cpu 335us(70)    gpu 343us(69)    0.98x   ~tie
//	 151 MFLOP  M=128  K=768  N=768   cpu 599us(252)   gpu 771us(196)   0.78x   CPU
//	 604 MFLOP  M=128  K=768  N=3072  cpu 2.34ms(258)  gpu 1.31ms(463)  1.80x   GPU
//	2416 MFLOP  M=512  K=768  N=3072  cpu 7.96ms(303)  gpu 4.35ms(556)  1.83x   GPU
//	8590 MFLOP  M=1024 K=1024 N=4096  cpu 34.2ms(251)  gpu 9.85ms(872)  3.47x   GPU
//
// The crossover is now between 151 MFLOP (GPU loses, 0.78×) and 604 MFLOP (GPU wins, 1.80×),
// so the threshold sits just below the proven win at 604. The M1-Pro CPU is still strong
// (~300 GFLOP/s in its blocked path), but gemm_f32_sg reaches ~872 GFLOP/s — enough of an
// edge to pay once the size amortizes the ~250us fixed dispatch+sync floor. The lone
// small-shape "win" (2 MFLOP, 2.17×) is a CPU artifact — the pure-Go path switches
// scalar→blocked at ~4 MFLOP — and is pure GPU floor; routing a forward's many small matmuls
// each to that floor would be slower in aggregate, so the threshold keeps them on the CPU.
//
// HONESTY NOTE (BENCH-gpu.md: microbenchmarks tune, end-to-end publishes):
// TestEncMetal_endToEnd times a REAL MiniLM Encode. Even at this lower crossover a single
// short-sequence forward (small L, hidden 384) has no matmul above the threshold, so the
// backend correctly stays entirely on the CPU and the numerics are identical; the device
// pays for batched or larger-model encode.
const minGPUFlops = 1 << 29 // ~537 MFLOP — derived from the sweep above (simdgroup crossover)

// Backend is the Metal encoder backend.
type Backend struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	cpu encoder.Backend // the pure-Go path, for shapes below the threshold

	// Device buffers are reused across calls (grown on demand) — safe, since they are
	// re-copied every call. Caching CONTENTS is what is NOT safe; see the package note.
	mu               sync.Mutex
	aBuf, bBuf, cBuf gpu.Buffer
	aCap, bCap, cCap int
	mS, nS, kS       gpu.Buffer // reusable scalar buffers (Metal has no by-value arg)
}

// New creates the backend. It fails (rather than degrading silently) when there is no
// Metal device, so encoder.NewBackend("metal") reports the real reason.
func New() (*Backend, error) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, fmt.Errorf("encmetal: %w", err)
	}
	k, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, fmt.Errorf("encmetal: load kernels: %w", err)
	}
	cpu, err := encoder.NewBackend("cpu")
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	b := &Backend{dev: dev, q: dev.NewCommandQueue(), k: k, cpu: cpu}
	b.mS, b.nS, b.kS = dev.NewBufferU32(0), dev.NewBufferU32(0), dev.NewBufferU32(0)
	return b, nil
}

func (b *Backend) Name() string { return "metal" }

// grow ensures buf holds at least n float32s, reallocating when it does not.
func (b *Backend) grow(buf *gpu.Buffer, cap *int, n int) {
	if n <= *cap {
		return
	}
	if *cap > 0 {
		b.dev.ReleaseBuf(*buf)
	}
	*buf, *cap = b.dev.NewBufferLen(n), n
}

// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ.
//
// Small shapes go to the CPU path — see minGPUFlops. Errors are not in the signature
// (encoder.Backend predates this backend), so a device failure falls back to the CPU path
// rather than producing wrong numbers. Silent wrongness is the one outcome not on the table.
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
			err = fmt.Errorf("encmetal: device allocation failed: %v", r)
		}
	}()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.grow(&b.aBuf, &b.aCap, M*K)
	b.grow(&b.bBuf, &b.bCap, N*K)
	b.grow(&b.cBuf, &b.cCap, M*N)
	copy(b.aBuf.Floats()[:M*K], a[:M*K]) // UMA "upload": memcpy into a shared buffer
	copy(b.bBuf.Floats()[:N*K], w[:N*K])
	b.mS.SetU32(uint32(int32(M)))
	b.nS.SetU32(uint32(int32(N)))
	b.kS.SetU32(uint32(int32(K)))
	gx, gy, tgx, tgy := gpu.SGDims(M, N)
	b.q.Run2D(b.k.GEMMF32SG, gx, gy, tgx, tgy, b.aBuf, b.bBuf, b.cBuf, b.mS, b.nS, b.kS)
	copy(dst[:M*N], b.cBuf.Floats()[:M*N]) // UMA "download" (Run2D already waited)
	return nil
}

// Close releases the device and every reused buffer (buffers first, so no in-flight work
// references freed memory; the last Run2D already waited).
func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.dev != nil {
		b.dev.ReleaseAll()
		b.dev.ReleaseObjects()
		b.dev = nil
	}
	return nil
}

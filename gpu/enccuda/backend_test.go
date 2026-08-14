//go:build linux

package enccuda

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/encoder"
)

func mustBackend(t testing.TB) *Backend {
	t.Helper()
	b, err := New()
	if err != nil {
		t.Skipf("no CUDA encoder backend: %v", err)
	}
	return b
}

func randF(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

// refMatmulBT is the float64 ground truth: dst[M,N] = a[M,K] · b[N,K]ᵀ.
func refMatmulBT(a, w []float32, M, K, N int) []float64 {
	out := make([]float64, M*N)
	for m := range M {
		for n := range N {
			var acc float64
			for k := range K {
				acc += float64(a[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = acc
		}
	}
	return out
}

// TestEncCUDA_parity gates the backend against a float64 reference across shapes on
// BOTH sides of the dispatch threshold, so the CPU-fallback path and the device path
// are each covered. A backend that silently returned garbage above the threshold, or
// that mis-routed, would show here.
func TestEncCUDA_parity(t *testing.T) {
	b := mustBackend(t)
	defer func() { _ = b.Close() }()
	rng := rand.New(rand.NewSource(3))
	for _, s := range []struct {
		name    string
		M, K, N int
	}{
		{"tiny/cpu-path", 8, 32, 32},
		{"small/cpu-path", 80, 64, 80},
		{"bert-qkv/L128", 128, 768, 768},
		{"bert-fc1/L128", 128, 768, 3072},
		{"batch/L512", 512, 768, 3072},
	} {
		t.Run(s.name, func(t *testing.T) {
			a, w := randF(rng, s.M*s.K), randF(rng, s.N*s.K)
			dst := make([]float32, s.M*s.N)
			b.MatmulBT(a, w, dst, s.M, s.K, s.N)
			ref := refMatmulBT(a, w, s.M, s.K, s.N)
			worst := 0.0
			for i := range ref {
				den := math.Abs(ref[i])
				if den < 1 {
					den = 1
				}
				if r := math.Abs(float64(dst[i])-ref[i]) / den; r > worst {
					worst = r
				}
			}
			onGPU := 2*int64(s.M)*int64(s.K)*int64(s.N) >= minGPUFlops
			// Bound from the analytic f32 error: a K-term dot drifts ~sqrt(K)*eps
			// relative against float64, ~5e-5 at K=768. 2e-4 clears that and still
			// rejects anything structural.
			if worst > 2e-4 {
				t.Fatalf("worst relative Δ %.3g (gpu=%v)", worst, onGPU)
			}
			t.Logf("M=%d K=%d N=%d gpu=%v worst relative Δ %.3g", s.M, s.K, s.N, onGPU, worst)
		})
	}
}

// TestEncCUDA_noStaleOperands is the regression gate for the trap this backend walked
// into and backed out of: a pointer-keyed weight cache.
//
// In attention, `b` is a POOLED SCRATCH slice — same backing array every call, different
// contents. A cache keyed on the pointer would serve the first call's bytes forever,
// silently, and only for attention. This reproduces that exact shape: same slice header,
// mutated contents, two calls, results must differ.
func TestEncCUDA_noStaleOperands(t *testing.T) {
	b := mustBackend(t)
	defer func() { _ = b.Close() }()
	rng := rand.New(rand.NewSource(4))
	const M, K, N = 128, 768, 768
	a := randF(rng, M*K)
	w := randF(rng, N*K) // the "scratch" operand — reused, rewritten in place
	d1, d2 := make([]float32, M*N), make([]float32, M*N)

	b.MatmulBT(a, w, d1, M, K, N)
	copy(w, randF(rng, N*K)) // SAME slice, new contents — exactly what attention does
	b.MatmulBT(a, w, d2, M, K, N)

	same := true
	for i := range d1 {
		if d1[i] != d2[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("rewriting the operand in place changed nothing — the backend is caching contents by pointer and serving stale data")
	}
	// And the second result must be RIGHT, not merely different.
	ref := refMatmulBT(a, w, M, K, N)
	worst := 0.0
	for i := range ref {
		den := math.Abs(ref[i])
		if den < 1 {
			den = 1
		}
		if r := math.Abs(float64(d2[i])-ref[i]) / den; r > worst {
			worst = r
		}
	}
	if worst > 2e-4 {
		t.Fatalf("after in-place rewrite, worst relative Δ %.3g", worst)
	}
	t.Log("in-place operand rewrite is observed correctly (no pointer-keyed content cache)")
}

// TestEncCUDA_registersWithEncoder proves the registration half: importing this package
// must make encoder.NewBackend("cuda") resolve. Before Phase 4's wiring a resolved
// backend did nothing at all, so this pairs with encoder's own dispatch gates.
func TestEncCUDA_registersWithEncoder(t *testing.T) {
	be, err := encoder.NewBackend("cuda")
	if err != nil {
		t.Skipf("NewBackend(cuda): %v", err)
	}
	defer func() { _ = be.Close() }()
	if be.Name() != "cuda" {
		t.Fatalf("NewBackend(cuda).Name() = %q", be.Name())
	}
	t.Log(`encoder.NewBackend("cuda") resolves to this backend`)
}

// BenchmarkEncoderMatmulBT is the crossover sweep. Per docs/BENCH-gpu.md the PRODUCT of
// this benchmark is the dispatch threshold (minGPUFlops), not a speedup headline: it
// reports CPU and GPU at the same shapes on the same box so the crossing point is
// visible. GPU rows include the activation upload and result download, because the
// encoder.Backend interface forces them on every call — excluding them would measure a
// kernel this backend cannot actually offer.
func BenchmarkEncoderMatmulBT(b *testing.B) {
	gb := mustBackend(b)
	defer func() { _ = gb.Close() }()
	cpu, err := encoder.NewBackend("cpu")
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = cpu.Close() }()
	rng := rand.New(rand.NewSource(9))
	for _, s := range []struct{ M, K, N int }{
		{80, 64, 80},     // per-head QK^T at L=80 — the smallest matmul in a forward
		{128, 64, 128},   // per-head QK^T at L=128
		{64, 384, 384},   // short sequence
		{80, 384, 384},   // MiniLM-ish, short sequence
		{128, 768, 768},  // BERT-base QKV, L=128
		{128, 768, 3072}, // BERT-base FC1
		{512, 768, 3072}, // batched
		{1024, 1024, 4096},
	} {
		a, w := randF(rng, s.M*s.K), randF(rng, s.N*s.K)
		dst := make([]float32, s.M*s.N)
		mf := 2.0 * float64(s.M) * float64(s.K) * float64(s.N) / 1e6
		name := fmt.Sprintf("M%d_K%d_N%d_%.0fMFLOP", s.M, s.K, s.N, mf)
		b.Run(name+"/cpu", func(b *testing.B) {
			for range b.N {
				cpu.MatmulBT(a, w, dst, s.M, s.K, s.N)
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
		b.Run(name+"/gpu", func(b *testing.B) {
			// force the device path regardless of the threshold, so the sweep shows
			// where the crossing actually is rather than where we guessed it is
			if err := gb.gpuMatmul(a, w, dst, s.M, s.K, s.N); err != nil {
				b.Fatalf("gpuMatmul warm: %v", err)
			}
			b.ResetTimer()
			for range b.N {
				if err := gb.gpuMatmul(a, w, dst, s.M, s.K, s.N); err != nil {
					b.Fatalf("gpuMatmul: %v", err)
				}
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
	}
}

// --- int8 (weight-only) path ---

// cpuQ8Ref reproduces encoder/linalg_q8.go's matmulBTQ8Into exactly: widen the int8
// weights to f32, then an f32 matmul. Activations untouched. In float64 so the GPU is
// gated against ground truth rather than against another f32 accumulation.
func cpuQ8Ref(a []float32, wq []int8, wscales []float32, M, K, N int) []float64 {
	out := make([]float64, M*N)
	for m := range M {
		for n := range N {
			sc := float64(wscales[n])
			var acc float64
			for k := range K {
				acc += float64(a[m*K+k]) * (float64(wq[n*K+k]) * sc)
			}
			out[m*N+n] = acc
		}
	}
	return out
}

func randQ8(rng *rand.Rand, n int) []int8 {
	out := make([]int8, n)
	for i := range out {
		out[i] = int8(rng.Intn(255) - 127)
	}
	return out
}

// TestEncCUDA_q8Parity gates the int8 path against the weight-only reference on both
// sides of the dispatch threshold.
//
// It also pins the thing most likely to be "optimized" wrong later: this path must NOT
// quantize the activations. A W8A8 implementation would still look plausible and would
// still produce roughly-right numbers — it would just quietly fall below the 0.97
// reranker bar that TestModelQ8_cosineMatchesF32 enforces. The tight bound here is what
// separates the two: weight-only dequant is exact, so only the f32 accumulation
// reassociates, and a quantized-activation kernel cannot fit inside 2e-4.
func TestEncCUDA_q8Parity(t *testing.T) {
	b := mustBackend(t)
	defer func() { _ = b.Close() }()
	rng := rand.New(rand.NewSource(11))
	for _, s := range []struct {
		name    string
		M, K, N int
	}{
		{"tiny/declined", 8, 32, 32},
		{"bert-qkv/L128", 128, 768, 768},
		{"bert-fc1/L128", 128, 768, 3072},
	} {
		t.Run(s.name, func(t *testing.T) {
			a := randF(rng, s.M*s.K)
			wq := randQ8(rng, s.N*s.K)
			ws := make([]float32, s.N)
			for i := range ws {
				ws[i] = 0.002 + float32(i%13)*0.0001
			}
			dst := make([]float32, s.M*s.N)
			handled := b.MatmulBTQ8(dst, a, wq, ws, s.M, s.K, s.N)
			wantHandled := 2*int64(s.M)*int64(s.K)*int64(s.N) >= minGPUFlops
			if handled != wantHandled {
				t.Fatalf("MatmulBTQ8 handled=%v, want %v for %d FLOP", handled, wantHandled, 2*s.M*s.K*s.N)
			}
			if !handled {
				t.Logf("declined to CPU as expected (%d FLOP < %d)", 2*s.M*s.K*s.N, minGPUFlops)
				return
			}
			ref := cpuQ8Ref(a, wq, ws, s.M, s.K, s.N)
			worst := 0.0
			for i := range ref {
				den := math.Abs(ref[i])
				if den < 1 {
					den = 1
				}
				if r := math.Abs(float64(dst[i])-ref[i]) / den; r > worst {
					worst = r
				}
			}
			if worst > 2e-4 {
				t.Fatalf("worst relative Δ %.3g vs the weight-only reference — is this quantizing activations?", worst)
			}
			t.Logf("M=%d K=%d N=%d worst relative Δ %.3g", s.M, s.K, s.N, worst)
		})
	}
}

// TestEncCUDA_q8WeightResidency proves the cache is real AND correctly keyed, and — the
// part that matters — that it caches only the WEIGHT.
//
// The f32 backend has no cache because its `b` operand is pooled scratch (see
// TestEncCUDA_noStaleOperands). The int8 `wq` is a model weight, written once, so
// residency is sound. But the ACTIVATION still changes every call, and caching it would
// be the same bug wearing different clothes — so that is asserted separately.
func TestEncCUDA_q8WeightResidency(t *testing.T) {
	b := mustBackend(t)
	defer func() { _ = b.Close() }()
	rng := rand.New(rand.NewSource(12))
	const M, K, N = 128, 768, 768
	a1, a2 := randF(rng, M*K), randF(rng, M*K)
	w1, w2 := randQ8(rng, N*K), randQ8(rng, N*K)
	ws := make([]float32, N)
	for i := range ws {
		ws[i] = 0.003
	}
	d1, d2, d3 := make([]float32, M*N), make([]float32, M*N), make([]float32, M*N)

	b.MatmulBTQ8(d1, a1, w1, ws, M, K, N)
	if got := len(b.q8); got != 1 {
		t.Fatalf("after 1 weight: %d cached, want 1", got)
	}
	b.MatmulBTQ8(d2, a1, w2, ws, M, K, N)
	if got := len(b.q8); got != 2 {
		t.Fatalf("after 2 distinct weights: %d cached, want 2 — a key collision would silently reuse the wrong weights", got)
	}
	b.MatmulBTQ8(d3, a1, w1, ws, M, K, N) // must HIT
	if got := len(b.q8); got != 2 {
		t.Fatalf("re-using w1 grew the cache to %d — residency is not working", got)
	}
	for i := range d1 {
		if d1[i] != d3[i] {
			t.Fatalf("cached weight gave a different result at %d", i)
		}
	}
	distinct := false
	for i := range d1 {
		if d1[i] != d2[i] {
			distinct = true
			break
		}
	}
	if !distinct {
		t.Fatal("two different weights produced identical output — the cache is serving the wrong buffer")
	}

	// The negative control: a NEW activation against the SAME (cached) weight must
	// change the result, and be correct. Caching the activation would pass every
	// assertion above and fail this one.
	d4 := make([]float32, M*N)
	b.MatmulBTQ8(d4, a2, w1, ws, M, K, N)
	same := true
	for i := range d1 {
		if d1[i] != d4[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("a different activation gave identical output — the ACTIVATION is being cached")
	}
	ref := cpuQ8Ref(a2, w1, ws, M, K, N)
	worst := 0.0
	for i := range ref {
		den := math.Abs(ref[i])
		if den < 1 {
			den = 1
		}
		if r := math.Abs(float64(d4[i])-ref[i]) / den; r > worst {
			worst = r
		}
	}
	if worst > 2e-4 {
		t.Fatalf("new activation against a cached weight: worst relative Δ %.3g", worst)
	}
	t.Log("weight residency sound; activation never cached")
}

// BenchmarkEncoderMatmulBTQ8 measures the int8 path against the CPU's
// matmulBTQ8Into-equivalent. The GPU advantage here should EXCEED the f32 one: the CPU
// redoes the N*K weight widen on every call, while the device does it once and keeps
// the f32 weight resident.
func BenchmarkEncoderMatmulBTQ8(b *testing.B) {
	gb := mustBackend(b)
	defer func() { _ = gb.Close() }()
	rng := rand.New(rand.NewSource(13))
	for _, s := range []struct{ M, K, N int }{
		{128, 768, 768},
		{128, 768, 3072},
		{512, 768, 3072},
	} {
		a := randF(rng, s.M*s.K)
		wq := randQ8(rng, s.N*s.K)
		ws := make([]float32, s.N)
		for i := range ws {
			ws[i] = 0.003
		}
		dst := make([]float32, s.M*s.N)
		deq := make([]float32, s.N*s.K)
		mf := 2.0 * float64(s.M) * float64(s.K) * float64(s.N) / 1e6
		name := fmt.Sprintf("M%d_K%d_N%d_%.0fMFLOP", s.M, s.K, s.N, mf)
		b.Run(name+"/cpu", func(b *testing.B) {
			for range b.N {
				// what matmulBTQ8Into does: widen every call, then f32 matmul
				for n := range s.N {
					sc := ws[n]
					row, src := deq[n*s.K:(n+1)*s.K], wq[n*s.K:(n+1)*s.K]
					for i := range row {
						row[i] = float32(src[i]) * sc
					}
				}
				gb.cpu.MatmulBT(a, deq, dst, s.M, s.K, s.N)
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
		b.Run(name+"/gpu", func(b *testing.B) {
			gb.MatmulBTQ8(dst, a, wq, ws, s.M, s.K, s.N) // warm + resident
			b.ResetTimer()
			for range b.N {
				gb.MatmulBTQ8(dst, a, wq, ws, s.M, s.K, s.N)
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
	}
}

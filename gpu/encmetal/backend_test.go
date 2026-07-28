//go:build darwin

package encmetal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/encoder"
)

func mustBackend(t testing.TB) *Backend {
	t.Helper()
	b, err := New()
	if err != nil {
		t.Skipf("no Metal encoder backend: %v", err)
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

// TestEncMetal_parity gates the backend against a float64 reference across shapes on BOTH
// sides of the dispatch threshold, so the CPU-fallback path and the device path are each
// covered. A backend that silently returned garbage above the threshold, or that
// mis-routed, would show here.
func TestEncMetal_parity(t *testing.T) {
	b := mustBackend(t)
	defer b.Close()
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
			// Bound from the analytic f32 error: a K-term dot drifts ~sqrt(K)*eps relative
			// against float64, ~5e-5 at K=768. 2e-4 clears that and still rejects anything
			// structural.
			if worst > 2e-4 {
				t.Fatalf("worst relative Δ %.3g (gpu=%v)", worst, onGPU)
			}
			t.Logf("M=%d K=%d N=%d gpu=%v worst relative Δ %.3g", s.M, s.K, s.N, onGPU, worst)
		})
	}
}

// TestEncMetal_noStaleOperands is the regression gate for the trap enccuda walked into and
// backed out of: a pointer-keyed weight cache.
//
// In attention, `b` is a POOLED SCRATCH slice — same backing array every call, different
// contents. A cache keyed on the pointer would serve the first call's bytes forever,
// silently, and only for attention. This reproduces that exact shape: same slice header,
// mutated contents, two calls, results must differ AND the second must be correct.
func TestEncMetal_noStaleOperands(t *testing.T) {
	b := mustBackend(t)
	defer b.Close()
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

// TestEncMetal_registersWithEncoder proves the registration half: importing this package
// must make encoder.NewBackend("metal") resolve.
func TestEncMetal_registersWithEncoder(t *testing.T) {
	be, err := encoder.NewBackend("metal")
	if err != nil {
		t.Skipf("NewBackend(metal): %v", err)
	}
	defer be.Close()
	if be.Name() != "metal" {
		t.Fatalf("NewBackend(metal).Name() = %q", be.Name())
	}
	t.Log(`encoder.NewBackend("metal") resolves to this backend`)
}

// TestEncMetal_endToEnd is the number the NVIDIA box could not get: a REAL MiniLM Encode
// on CPU vs the Metal backend, same model, same text. It gates parity (the whole forward
// through the seam must match the pure-Go forward by cosine) AND reports wall time — the
// end-to-end figure BENCH-gpu.md says is the one that publishes, capturing the ~72
// sequential backend calls each paying a device synchronize that a per-call microbenchmark
// cannot show.
func TestEncMetal_endToEnd(t *testing.T) {
	const dir = "../../testdata/minilm-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no MiniLM model at %s — fetch via scripts/README.md", dir)
	}
	be := mustBackend(t)
	defer be.Close()

	cpuB, err := encoder.LoadBERT(dir)
	if err != nil {
		t.Fatalf("LoadBERT(cpu): %v", err)
	}
	defer cpuB.Close()
	gpuB, err := encoder.LoadBERT(dir)
	if err != nil {
		t.Fatalf("LoadBERT(gpu): %v", err)
	}
	defer gpuB.Close()
	gpuB.UseBackend(be)

	const text = "The quick brown fox jumps over the lazy dog, and then encodes itself into a dense vector."
	cpuEmb, err := cpuB.Encode(text)
	if err != nil {
		t.Fatalf("cpu Encode: %v", err)
	}
	gpuEmb, err := gpuB.Encode(text)
	if err != nil {
		t.Fatalf("gpu Encode: %v", err)
	}
	if len(cpuEmb) != len(gpuEmb) {
		t.Fatalf("embedding len %d != %d", len(gpuEmb), len(cpuEmb))
	}
	var dot, na, nb float64
	for i := range cpuEmb {
		dot += float64(cpuEmb[i]) * float64(gpuEmb[i])
		na += float64(cpuEmb[i]) * float64(cpuEmb[i])
		nb += float64(gpuEmb[i]) * float64(gpuEmb[i])
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
	if cos < 1-1e-5 {
		t.Fatalf("end-to-end CPU≡Metal cosine %.9f below 1-1e-5 — the backend changed the forward's numerics", cos)
	}

	// Wall-clock: median of a few runs each (a forward is ~72 sequential MatmulBT calls).
	timeEncode := func(b *encoder.BERT) time.Duration {
		for range 2 { // warm
			_, _ = b.Encode(text)
		}
		const reps = 20
		best := time.Hour
		for range reps {
			t0 := time.Now()
			_, _ = b.Encode(text)
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}
	cpuT := timeEncode(cpuB)
	gpuT := timeEncode(gpuB)
	t.Logf("end-to-end MiniLM Encode: CPU %v, Metal %v (%.2fx), cosine %.9f",
		cpuT, gpuT, float64(cpuT)/float64(gpuT), cos)
	t.Log("NOTE: end-to-end publishes (BENCH-gpu.md). This single-text forward is short-sequence; " +
		"the per-call sweep (BenchmarkEncoderMatmulBT) shows where the device pays.")
}

// BenchmarkEncoderMatmulBT is the crossover sweep — the PRODUCT is the dispatch threshold
// (minGPUFlops), not a speedup headline. It reports CPU and GPU at the same shapes on the
// same box so the crossing point is visible. GPU rows include the UMA copy in and out,
// because the encoder.Backend interface forces them on every call.
func BenchmarkEncoderMatmulBT(b *testing.B) {
	gb := mustBackend(b)
	defer gb.Close()
	cpu, err := encoder.NewBackend("cpu")
	if err != nil {
		b.Fatal(err)
	}
	defer cpu.Close()
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
			// force the device path regardless of the threshold, so the sweep shows where
			// the crossing actually is rather than where we guessed it
			_ = gb.gpuMatmul(a, w, dst, s.M, s.K, s.N)
			b.ResetTimer()
			for range b.N {
				_ = gb.gpuMatmul(a, w, dst, s.M, s.K, s.N)
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
	}
}

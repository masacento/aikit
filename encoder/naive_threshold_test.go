package encoder

import (
	"math"
	"math/rand"
	"testing"
)

// The 4-MFLOP naive/blocked threshold in matmulBT/matmulBTInto (items 27 + 41).
// linalg deleted its own copy of this threshold because the naive span used a
// different reduction order (breaking M-invariance) AND was measured slower than
// the blocked kernel at small-M shapes. This benchmark re-measures that claim on
// amd64 before the encoder's copy is removed, per measuring-performance.md §1.11:
// per-kernel tables do not transfer between machines.
//
// Shapes are the ones the threshold actually diverts: per-head attention QKᵀ
// (L, headDim, L) and scores·V (L, L, headDim) below L≈250, plus the small
// projections.
func BenchmarkNaiveVsBlocked(b *testing.B) {
	rng := rand.New(rand.NewSource(41))
	for _, sh := range []struct {
		name    string
		M, K, N int
	}{
		{"qk_L64", 64, 64, 64},
		{"qk_L128", 128, 64, 128},
		{"qk_L250", 250, 64, 250},
		{"ctx_L128", 128, 128, 64},
		{"ctx_L250", 250, 250, 64},
		{"proj_L22", 22, 768, 768},
	} {
		a := randF32(rng, sh.M*sh.K)
		w := randF32(rng, sh.N*sh.K)
		dst := make([]float32, sh.M*sh.N)
		flops := int64(sh.M) * int64(sh.K) * int64(sh.N)
		b.Run(sh.name+"/naive", func(b *testing.B) {
			b.ReportMetric(float64(flops)/1e6, "MFLOP")
			for b.Loop() {
				matmulBTNaiveInto(a, w, dst, sh.M, sh.K, sh.N)
			}
		})
		b.Run(sh.name+"/blocked", func(b *testing.B) {
			for b.Loop() {
				matmulBTBlockedInto(a, w, dst, sh.M, sh.K, sh.N)
			}
		})
	}
}

// TestNaiveVsBlocked_reductionOrderDiffers documents WHY the threshold is a
// correctness problem, not only a speed one: the two kernels do not agree, so
// which one runs is observable in the output. Once the threshold is gone this
// difference stops being reachable from the encoder.
func TestNaiveVsBlocked_reductionOrderDiffers(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const M, K, N = 128, 64, 128
	a := randF32(rng, M*K)
	w := randF32(rng, N*K)

	naive := make([]float32, M*N)
	matmulBTNaiveInto(a, w, naive, M, K, N)
	blocked := make([]float32, M*N)
	matmulBTBlockedInto(a, w, blocked, M, K, N)

	var diff int
	var worst float64
	for i := range naive {
		if naive[i] != blocked[i] {
			diff++
			if d := math.Abs(float64(naive[i]) - float64(blocked[i])); d > worst {
				worst = d
			}
		}
	}
	if diff == 0 {
		t.Skip("the two kernels happen to agree bitwise on this input; nothing to document")
	}
	t.Logf("naive vs blocked at M=%d K=%d N=%d: %d/%d elements differ, worst |Δ|=%g "+
		"— this is what the 4-MFLOP threshold made observable", M, K, N, diff, len(naive), worst)
}

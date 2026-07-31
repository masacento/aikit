package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// Item 24: K=4096 (large-encoder fc2, kSpan=1024 power-of-two → padded) must stay
// bit-close to naive. Exercises the padded packed stride across M and the tail.
func TestPackedFillPad_matchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(24))
	for _, s := range []struct{ M, K, N int }{
		{1, 4096, 1024}, {8, 4096, 1024}, {357, 4096, 1024}, // bge-large fc2 shape
		{16, 2048, 640}, {5, 2560, 768}, // other K%768!=0 → kSpan=1024 pow2 + tail
	} {
		a := make([]float32, s.M*s.K)
		b := make([]float32, s.N*s.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range b {
			b[i] = float32(rng.NormFloat64())
		}
		got := make([]float32, s.M*s.N)
		MatmulBT(a, b, got, s.M, s.K, s.N)
		want := naiveMatmulBT(a, b, s.M, s.K, s.N)
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-3+1e-3*math.Abs(float64(want[i])) {
				t.Fatalf("shape %+v out[%d]=%g want~%g Δ%g", s, i, got[i], want[i], d)
			}
		}
	}
}

package ann

import (
	"math"
	"math/rand"
	"testing"
)

// BenchmarkHNSWQueryBatched is the arbiter for item 15. The doc's prototype
// measured 1.40× at ef=64 and 1.36× at ef=200 on n=50k, d=256.
func BenchmarkHNSWQueryBatched(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n, d = 50_000, 256
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, d)
		var norm float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		inv := float32(1 / math.Sqrt(norm))
		for j := range v {
			v[j] *= inv
		}
		vecs[i] = v
	}
	h := BuildHNSW(vecs, Config{})
	q := vecs[0]
	for _, ef := range []int{64, 200} {
		b.Run("ef"+itoaAnn(ef), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkHits = h.QueryEf(q, 10, ef)
			}
		})
	}
}

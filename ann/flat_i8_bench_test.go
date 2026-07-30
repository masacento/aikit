package ann

import (
	"math"
	"math/rand"
	"testing"
)

// BenchmarkFlatI8Query is the arbiter for item 4 — the per-query score buffer
// and kernel Workspace. The estimate was "10–25% now, large at N≥1M"; allocation
// per query is the quantity it is really about.
func BenchmarkFlatI8Query(b *testing.B) {
	for _, n := range []int{10_000, 100_000} {
		rng := rand.New(rand.NewSource(int64(n)))
		const d = 384
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
		f := NewFlatI8(vecs)
		q := vecs[0]
		b.Run(shapeN(n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkHits = f.Query(q, 10)
			}
		})
	}
}

func shapeN(n int) string {
	if n >= 1000 {
		return "N" + itoaAnn(n/1000) + "k"
	}
	return "N" + itoaAnn(n)
}

func itoaAnn(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

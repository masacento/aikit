//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// Per-DOT-PRODUCT cost, so the shapes are comparable: each arm computes a different
// number of dot products, and ReportMetric normalises by that.
func BenchmarkKernelShapes(b *testing.B) {
	const K = 768
	rng := rand.New(rand.NewPCG(1, 2))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	a := [3][]float32{rv(K), rv(K), rv(K)}
	bb := [8][]float32{rv(K), rv(K), rv(K), rv(K), rv(K), rv(K), rv(K), rv(K)}
	p := func(i int) *float32 { return &bb[i][0] }
	gmac := func(b *testing.B, dots int) {
		b.ReportMetric(float64(dots)*float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	}

	b.Run("1x8_dotFMA8", func(b *testing.B) {
		var o [8]float32
		for b.Loop() {
			dotFMA8(&a[0][0], p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), K, &o)
		}
		gmac(b, 8)
	})
	b.Run("3x4_dotFMA3x4", func(b *testing.B) {
		var o [12]float32
		for b.Loop() {
			dotFMA3x4(&a[0][0], &a[1][0], &a[2][0], p(0), p(1), p(2), p(3), K, &o)
		}
		gmac(b, 12)
	})
}

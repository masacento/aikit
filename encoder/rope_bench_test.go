package encoder

import (
	"math/rand"
	"testing"
)

// BenchmarkRotateHalfInto reports ns/elem at half=32 (headDim=64, a common
// text-encoder shape). Reflects whichever build (scalar or
// GOEXPERIMENT=simd) it's run under, since rotateHalfInto is the same call
// either way.
func BenchmarkRotateHalfInto(b *testing.B) {
	const half = 32
	rng := rand.New(rand.NewSource(1))
	x1 := make([]float32, half)
	x2 := make([]float32, half)
	c := make([]float32, half)
	s := make([]float32, half)
	for i := range x1 {
		x1[i] = float32(rng.NormFloat64())
		x2[i] = float32(rng.NormFloat64())
		c[i] = float32(rng.Float64())
		s[i] = float32(rng.Float64())
	}
	for b.Loop() {
		rotateHalfInto(x1, x2, c, s)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*half*2), "ns/elem")
}

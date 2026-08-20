//go:build goexperiment.simd

package encoder

import (
	"math"
	"math/rand"
	"testing"
)

// rotateHalfIntoScalar is rope_scalar.go's body, reproduced here since that
// file is excluded from this build — the reference this test checks
// rotateHalfInto (the SIMD build) against.
func rotateHalfIntoScalar(x1, x2, c, s []float32) {
	for d := range x1 {
		a, b := x1[d], x2[d]
		cd, sd := c[d], s[d]
		x1[d] = a*cd - b*sd
		x2[d] = b*cd + a*sd
	}
}

// TestRotateHalfInto_matchesScalar checks the SIMD rotation against the
// scalar reference at every half-width that plausibly occurs (small ones to
// exercise every tail remainder, plus real headDim/2 values). Pure
// Mul/Sub/Add carries no MulAdd/FMA contraction, so this measures the actual
// gap rather than assuming bit-identity (see vision/rope_simd_test.go, the
// same shape).
func TestRotateHalfInto_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	var maxDiff float32
	for _, half := range []int{1, 2, 3, 4, 7, 8, 16, 32, 64} {
		for trial := range 20 {
			x1a := make([]float32, half)
			x2a := make([]float32, half)
			c := make([]float32, half)
			s := make([]float32, half)
			for i := range x1a {
				x1a[i] = float32(rng.NormFloat64())
				x2a[i] = float32(rng.NormFloat64())
				theta := rng.Float64() * math.Pi
				c[i] = float32(math.Cos(theta))
				s[i] = float32(math.Sin(theta))
			}
			x1b := append([]float32(nil), x1a...)
			x2b := append([]float32(nil), x2a...)

			rotateHalfIntoScalar(x1a, x2a, c, s)
			rotateHalfInto(x1b, x2b, c, s)

			for i := range x1a {
				for _, d := range []float32{
					float32(math.Abs(float64(x1a[i] - x1b[i]))),
					float32(math.Abs(float64(x2a[i] - x2b[i]))),
				} {
					if d > maxDiff {
						maxDiff = d
					}
					if d > 1e-5 {
						t.Fatalf("half=%d trial=%d i=%d: scalar=(%v,%v) simd=(%v,%v)",
							half, trial, i, x1a[i], x2a[i], x1b[i], x2b[i])
					}
				}
			}
		}
	}
	t.Logf("max abs diff vs scalar: %v", maxDiff)
}

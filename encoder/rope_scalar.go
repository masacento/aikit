//go:build !goexperiment.simd

package encoder

// rotateHalfInto (default build) — byte-for-byte the loop applyRows (rope.go)
// had before rope_simd.go existed.
func rotateHalfInto(x1, x2, c, s []float32) {
	for d := range x1 {
		a, b := x1[d], x2[d]
		cd, sd := c[d], s[d]
		x1[d] = a*cd - b*sd
		x2[d] = b*cd + a*sd
	}
}

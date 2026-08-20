//go:build goexperiment.simd

package encoder

import "simd"

// rotateHalfInto (SIMD build) — see rope.go for the algorithm and
// rope_scalar.go for the default-build body. x1/x2 read fully into vector
// registers before either is overwritten, matching the scalar body's own
// ordering guarantee. Pure Mul/Sub/Add (no MulAdd) — see
// vision/rope_simd.go's identical shape and its test for why this is
// measured, not assumed, against the scalar reference.
func rotateHalfInto(x1, x2, c, s []float32) {
	n := len(x1)
	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= n; i += L {
		av := simd.LoadFloat32s(x1[i : i+L])
		bv := simd.LoadFloat32s(x2[i : i+L])
		cv := simd.LoadFloat32s(c[i : i+L])
		sv := simd.LoadFloat32s(s[i : i+L])
		newA := av.Mul(cv).Sub(bv.Mul(sv))
		newB := bv.Mul(cv).Add(av.Mul(sv))
		newA.Store(x1[i : i+L])
		newB.Store(x2[i : i+L])
	}
	for ; i < n; i++ { // tail
		a, b := x1[i], x2[i]
		cd, sd := c[i], s[i]
		x1[i] = a*cd - b*sd
		x2[i] = b*cd + a*sd
	}
}

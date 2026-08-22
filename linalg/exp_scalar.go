//go:build !goexperiment.simd

package linalg

// softmaxRowIntoRaw is SoftmaxRowInto (exp.go) with the length/empty checks
// already done. This is the default build — byte-for-byte the body
// SoftmaxRowInto had before exp_simd.go existed; see that file for the
// GOEXPERIMENT=simd alternative.
func softmaxRowIntoRaw(dst, src []float32) {
	maxV := src[0]
	for _, v := range src[1:] {
		if v > maxV {
			maxV = v
		}
	}
	var sum float64
	for i, v := range src {
		// v-maxV <= 0 by construction, so only the underflow end is reachable
		// and expF32Core's guards suffice.
		e := expF32Core(v - maxV)
		dst[i] = e
		sum += float64(e)
	}
	if sum == 0 {
		u := 1.0 / float32(len(dst))
		for i := range dst {
			dst[i] = u
		}
		return
	}
	inv := float32(1.0 / sum)
	for i := range dst {
		dst[i] *= inv
	}
}

// siluIntoRaw is SiLUInto (exp.go) with the length/empty checks already done.
// This is the default build — byte-for-byte the loop SiLUInto had before
// exp_simd.go's siluIntoRaw existed; see that file for the GOEXPERIMENT=simd
// alternative.
func siluIntoRaw(dst, src []float32) {
	for i, v := range src {
		dst[i] = SiLUF32(v)
	}
}

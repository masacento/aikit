//go:build !amd64 && !arm64

package linalg

// dequantRowInt8 on other arches is the portable scalar widen. amd64 (AVX2) and
// arm64 (NEON) have vectorized kernels in their own files.
func dequantRowInt8(dst []float32, q []int8, scale float32) {
	dequantRowInt8Scalar(dst, q, scale)
}

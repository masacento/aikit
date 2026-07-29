//go:build !amd64

package linalg

// dequantRowInt8 on non-amd64 is the portable scalar widen. An arm64 NEON
// version (SXTL + SCVTF + FMUL) is the obvious follow-up; it is not written yet
// because the measurement that justified this kernel was taken on amd64.
func dequantRowInt8(dst []float32, q []int8, scale float32) {
	dequantRowInt8Scalar(dst, q, scale)
}

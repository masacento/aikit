//go:build arm64

package linalg

// dequantI8NEON widens n int8 values (n a multiple of 16) into dst, scaling each
// by scale: dst[i] = float32(q[i]) * scale. Signed widen (SXTL/SXTL2) → SCVTF →
// FMUL, 16 elements per iteration. Implemented in dequant_i8_arm64.s. All base
// ARMv8-A NEON, so no feature check.
//
//go:noescape
func dequantI8NEON(q *int8, dst *float32, n int, scale float32)

// dequantRowInt8 widens one row. The NEON kernel handles the 16-aligned prefix
// and the Go tail finishes the remainder (every real weight row here — K is
// 768/2048/3072 — is a multiple of 16, so the tail is only for odd test shapes).
func dequantRowInt8(dst []float32, q []int8, scale float32) {
	if len(q) < 16 {
		dequantRowInt8Scalar(dst, q, scale)
		return
	}
	n := len(q) &^ 15
	dequantI8NEON(&q[0], &dst[0], n, scale)
	if n < len(q) {
		dequantRowInt8Scalar(dst[n:], q[n:], scale)
	}
}

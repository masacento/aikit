//go:build !arm64

package linalg

// dotPartials16 on non-arm64: the scalar reference (correct, not fast — §5.5). The SDOT path lives
// in kquant_dp_arm64.go.
func dotPartials16(codes, qs []int8, out []int32) { dotPartials16Scalar(codes, qs, out) }

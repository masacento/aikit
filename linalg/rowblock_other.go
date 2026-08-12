//go:build !amd64

package linalg

// blockRows3x4 has no 12-accumulator kernel on this architecture and consumes no
// rows. arm64 takes its own 2×8 dual-row path in the caller; everything else runs
// single-row.
func blockRows3x4(a, b, dst []float32, i, iEnd, K, N, k0, k4, kSpan, nStart, nEnd int) int {
	return i
}

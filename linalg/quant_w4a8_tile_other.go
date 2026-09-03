//go:build !amd64

package linalg

// w4a8TileRows claims no activation rows off amd64. The W4A8 activation-blocked
// tile (docs/task-simd-audit.md S-01, amd64 half) is an AVX2 kernel; arm64's
// answer to the same finding is the 4x4 row4 tile reached through
// WeightMat.MatmulBTW4A8Into, which needs the repacked layout and so lives on the
// WeightMat path rather than inside this canonical span.
//
// Returning 0 makes w4a8Span run precisely the loop it ran before any tile
// existed.
func w4a8TileRows(aq []int8, aScales []float32, w4 []byte, wScales, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) int {
	return 0
}

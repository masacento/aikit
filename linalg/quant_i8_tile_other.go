//go:build !arm64

package linalg

// w8a8TileRect claims nothing off arm64: the register-blocked W8A8 tile
// (docs/task-simd-audit.md S-01b) is an SDOT kernel. Returning (0, j0) means "I
// took no rows and no columns", so w8a8Span's first fallback call covers the
// whole span and the second is empty — the pre-tile code path exactly.
//
// amd64's equivalent would be an M-tile over dotI8AVX2's four accumulators; it is
// listed as open in S-01b and needs the other box to measure, so it is not
// guessed at here.
func w8a8TileRect(aq []int8, aScales []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1 int) (mTiled, jTiled int) {
	return 0, j0
}

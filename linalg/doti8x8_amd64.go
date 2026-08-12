//go:build amd64

package linalg

// dotI8x8AVX2 computes eight int8 dot products of one a-row against eight b-rows,
// sharing the a-side widening across all eight. n must be a multiple of 16.
//
//go:noescape
func dotI8x8AVX2(a, b0, b1, b2, b3, b4, b5, b6, b7 *int8, n int, out *[8]int32)

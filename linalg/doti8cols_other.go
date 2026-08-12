//go:build !amd64

package linalg

// dotI8Cols8 has no multi-column int8 kernel on this architecture.
func dotI8Cols8(a []int8, bQ []int8, K, j int, out *[8]int32) {
	dotI8Cols8Generic(a, bQ, K, j, out)
}

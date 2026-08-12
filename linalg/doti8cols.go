package linalg

// dotI8Cols8Generic scores one a-row against eight consecutive b-rows the plain way,
// one column at a time. It is the reference the blocked path must equal, and the
// implementation every architecture without a multi-column int8 kernel uses.
func dotI8Cols8Generic(a []int8, bQ []int8, K, j int, out *[8]int32) {
	for c := range 8 {
		out[c] = dotI8(a, bQ[(j+c)*K:(j+c)*K+K])
	}
}

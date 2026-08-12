//go:build amd64

package linalg

// dotI8Cols8 scores one a-row against eight consecutive b-rows, sharing the a-side
// widening across all eight (see doti8x8_amd64.s). Falls back to the per-column path
// without AVX2, or when K is too short for the 16-element kernel to be worth entering.
//
// The kernel consumes K&^15 elements; the K%16 remainder is finished per column in
// Go. Integer arithmetic is associative, so splitting the sum this way is exactly
// equal to the unsplit one — no reassociation concern of the kind the f32 kernels
// carry.
func dotI8Cols8(a []int8, bQ []int8, K, j int, out *[8]int32) {
	if !hasAVX2 || K < 16 {
		dotI8Cols8Generic(a, bQ, K, j, out)
		return
	}
	n16 := K &^ 15
	dotI8x8AVX2(&a[0],
		&bQ[(j+0)*K], &bQ[(j+1)*K], &bQ[(j+2)*K], &bQ[(j+3)*K],
		&bQ[(j+4)*K], &bQ[(j+5)*K], &bQ[(j+6)*K], &bQ[(j+7)*K],
		n16, out)
	if n16 == K {
		return
	}
	for c := range 8 {
		bj := bQ[(j+c)*K : (j+c)*K+K]
		s := out[c]
		for k := n16; k < K; k++ {
			s += int32(a[k]) * int32(bj[k])
		}
		out[c] = s
	}
}

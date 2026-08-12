package linalg

// dot8ColsGeneric folds Dot8x4's 4-lane partial sums into eight totals, in the
// left-to-right order the blocked GEMM has always used. It is the reduction every
// architecture without a fully-reducing kernel takes, and the reference the amd64
// fast path must agree with.
func dot8ColsGeneric(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, out *[8]float32) {
	var sums [32]float32
	Dot8x4(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, &sums)
	for r := range 8 {
		out[r] = sums[r*4] + sums[r*4+1] + sums[r*4+2] + sums[r*4+3]
	}
}

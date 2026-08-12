//go:build !amd64

package linalg

// dot8ColsInto has no fully-reducing kernel on this architecture, so it takes the
// portable fold. On arm64 that is the intended path: dotNEON8x4's four live 4-lane
// accumulators per column are genuine partial sums, not padding.
func dot8ColsInto(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, out *[8]float32) {
	dot8ColsGeneric(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, out)
}

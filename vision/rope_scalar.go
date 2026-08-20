//go:build !goexperiment.simd

package vision

// applyRotaryVision (default build) — byte-for-byte the body this function had
// before rope_simd.go existed. In-place pairwise: each (vec[d], vec[d+half])
// rotates into itself, so no per-call scratch is needed — this was ~8M tiny
// allocs on a realistic image (~8k patches x 16 heads x 32 blocks). Reads x,y
// before overwriting either, and is bit-identical to the tmp version
// (a+(-b)*s == a-b*s in IEEE).
func applyRotaryVision(vec, cos, sin []float32) {
	half := len(vec) / 2
	for d := range half {
		x, y := vec[d], vec[d+half]
		vec[d] = x*cos[d] - y*sin[d]
		vec[d+half] = y*cos[d+half] + x*sin[d+half]
	}
}

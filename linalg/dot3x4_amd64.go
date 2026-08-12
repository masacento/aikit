//go:build amd64

package linalg

// dotFMA3x4 computes twelve dot products — three a-rows against four b-rows — with
// twelve live accumulator chains, the first shape that clears Zen 2's 10-chain
// latency threshold. See dot3x4_amd64.s for the measured basis.
//
// n must be a multiple of 4. out is [a0·b0..a0·b3, a1·b0..a1·b3, a2·b0..a2·b3].
//
//go:noescape
func dotFMA3x4(a0, a1, a2, b0, b1, b2, b3 *float32, n int, out *[12]float32)

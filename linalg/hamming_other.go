//go:build !amd64

package linalg

// hammingRows on every non-amd64 arch is the portable kernel.
//
// arm64 needs no assembly here: math/bits.OnesCount64 IS intrinsified on arm64
// (VCNT + VADDV), unlike amd64 where it is gated behind GOAMD64 >= v2 and
// compiles to the SWAR fallback by default. That asymmetry is the whole reason
// hamming_amd64.s exists.
func hammingRows(q, codes []uint64, words, n int, dst []uint16) {
	hammingRowsGeneric(q, codes, words, n, dst)
}

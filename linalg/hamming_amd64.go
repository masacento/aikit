//go:build amd64

package linalg

// hammingPOPCNT is the POPCNTQ kernel in hamming_amd64.s.
//
//go:noescape
func hammingPOPCNT(q *uint64, codes *uint64, words int, n int, dst *uint16)

// hasPOPCNT reports whether the CPU implements POPCNT (SSE4.2 era, 2008 and
// later). Detected at init for the same reason hasAVX2 is: so one binary runs
// the fast kernel where it can and the portable one where it cannot, instead of
// the build's GOAMD64 level deciding for every user.
//
// No OS-support check is needed, unlike AVX2 — POPCNT adds no architectural
// state, so there is nothing for the kernel to save or restore.
var hasPOPCNT = detectPOPCNT()

func detectPOPCNT() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 1 {
		return false
	}
	// Leaf 1: ECX bit 23 = POPCNT.
	_, _, ecx1, _ := cpuid(1, 0)
	const bitPOPCNT = 1 << 23
	return ecx1&bitPOPCNT != 0
}

func hammingRows(q, codes []uint64, words, n int, dst []uint16) {
	if !hasPOPCNT || n == 0 || words == 0 {
		hammingRowsGeneric(q, codes, words, n, dst)
		return
	}
	hammingPOPCNT(&q[0], &codes[0], words, n, &dst[0])
}

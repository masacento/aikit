package linalg

import "math/bits"

// Binary (1-bit) quantization: each float32 component becomes one sign bit, and
// the similarity proxy is the Hamming distance between the packed codes.
//
// The geometry it rests on: for unit vectors the expected Hamming distance
// between sign codes is proportional to the ANGLE between them
// (Goemans–Williamson / SimHash, E[H]/dim = θ/π), so ordering by ascending
// Hamming distance approximates ordering by descending cosine. It is a
// monotone-in-expectation proxy, not an exact one — which is why the only
// intended use is a PREFILTER whose survivors get rescored exactly
// (ann.FlatBinary).
//
// The point is the memory traffic. A float32 dot at dim 768 reads 3072 B per
// candidate; the packed code reads 96 B — 32× less — and the whole corpus'
// codes fit where 1/32nd of it used to.

// PackSignBitsRow packs the sign bits of v into dst, one bit per component,
// LSB-first within each word: bit (i%64) of dst[i/64] is 1 iff v[i] >= 0.
// dst must hold at least PackedWords(len(v)) words, and is fully written
// (trailing bits of the last word are zeroed), so it may be reused.
//
// Zero maps to 1, matching math.Signbit's "is it negative" reading. Which side
// zero falls on does not matter for retrieval — it must only be CONSISTENT
// between the query and the corpus, and both go through this function.
func PackSignBitsRow(dst []uint64, v []float32) {
	w := PackedWords(len(v))
	clear(dst[:w])
	for i, x := range v {
		if x >= 0 {
			dst[i>>6] |= 1 << uint(i&63)
		}
	}
}

// PackedWords is the number of uint64 words one dim-component code occupies.
func PackedWords(dim int) int { return (dim + 63) / 64 }

// PackSignBits packs a row-major [n, dim] float32 block into [n, words] codes.
func PackSignBits(dst []uint64, src []float32, n, dim int) {
	w := PackedWords(dim)
	for i := range n {
		PackSignBitsRow(dst[i*w:(i+1)*w], src[i*dim:(i+1)*dim])
	}
}

// HammingRows computes dst[i] = popcount(q ^ codes[i·words:]) for i in [0, n).
//
// dst is uint16 because the distance is bounded by dim, and a narrow
// destination is the point: the caller scans it right after this returns, and
// at n = 10⁶ the difference is a 2 MB pass instead of 8 MB.
func HammingRows(q, codes []uint64, words, n int, dst []uint16) {
	hammingRows(q, codes, words, n, dst)
}

// hammingRowsGeneric is the portable implementation, and the reference the
// arch-specific kernels are tested against.
//
// The word count is a compile-time-unknown but tiny loop bound (12 at dim 768),
// so the inner loop is where all the time goes: two loads, an XOR and a
// popcount per word.
func hammingRowsGeneric(q, codes []uint64, words, n int, dst []uint16) {
	for i := range n {
		row := codes[i*words : (i+1)*words : (i+1)*words]
		s := 0
		for j, c := range row {
			s += bits.OnesCount64(c ^ q[j])
		}
		dst[i] = uint16(s)
	}
}

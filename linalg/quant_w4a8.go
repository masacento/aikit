package linalg

// dotW4A8Scalar is the portable reference for one W4A8 output: the int8
// activation row dotted against a group-int4 weight row, each group's integer
// dot scaled by its f32 weight scale. The asm kernels (dotW4A8SDOT) must match
// this bit-for-bit on the integer accumulation; the per-group f32 fold may
// differ only in float rounding. group need not divide K (ragged final group).
func dotW4A8Scalar(act []int8, packed []byte, scales []float32, group, K int) float32 {
	var total float32
	nGroups := (K + group - 1) / group
	for g := range nGroups {
		ks := g * group
		ke := min(ks+group, K)
		var acc int32
		for k := ks; k < ke; k++ {
			b := packed[k>>1]
			nib := b & 0x0F
			if k&1 == 1 {
				nib = b >> 4
			}
			acc += int32(act[k]) * int32(int(nib)-8)
		}
		total += float32(acc) * scales[g]
	}
	return total
}

// dotW4A8UncenteredScalar is the reference for the SDOTv2 kernel's arithmetic:
// the algebraic rearrangement Σ(nib_k-8)·act_k = Σnib_k·act_k - 8·Σact_k,
// applied per group, with the correction term precomputed in sumAct
// (SumActGroupsInto). Exact — every step here is integer arithmetic until the
// single per-group float scale multiply, the same point dotW4A8Scalar first
// touches a float, so this must be BIT-IDENTICAL to it (not just close): both
// reduce to "float32(acc)*scales[g]" for the identical int32 acc, summed in
// the identical group order. The test asserting that equality is the real
// correctness gate for this rearrangement — this function's own body proves
// nothing about the SDOTv2 asm kernel, only about the algebra.
func dotW4A8UncenteredScalar(act []int8, packed []byte, scales []float32, sumAct []int32, group, K int) float32 {
	var total float32
	nGroups := (K + group - 1) / group
	for g := range nGroups {
		ks := g * group
		ke := min(ks+group, K)
		var raw int32
		for k := ks; k < ke; k++ {
			b := packed[k>>1]
			nib := b & 0x0F
			if k&1 == 1 {
				nib = b >> 4
			}
			raw += int32(act[k]) * int32(nib)
		}
		acc := raw - 8*sumAct[g]
		total += float32(acc) * scales[g]
	}
	return total
}

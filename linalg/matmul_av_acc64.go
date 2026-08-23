package linalg

// MatmulAVAcc64 computes dst[M,hd] = scores[M,nKeys] · V_head[nKeys,hd], where
// V_head is one KV head's slice of a row-major [nKeys, rowStride] buffer
// (rowStride is the full kvDim; headOff selects this head's hd-wide column
// range within each row) — the attention scores·V step, f64-accumulated.
//
// WHY THIS EXISTS, over MatmulBTAcc64Strided (attention's other current option).
// That kernel's strided accessor reads b[j][k] = vals[headOff + j + k*rowStride]
// — for a FIXED output dim j, it walks k=0..nKeys-1 at stride rowStride (one
// full V row apart, ~1KB at typical hd), i.e. one cache line touched per f64
// MAC, repeated for every one of the hd output dims. This kernel instead reads
// each key's V row ONCE, contiguously (rowStride-1 wasted floats aside, the
// hd-wide head slice itself is contiguous), and folds it into hd INDEPENDENT
// f64 accumulators — one per output dim — in one pass over the keys.
//
// BIT-IDENTICAL BY CONSTRUCTION, not by parity test. Each output dim's
// accumulator receives EXACTLY the same sequence of adds, in the exact same
// key-ascending order, as MatmulBTAcc64Strided's per-dim reduction: only the
// loop NESTING changes (keys-outer/dims-inner instead of dims-outer/keys-
// inner), and floating-point addition depends on the order of operations
// applied to ONE accumulator, not on what unrelated accumulators do between
// those operations. This is the "split the independent axis, never the
// reduction" principle (docs/task-decode-splitkv-attention.md in goinfer),
// applied by loop-nest interchange rather than parallelism — but it composes
// with parallelism too: MatmulAVAcc64PerQuery below runs each query row's own
// hd-accumulator sweep independently, so distributing queries (or, in the
// decode caller, heads) across goroutines touches none of this ordering.
//
// acc is caller-provided [hd]-float64 scratch (steady-state decode calls this
// hundreds of times per token; a fresh make([]float64, hd) per call would be
// real allocation pressure). Zeroed on entry, not preserved on exit.
func MatmulAVAcc64(scores, vals, dst []float32, acc []float64, M, nKeys, hd, headOff, rowStride int) {
	checkMatmulAVAcc64(len(scores), len(vals), len(dst), len(acc), M, nKeys, hd, headOff, rowStride)
	for i := range M {
		srow := scores[i*nKeys : i*nKeys+nKeys]
		for d := range acc[:hd] {
			acc[d] = 0
		}
		for s := range nKeys {
			w := float64(srow[s])
			vrow := vals[headOff+s*rowStride : headOff+s*rowStride+hd]
			for d := range hd {
				acc[d] += w * float64(vrow[d])
			}
		}
		drow := dst[i*hd : i*hd+hd]
		for d := range hd {
			drow[d] = float32(acc[d])
		}
	}
}

func checkMatmulAVAcc64(scoresLen, valsLen, dstLen, accLen, M, nKeys, hd, headOff, rowStride int) {
	if M < 0 || nKeys < 0 || hd < 0 || headOff < 0 || rowStride < hd {
		panic("linalg: MatmulAVAcc64 invalid shape")
	}
	requireExactLen("MatmulAVAcc64", "scores", scoresLen, mul(M, nKeys))
	requireExactLen("MatmulAVAcc64", "dst", dstLen, mul(M, hd))
	if accLen < hd {
		panic("linalg: MatmulAVAcc64 acc scratch shorter than hd")
	}
	need := headOff + max(0, nKeys-1)*rowStride + hd
	if valsLen < need {
		panic("linalg: MatmulAVAcc64 vals too short for the given shape/strides")
	}
}

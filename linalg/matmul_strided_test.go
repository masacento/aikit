package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestMatmulBTAcc64Strided_bitIdenticalToPacked asserts (P1 step 3) that reading the second
// operand with a stride is byte-for-byte identical to MatmulBTAcc64 run on a packed/transposed
// copy of the same logical data — the confirmation the substitution needs, since the reduction
// order is unchanged and only b's addressing differs. It exercises the two attention shapes the
// call sites use, out of a realistic interleaved KV cache [nKeys, kvDim=nKV·hd]:
//   - K re-copy   (QKᵀ):      strided rows, contiguous elements
//   - V re-transpose (scores·V): contiguous rows, strided elements
//
// at every KV head, and across M=1 (decode) and M>1 (prefill/verify).
func TestMatmulBTAcc64Strided_bitIdenticalToPacked(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	shapes := []struct {
		Mq, nKeys, nKV, hd int
		note               string
	}{
		{1, 512, 2, 64, "decode M=1, GQA nKV=2"},
		{1, 4096, 4, 128, "decode M=1, long ctx, nKV=4 hd=128"},
		{8, 300, 8, 64, "prefill M=8, MHA-ish nKV=8"},
		{5, 40, 3, 16, "tiny odd shape"},
	}
	for _, s := range shapes {
		Mq, nKeys, nKV, hd := s.Mq, s.nKeys, s.nKV, s.hd
		kvDim := nKV * hd

		// interleaved KV cache [nKeys, kvDim] and the query [Mq, nH*hd] (nH≥nKV; use nKV here).
		cache := make([]float32, nKeys*kvDim)
		for i := range cache {
			cache[i] = float32(rng.NormFloat64())
		}
		qh := make([]float32, Mq*hd)
		for i := range qh {
			qh[i] = float32(rng.NormFloat64())
		}
		scores := make([]float32, Mq*nKeys)
		for i := range scores {
			scores[i] = float32(rng.NormFloat64())
		}

		for kvh := 0; kvh < nKV; kvh++ {
			off := kvh * hd

			// --- K re-copy: QKᵀ, dst[Mq,nKeys] = qh[Mq,hd]·K_head[nKeys,hd]ᵀ ---
			// reference: pack K_head then MatmulBTAcc64.
			kPacked := make([]float32, nKeys*hd)
			for j := 0; j < nKeys; j++ {
				copy(kPacked[j*hd:j*hd+hd], cache[j*kvDim+off:j*kvDim+off+hd])
			}
			refK := make([]float32, Mq*nKeys)
			MatmulBTAcc64(qh, kPacked, refK, Mq, hd, nKeys)
			gotK := make([]float32, Mq*nKeys)
			// strided: b[j][k] = cache[off + j*kvDim + k*1]
			MatmulBTAcc64Strided(qh, cache, gotK, Mq, hd, nKeys, off, kvDim, 1)
			assertBitIdentical(t, "K/"+s.note, kvh, gotK, refK)

			// --- V re-transpose: scores·V, dst[Mq,hd] = scores[Mq,nKeys]·V_head[nKeys,hd] ---
			// reference: transpose V_head to vt[hd,nKeys] then MatmulBTAcc64.
			vt := make([]float32, hd*nKeys)
			for j := 0; j < nKeys; j++ {
				for d := 0; d < hd; d++ {
					vt[d*nKeys+j] = cache[j*kvDim+off+d]
				}
			}
			refV := make([]float32, Mq*hd)
			MatmulBTAcc64(scores, vt, refV, Mq, nKeys, hd)
			gotV := make([]float32, Mq*hd)
			// strided: b[j][k] = cache[off + j*1 + k*kvDim]
			MatmulBTAcc64Strided(scores, cache, gotV, Mq, nKeys, hd, off, 1, kvDim)
			assertBitIdentical(t, "V/"+s.note, kvh, gotV, refV)
		}
	}
}

// TestMatmulBTAcc64Strided_packedEqualsMatmulBTAcc64 asserts the packed strides (bOff=0,
// bRowStride=K, bElemStride=1) reduce exactly to MatmulBTAcc64 — the base case the two
// attention addressings both specialise.
func TestMatmulBTAcc64Strided_packedEqualsMatmulBTAcc64(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	M, K, N := 3, 128, 200
	a := make([]float32, M*K)
	b := make([]float32, N*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range b {
		b[i] = float32(rng.NormFloat64())
	}
	ref := make([]float32, M*N)
	MatmulBTAcc64(a, b, ref, M, K, N)
	got := make([]float32, M*N)
	MatmulBTAcc64Strided(a, b, got, M, K, N, 0, K, 1)
	assertBitIdentical(t, "packed==MatmulBTAcc64", 0, got, ref)
}

func assertBitIdentical(t *testing.T, tag string, kvh int, got, ref []float32) {
	t.Helper()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
			t.Fatalf("%s kvh=%d idx=%d: strided %08x != packed %08x",
				tag, kvh, i, math.Float32bits(got[i]), math.Float32bits(ref[i]))
		}
	}
}

package linalg

import (
	"math/rand"
	"testing"
)

// BenchmarkAttnStridedVsPacked answers the P1 "does the strided read earn its place" question
// at the attention shapes, separately for K and V, without a full decoder. Baseline is what
// goinfer does today — materialise a packed (K) or transposed (V) copy of a head-block out of
// the interleaved [nKeys, nKV·hd] KV cache, then MatmulBTAcc64. Strided is MatmulBTAcc64Strided
// reading the cache in place. Same result to the bit (matmul_strided_test.go); this times them.
//
//	K: gather+matmul   vs  strided     — expected clean win (contiguous inner dot either way,
//	                                      strided just drops the copy pass)
//	V: transpose+matmul vs strided     — the real question: the strided read is strided by kvDim,
//	                                      but the transpose it removes was also a strided write
func BenchmarkAttnStridedVsPacked(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	for _, s := range []struct {
		nKeys, nKV, hd int
		name           string
	}{
		{4096, 4, 128, "nKeys4096_nKV4_hd128"}, // ~1.5B/7B decode at 4k
		{2048, 2, 64, "nKeys2048_nKV2_hd64"},   // ~0.5B decode at 2k
	} {
		nKeys, nKV, hd := s.nKeys, s.nKV, s.hd
		kvDim := nKV * hd
		cache := make([]float32, nKeys*kvDim)
		for i := range cache {
			cache[i] = float32(rng.NormFloat64())
		}
		qh := make([]float32, hd) // M=1 decode query head
		for i := range qh {
			qh[i] = float32(rng.NormFloat64())
		}
		scores := make([]float32, nKeys) // M=1 scores row
		for i := range scores {
			scores[i] = float32(rng.NormFloat64())
		}
		kh := make([]float32, nKeys*hd)
		vt := make([]float32, hd*nKeys)
		outK := make([]float32, nKeys)
		outV := make([]float32, hd)
		off := (nKV / 2) * hd // some interior head

		b.Run(s.name+"/K_gather+matmul", func(b *testing.B) {
			for b.Loop() {
				for j := 0; j < nKeys; j++ {
					copy(kh[j*hd:j*hd+hd], cache[j*kvDim+off:j*kvDim+off+hd])
				}
				MatmulBTAcc64(qh, kh, outK, 1, hd, nKeys)
			}
		})
		b.Run(s.name+"/K_strided", func(b *testing.B) {
			for b.Loop() {
				MatmulBTAcc64Strided(qh, cache, outK, 1, hd, nKeys, off, kvDim, 1)
			}
		})
		b.Run(s.name+"/V_transpose+matmul", func(b *testing.B) {
			for b.Loop() {
				for j := 0; j < nKeys; j++ {
					for d := 0; d < hd; d++ {
						vt[d*nKeys+j] = cache[j*kvDim+off+d]
					}
				}
				MatmulBTAcc64(scores, vt, outV, 1, nKeys, hd)
			}
		})
		b.Run(s.name+"/V_strided", func(b *testing.B) {
			for b.Loop() {
				MatmulBTAcc64Strided(scores, cache, outV, 1, nKeys, hd, off, 1, kvDim)
			}
		})
	}
}

package linalg

import (
	"math/rand"
	"testing"
)

// BenchmarkAttnStridedVsPacked times the strided read vs a packed (K) / transposed (V) copy at
// the attention shapes, without a full decoder. Baseline materialises a head-block out of the
// interleaved [nKeys, nKV·hd] KV cache, then MatmulBTAcc64; strided is MatmulBTAcc64Strided
// reading the cache in place. Same result to the bit (matmul_strided_test.go); this times them.
//
// DO NOT DECIDE ADOPTION FROM THIS BENCHMARK — it was MISLEADING IN BOTH DIRECTIONS, and only the
// on-target end-to-end decode A/B settles the question (see the P1 measurement doc):
//
//  1. It does not model the GQA group. A real head-block gather is done ONCE and reused across the
//     `group` query heads; this times one gather : one matmul. So it overstates the V win (the
//     transpose it "removes" is amortised across the group in the real forward) and understates the
//     K loss.
//  2. It cannot see cache-line geometry, and it runs on whatever box invokes it. The V strided read
//     has bElemStride = kvDim, so every k lands on its own cache line (~13.7× the line-bytes of the
//     transpose). Whether that is survivable is an ISA property: on arm64/M1 (128 B lines, ~200 GB/s,
//     strong prefetch) strided V wins; on x86-64 (64 B lines) it is a ~40% decode regression at 4k.
//     A benchmark run on one box is not evidence for a portable decision.
//
// It is kept only as a shape/allocation smoke; the perf conclusion belongs to the per-ISA A/B.
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
				for j := range nKeys {
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
				for j := range nKeys {
					for d := range hd {
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

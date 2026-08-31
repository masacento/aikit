//go:build amd64

package linalg

import (
	"math/rand"
	"testing"
)

// BenchmarkP14_W4A8OpsPerByte answers the next step goinfer's queue-performance.md P14 names:
// micro-benchmark dotW4A8FoldAVX2 against its OWN theoretical ops/byte, before anyone redesigns a
// quant format.
//
// THE QUESTION. P14 measured goinfer at 11.7 GB/s of weights against ollama's 27.6 GB/s on the same
// model, with neither engine near DDR4-3200's ~51 GB/s — so the gap is compute, not bandwidth, and
// the kernel is doing ~2.4x more work per weight byte. P14 ruled out threading, the
// int4ParThreshold bug class, and the byte-count story (int4's 5.0 bits/weight against Q4_K_M's
// ~4.5 is ~11%, nowhere near 2.55x). What is left is the 4-bit machinery itself: nibble unpack,
// per-group scale handling, accumulator width.
//
// THE MEASUREMENT. dotI8AVX2 is the same MAC count on the same activations with none of that
// machinery — int8 weights, no unpack, no per-group scale. The two kernels' ratio isolates exactly
// the term in question, as a difference rather than an estimate:
//
//	dotI8AVX2        K MACs over K weight bytes             = 1.00 MAC/byte
//	dotW4A8FoldAVX2  K MACs over K/2 + (K/32)*4 bytes       = 1.60 MAC/byte
//
// W4A8 moves 1.6x FEWER bytes per MAC. On a bandwidth-bound path it would therefore be ~1.6x
// FASTER. Whatever it actually achieves against that 1.6x is the arithmetic cost of the 4-bit
// machinery, and that is the number that decides whether a Q4_K-style redesign — fewer scale loads
// via 6-bit scales shared over a 256-weight superblock — is aimed at the right term at all.
//
// WHAT IT CANNOT SAY. This is a micro-benchmark, L1-resident by construction, so it measures the
// kernel and not the memory system. It proves the primitive and never the composition: P14's 2.4x
// is an end-to-end figure over a whole decode, and no ratio here restates it. It bounds how much of
// that gap can live in this kernel.
func BenchmarkP14_W4A8OpsPerByte(b *testing.B) {
	if !hasAVX2 {
		b.Skip("no AVX2")
	}
	const group = 32
	// Real projection input dims from P14's own table (Qwen3.8-27B: hidden 5120, ffn 17408), plus
	// the 768 the sibling benchmarks use so the numbers line up with those.
	for _, K := range []int{768, 5120, 17408} {
		nGroups := K / group
		rng := rand.New(rand.NewSource(1))
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(256) - 128)
		}
		packed := make([]byte, K/2)
		for i := range packed {
			packed[i] = byte(rng.Intn(256))
		}
		w8 := make([]int8, K)
		for i := range w8 {
			w8[i] = int8(rng.Intn(256) - 128)
		}
		scales := make([]float32, nGroups)
		for i := range scales {
			scales[i] = float32(rng.Float64()*0.05 + 0.0001)
		}

		// bytes of WEIGHT-side traffic per call — the activations are shared across output rows in
		// a real matmul, so they are not what the format changes.
		i8Bytes := float64(K)
		w4Bytes := float64(K/2 + nGroups*4)

		b.Run(p14ShapeName("int8/dotI8AVX2", K), func(b *testing.B) {
			var acc int32
			b.ResetTimer()
			for range b.N {
				acc += dotI8AVX2(&act[0], &w8[0], K)
			}
			_ = acc
			s := b.Elapsed().Seconds()
			b.ReportMetric(float64(K)*float64(b.N)/s/1e9, "GMAC/s")
			b.ReportMetric(i8Bytes*float64(b.N)/s/1e9, "GB/s-weights")
			b.ReportMetric(float64(K)/i8Bytes, "MAC/byte")
		})

		b.Run(p14ShapeName("w4a8/dotW4A8FoldAVX2", K), func(b *testing.B) {
			var acc float32
			b.ResetTimer()
			for range b.N {
				acc += dotW4A8FoldAVX2(&act[0], &packed[0], &scales[0], nGroups)
			}
			_ = acc
			s := b.Elapsed().Seconds()
			b.ReportMetric(float64(K)*float64(b.N)/s/1e9, "GMAC/s")
			b.ReportMetric(w4Bytes*float64(b.N)/s/1e9, "GB/s-weights")
			b.ReportMetric(float64(K)/w4Bytes, "MAC/byte")
		})
	}
}

func p14ShapeName(prefix string, K int) string {
	switch K {
	case 5120:
		return prefix + "/K=5120(hidden)"
	case 17408:
		return prefix + "/K=17408(ffn)"
	default:
		return prefix + "/K=768"
	}
}

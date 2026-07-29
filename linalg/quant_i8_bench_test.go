package linalg

import "testing"

// quant_i8_bench_test.go exists because the int8 dot had NO benchmark, which is how
// it went unnoticed that it is slower PER MAC than the f32 register-blocked kernel on
// amd64 (perf-campaign-2026-07-28.md, finding B).
//
// Both sides report GMAC/s so the comparison is direct: a "4× smaller index" is only a
// win if the MAC rate holds up, and here it does not. K=768 keeps both operands
// L1-resident, so this measures the kernels rather than memory.
func BenchmarkDotI8VsF32_K768(b *testing.B) {
	const K = 768

	ai := make([]int8, K)
	bi := make([]int8, K)
	for i := range ai {
		ai[i], bi[i] = int8(i%127-63), int8((i*7)%127-63)
	}
	b.Run("int8/DotI8", func(b *testing.B) {
		var acc int32
		b.ResetTimer()
		for range b.N {
			acc += DotI8(ai, bi)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})

	af := make([]float32, K)
	for i := range af {
		af[i] = float32(i%17) * 0.1
	}
	cols := make([][]float32, 8)
	for c := range cols {
		cols[c] = make([]float32, K)
		for i := range cols[c] {
			cols[c][i] = float32((i+c)%13) * 0.1
		}
	}
	b.Run("f32/Dot8x4", func(b *testing.B) {
		var sums [32]float32
		b.ResetTimer()
		for range b.N {
			Dot8x4(&af[0], &cols[0][0], &cols[1][0], &cols[2][0], &cols[3][0],
				&cols[4][0], &cols[5][0], &cols[6][0], &cols[7][0], K/4, &sums)
		}
		// Dot8x4 computes 8 columns × K MACs per call.
		b.ReportMetric(8*float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}

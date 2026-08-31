//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
)

// BenchmarkP14_W4A8OpsPerByte is the arm64 half of goinfer P14's ops/byte question, and it exists
// to answer one thing the amd64 run cannot: is W4A8's ~3x MAC-throughput penalty against int8
// specific to the AVX2 kernel, or inherent to the format?
//
// That distinction decides where the work goes. If arm64's ratio is much better, the AVX2 kernel
// has its own headroom and the fix is a targeted amd64 one. If arm64 is equally penalised, the cost
// is in the W4A8 approach itself — per-group scales and a nibble unpack against a MAC that is
// otherwise one instruction — and the lever is the layout repack that
// goinfer's docs/task-w4a8-neon-bandwidth.md calls "the one real lever Gate 0 identified".
//
// The comparison mirrors the amd64 file exactly: dotI8SDOT is the same MAC count on the same
// activations with no unpack and no per-group scale.
//
//	dotI8SDOT        K MACs over K weight bytes           = 1.00 MAC/byte
//	dotW4A8FoldSDOT  K MACs over K/2 + (K/32)*4 bytes     = 1.60 MAC/byte
//
// PRIOR ART, so this is not re-derived: task-w4a8-neon-bandwidth.md records the kernel as
// ISSUE-LIMITED, and its Gate 1 items 1+2 (dropping the centering subtract) measured a 3%
// REGRESSION — v1 209 ns/call vs v2 215 ns at this same FFN shape — because any instructions added
// anywhere in the call cost real time. So this benchmark deliberately measures the SHIPPED kernel
// against int8; it does not re-open centering.
func BenchmarkP14_W4A8OpsPerByte(b *testing.B) {
	if !hasDotProd {
		b.Skip("no NEON DotProd")
	}
	const group = 32
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
		i8Bytes := float64(K)
		w4Bytes := float64(K/2 + nGroups*4)

		b.Run(p14ShapeNameARM("int8/dotI8SDOT", K), func(b *testing.B) {
			var acc int32
			b.ResetTimer()
			for range b.N {
				acc += dotI8SDOT(&act[0], &w8[0], K)
			}
			_ = acc
			s := b.Elapsed().Seconds()
			b.ReportMetric(float64(K)*float64(b.N)/s/1e9, "GMAC/s")
			b.ReportMetric(i8Bytes*float64(b.N)/s/1e9, "GB/s-weights")
		})

		b.Run(p14ShapeNameARM("w4a8/dotW4A8FoldSDOT", K), func(b *testing.B) {
			var acc float32
			b.ResetTimer()
			for range b.N {
				acc += dotW4A8FoldSDOT(&act[0], &packed[0], &scales[0], nGroups)
			}
			_ = acc
			s := b.Elapsed().Seconds()
			b.ReportMetric(float64(K)*float64(b.N)/s/1e9, "GMAC/s")
			b.ReportMetric(w4Bytes*float64(b.N)/s/1e9, "GB/s-weights")
		})
	}
}

func p14ShapeNameARM(prefix string, K int) string {
	switch K {
	case 5120:
		return prefix + "/K=5120(hidden)"
	case 17408:
		return prefix + "/K=17408(ffn)"
	default:
		return prefix + "/K=768"
	}
}

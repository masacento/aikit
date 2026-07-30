package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// BenchmarkRoundForms sizes perf-campaign item 26 before acting on it.
// math.Round is NOT an amd64 intrinsic (math.Trunc is), so it inlines ~12
// integer ops from floor.go; Trunc compiles to one ROUNDSD.
//
// No //go:noinline source here, deliberately. A first version routed every read
// through one, on the item-19 reflex of defeating constant folding — but the
// values come from a slice, which the compiler cannot fold anyway, and the
// added call per element cost ~2 ns and swamped the ~1 ns difference under test.
// The guard belongs where a CONSTANT would be folded, not on ordinary data.
func BenchmarkRoundForms(b *testing.B) {
	const n = 4096
	rng := rand.New(rand.NewSource(26))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 40)
	}
	dst := make([]float64, n)

	b.Run("math.Round", func(b *testing.B) {
		for b.Loop() {
			for i, v := range src {
				dst[i] = math.Round(float64(v))
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
	b.Run("Trunc+Copysign", func(b *testing.B) {
		for b.Loop() {
			for i, v := range src {
				x := float64(v)
				dst[i] = math.Trunc(x + math.Copysign(0.5, x))
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n), "ns/elem")
	})
	sinkRound = dst
}

var sinkRound []float64

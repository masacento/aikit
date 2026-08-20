package encoder

import (
	"math/rand"
	"testing"
)

// BenchmarkLog1pPoolInto reports ns/elem at V=30522 (a real SPLADE
// vocabulary size, encoder/splade.go's own comment). Reflects whichever
// build (scalar or GOEXPERIMENT=simd) it's run under, since log1pPoolInto is
// the same call either way. Mixed zero/positive input, matching the real
// pooled[] shape (a trained SPLADE's logits are "mostly negative by
// construction" per splade.go, so most vocab entries stay at 0).
func BenchmarkLog1pPoolInto(b *testing.B) {
	const V = 30522
	rng := rand.New(rand.NewSource(1))
	src := make([]float32, V)
	for i := range src {
		if rng.Float64() < 0.15 { // ~15% positive density, a plausible SPLADE shape
			src[i] = float32(rng.Float64() * 10)
		}
	}
	pooled := make([]float32, V)
	for b.Loop() {
		copy(pooled, src)
		log1pPoolInto(pooled)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*V), "ns/elem")
}

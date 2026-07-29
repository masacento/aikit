package ann

import "testing"

// BenchmarkFlatQueryShard is the arbiter for item 16. N·dim spans the
// flatParallelThreshold in both directions so the gate itself is exercised.
func BenchmarkFlatQueryShard(b *testing.B) {
	for _, tc := range []struct {
		name string
		n, d int
	}{
		{"N1k_d128", 1000, 128},
		{"N10k_d128", 10000, 128},
		{"N50k_d384", 50000, 384},
		{"N200k_d384", 200000, 384},
	} {
		f, q := buildFlatCase(tc.n, tc.d, "random")
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				sinkHits = f.Query(q, 10)
			}
		})
	}
}

var sinkHits []Hit

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

// BenchmarkFlatQueryFilter measures whether lifting the serial restriction on
// QueryFilter is worth a contract change. `keep` here is a read-only bitmap
// lookup — the documented use case (logical delete / live set), and about as
// cheap as a real filter gets, so this is the LEAST favourable case for
// parallelising: a costlier predicate would only widen the gap.
func BenchmarkFlatQueryFilter(b *testing.B) {
	for _, tc := range []struct {
		name string
		n, d int
		live float64 // fraction of ids kept
	}{
		{"N50k_d384_keep90", 50000, 384, 0.90},
		{"N200k_d384_keep90", 200000, 384, 0.90},
		{"N200k_d384_keep10", 200000, 384, 0.10},
	} {
		f, q := buildFlatCase(tc.n, tc.d, "random")
		// Bitmap live-set: one bit per id, read-only after construction.
		bits := make([]uint64, (tc.n+63)/64)
		for i := range tc.n {
			if float64(i%100)/100 < tc.live {
				bits[i/64] |= 1 << uint(i%64)
			}
		}
		keep := func(id int) bool { return bits[id/64]&(1<<uint(id%64)) != 0 }
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				sinkHits = f.QueryFilter(q, 10, keep)
			}
		})
	}
}

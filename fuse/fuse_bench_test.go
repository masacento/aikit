package fuse

import (
	"fmt"
	"math/rand"
	"testing"
)

// fuse had no benchmarks before perf-campaign A5, which is how RRF came to be
// 23.3% of a hybrid retrieval query (docs/internal/perf-amdahl-linux-amd64.md
// §2) without anyone noticing.
//
// SCOPE, stated up front because the percentage is misleading on its own: RRF is
// O(shortlist), not O(corpus). Its share of a query is large at repo scale and
// vanishes as the corpus grows — at n=1M it is under 0.1% of a query — and once
// a reranker is in the pipeline the entire retrieval stack is a thousandth of
// the query. This is a small-to-medium-corpus finding.

// rankings builds `lists` overlapping ranked id lists of length k, of the shape
// hybrid search produces: the lists agree on some of the head and diverge in the
// tail, so the fused key set is larger than one list and smaller than their sum.
func rankings(lists, k, overlap int, seed int64) [][]int {
	rng := rand.New(rand.NewSource(seed))
	shared := make([]int, overlap)
	for i := range shared {
		shared[i] = rng.Intn(k * lists)
	}
	out := make([][]int, lists)
	for l := range out {
		ids := make([]int, 0, k)
		ids = append(ids, shared...)
		for len(ids) < k {
			ids = append(ids, rng.Intn(1<<20))
		}
		rng.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
		out[l] = ids
	}
	return out
}

func BenchmarkRRF(b *testing.B) {
	for _, k := range []int{10, 50, 200, 1000} {
		rs := rankings(2, k, k/3, int64(k))
		b.Run(fmt.Sprintf("k%d/lists2", k), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkRRF = RRF(DefaultK, rs...)
			}
		})
	}
	// Three-way fusion (lexical + dense + sparse) at the shortlist size the
	// examples use.
	rs3 := rankings(3, 50, 20, 3)
	b.Run("k50/lists3", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkRRF = RRF(DefaultK, rs3...)
		}
	})
}

func BenchmarkRSF(b *testing.B) {
	for _, k := range []int{50, 1000} {
		rs := rankings(2, k, k/3, int64(k))
		scored := make([][]Scored[int], len(rs))
		for i, r := range rs {
			scored[i] = make([]Scored[int], len(r))
			for j, id := range r {
				scored[i][j] = Scored[int]{Key: id, Score: float64(len(r) - j)}
			}
		}
		b.Run(fmt.Sprintf("k%d", k), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkRSF = RSF(scored...)
			}
		})
	}
}

var (
	sinkRRF []Result[int]
	sinkRSF []Result[int]
)

package sparse

import (
	"math/rand"
	"sort"
	"testing"
)

// This package had no benchmarks at all before perf-campaign item 39, which is
// why it kept a full O(T log T) sort of the touched set for two items after
// bm25 replaced its own (item 44). The benchmark is the thing that made it
// visible; it stays so the next one does not go the same way.

// spladeCorpus builds a corpus shaped like SPLADE output: each document
// activates a few dozen DISTINCT terms out of a large vocabulary, drawn with a
// heavy head so a handful of terms appear nearly everywhere and most appear
// rarely.
//
// Distinct matters. A first version drew terms with repetition, which SparseVec
// permits and New stores as several postings for one (term, document) — a shape
// no encoder emits, and one that changes the corpus's whole impact
// distribution.
func spladeCorpus(nDocs, vocab, perDoc int, seed int64) []SparseVec {
	rng := rand.New(rand.NewSource(seed))
	docs := make([]SparseVec, nDocs)
	seen := make(map[uint32]bool, perDoc*2)
	for d := range docs {
		n := perDoc/2 + rng.Intn(perDoc)
		terms := make([]uint32, 0, n)
		weights := make([]float32, 0, n)
		clear(seen)
		for range n {
			u := rng.Float64()
			t := uint32(float64(vocab)*u*u*u) % uint32(vocab)
			if seen[t] {
				continue
			}
			seen[t] = true
			terms = append(terms, t)
			weights = append(weights, float32(0.05+rng.Float64()*2))
		}
		docs[d] = SparseVec{Terms: terms, Weights: weights}
	}
	return docs
}

// termsByDF returns every indexed term, most frequent first.
func termsByDF(ix *Index) []uint32 {
	out := make([]uint32, 0, len(ix.postings))
	for t := range ix.postings {
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool {
		if len(ix.postings[out[i]]) != len(ix.postings[out[j]]) {
			return len(ix.postings[out[i]]) > len(ix.postings[out[j]])
		}
		return out[i] < out[j] // deterministic across map iteration orders
	})
	return out
}

// BenchmarkQuery spans the shapes a sparse index actually sees. The 30-term one
// is the point: it is what a SPLADE expansion emits, and it is the shape item
// 11's note measured (a 30-term query over 22,185 postings). The 3-term shapes
// bracket it by selectivity.
func BenchmarkQuery(b *testing.B) {
	ix := New(spladeCorpus(200_000, 30_000, 40, 39))
	byDF := termsByDF(ix)
	n := len(byDF)
	rng := rand.New(rand.NewSource(391))

	// A few high-weight anchors and a long low-weight tail.
	splade := SparseVec{}
	for i := range 30 {
		var t uint32
		var w float32
		if i < 4 {
			t, w = byDF[rng.Intn(n/50)], float32(1.5+rng.Float64())
		} else {
			t, w = byDF[n/4+rng.Intn(n*3/4)], float32(0.05+rng.Float64()*0.4)
		}
		splade.Terms = append(splade.Terms, t)
		splade.Weights = append(splade.Weights, w)
	}
	mk := func(ts ...uint32) SparseVec {
		q := SparseVec{Terms: ts}
		for range ts {
			q.Weights = append(q.Weights, 1)
		}
		return q
	}
	for _, sh := range []struct {
		name string
		q    SparseVec
	}{
		{"splade30", splade},
		{"head+2rare", mk(byDF[0], byDF[n*3/4], byDF[n-5])},
		{"3head", mk(byDF[0], byDF[1], byDF[2])},
		{"3rare", mk(byDF[n-1], byDF[n-2], byDF[n-3])},
	} {
		postings := 0
		for _, t := range sh.q.Terms {
			postings += len(ix.postings[t])
		}
		b.Run(sh.name, func(b *testing.B) {
			b.ReportAllocs()
			b.Logf("%d postings", postings)
			b.ResetTimer()
			for b.Loop() {
				_ = ix.Query(sh.q, 10)
			}
		})
	}
}

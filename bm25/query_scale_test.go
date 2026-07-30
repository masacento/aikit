package bm25

import (
	"fmt"
	"math/rand"
	"testing"
)

// buildScaleIndex makes a corpus at the scale where finding A is visible. The existing
// bm25 benchmark tops out at 1,000 documents, where an 8 KB score array is L1-resident
// and the O(corpus) cost is invisible by construction — which is exactly why this went
// unnoticed (perf-campaign-2026-07-28.md, items 1 and 11).
func buildScaleIndex(n, docLen, vocab int) (*Index, []string) {
	rng := rand.New(rand.NewSource(7))
	terms := make([]string, vocab)
	for i := range terms {
		terms[i] = fmt.Sprintf("t%06d", i)
	}
	docs := make([][]string, n)
	for d := range docs {
		doc := make([]string, docLen)
		for j := range doc {
			// Zipf-ish: most mass on the head, so a random query term is selective.
			doc[j] = terms[int(rng.ExpFloat64()*float64(vocab)/6)%vocab]
		}
		docs[d] = doc
	}
	// A 3-term query from the TAIL of the vocab — selective, the case finding A is about.
	q := []string{terms[vocab-1], terms[vocab-2], terms[vocab-3]}
	return Build(docs), q
}

// TestTopK_touchedSetMatchesDenseScan is the equivalence gate for item 11. Selecting
// over the touched set must return EXACTLY what a full-corpus scan returns — same docs,
// same order — not merely an equally-defensible ranking.
//
// The risk being gated is the tie-break: topk keeps the FIRST-SEEN member of a tie, so
// which doc survives depends on push order. The dense scan pushed in ascending doc
// order; the touched set only reproduces that because it is sorted. Remove the sort in
// accum.sortTouched and this test fails on ties.
func TestTopK_touchedSetMatchesDenseScan(t *testing.T) {
	ix, q := buildScaleIndex(3000, 40, 300)
	for _, k := range []int{1, 5, 10, 50, -1} {
		got := ix.TopK(q, k)
		// Reference: the pre-item-11 shape — dense Scores, full range, same selector.
		dense := ix.Scores(q)
		want := denseTopK(dense, k)
		if len(got) != len(want) {
			t.Fatalf("k=%d: %d results, want %d", k, len(got), len(want))
		}
		for i := range want {
			if got[i].Doc != want[i].Doc || got[i].Score != want[i].Score {
				t.Fatalf("k=%d rank %d: got (doc %d, %v), want (doc %d, %v)",
					k, i, got[i].Doc, got[i].Score, want[i].Doc, want[i].Score)
			}
		}
	}
	// Vacuity: the fixture must actually produce ties, or the tie-break is untested.
	dense := ix.Scores(q)
	counts := map[float64]int{}
	ties := 0
	for _, s := range dense {
		if s > 0 {
			counts[s]++
			if counts[s] == 2 {
				ties++
			}
		}
	}
	if ties == 0 {
		t.Error("fixture produced no tied scores — the tie-break path is untested here")
	} else {
		t.Logf("touched-set selection ≡ dense scan across k∈{1,5,10,50,-1}; %d tied score groups exercised", ties)
	}
}

// denseTopK is the pre-item-11 selection, kept as the reference implementation.
func denseTopK(scores []float64, k int) []Result {
	type kv struct {
		d int
		s float64
	}
	var all []kv
	for d, s := range scores {
		if s > 0 {
			all = append(all, kv{d, s})
		}
	}
	// (score desc, doc asc) — the contract TopK documents.
	for i := range all {
		for j := i + 1; j < len(all); j++ {
			if all[j].s > all[i].s || (all[j].s == all[i].s && all[j].d < all[i].d) {
				all[i], all[j] = all[j], all[i]
			}
		}
	}
	if k >= 0 && k < len(all) {
		all = all[:k]
	}
	out := make([]Result, len(all))
	for i, e := range all {
		out[i] = Result{Doc: e.d, Score: e.s}
	}
	return out
}

// BenchmarkTopK spans both ends of query selectivity, because item 11's win is
// selectivity-dependent BY CONSTRUCTION: it replaces O(corpus) with O(postings
// touched). Reporting only the selective case would flatter it.
//
//	selective — 3 tail-of-vocab terms, a few hundred postings (finding A's shape)
//	common    — 3 head-of-vocab terms, touching a large fraction of the corpus
func BenchmarkTopK(b *testing.B) {
	ix, sel := buildScaleIndex(200_000, 120, 30_000)
	common := headTerms(ix, 3)
	b.Run("selective", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			_ = ix.TopK(sel, 10)
		}
	})
	b.Run("common", func(b *testing.B) {
		b.ReportAllocs()
		b.Logf("postings touched: %d of %d docs", touchedCount(ix, common), ix.N())
		b.ResetTimer()
		for range b.N {
			_ = ix.TopK(common, 10)
		}
	})
}

// headTerms returns the n highest-DF terms — the least selective query possible.
func headTerms(ix *Index, n int) []string {
	type td struct {
		t string
		d int
	}
	var all []td
	for t := range ix.terms {
		d := ix.DF(t)
		all = append(all, td{t, d})
	}
	for i := range all {
		for j := i + 1; j < len(all); j++ {
			if all[j].d > all[i].d {
				all[i], all[j] = all[j], all[i]
			}
		}
	}
	out := make([]string, 0, n)
	for i := 0; i < n && i < len(all); i++ {
		out = append(out, all[i].t)
	}
	return out
}

func touchedCount(ix *Index, q []string) int {
	a := ix.scoreQuery(q)
	defer putAccum(a)
	return len(a.touched)
}

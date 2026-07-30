package sparse

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// randCorpus builds n sparse vectors over a vocab of V terms, each with ~density
// non-zero positive weights — a stand-in for a SPLADE expansion.
func randCorpus(rng *rand.Rand, n, V, density int) []SparseVec {
	docs := make([]SparseVec, n)
	for d := range docs {
		seen := map[uint32]bool{}
		for range density {
			t := uint32(rng.IntN(V))
			if seen[t] {
				continue
			}
			seen[t] = true
			docs[d].Terms = append(docs[d].Terms, t)
			docs[d].Weights = append(docs[d].Weights, float32(rng.Float64()*2))
		}
	}
	return docs
}

// bruteScores is the independent reference: a dense per-document dot product,
// matching New's contract (skip weight ≤ 0; sum duplicate query terms).
func bruteScores(docs []SparseVec, q SparseVec) []float64 {
	qm := map[uint32]float64{}
	for i := 0; i < min(len(q.Terms), len(q.Weights)); i++ {
		qm[q.Terms[i]] += float64(q.Weights[i])
	}
	out := make([]float64, len(docs))
	for d, v := range docs {
		for i := 0; i < min(len(v.Terms), len(v.Weights)); i++ {
			if v.Weights[i] <= 0 {
				continue
			}
			out[d] += qm[v.Terms[i]] * float64(v.Weights[i])
		}
	}
	return out
}

func TestScores_matchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	docs := randCorpus(rng, 2000, 5000, 80)
	ix := New(docs)
	for qi := range 30 {
		q := randCorpus(rng, 1, 5000, 20)[0]
		got := ix.Scores(q)
		want := bruteScores(docs, q)
		for d := range want {
			if diff := math.Abs(got[d] - want[d]); diff > 1e-9*(math.Abs(want[d])+1) {
				t.Fatalf("query %d doc %d: score %.12f, brute %.12f (diff %.2e)", qi, d, got[d], want[d], diff)
			}
		}
	}
}

func TestQuery_topKMatchesSortedScores(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	docs := randCorpus(rng, 500, 1000, 40)
	ix := New(docs)
	q := randCorpus(rng, 1, 1000, 30)[0]

	scores := ix.Scores(q)
	// Reference top-k: every positive doc, sorted by (−score, +id).
	type sc struct {
		d int
		s float64
	}
	var ref []sc
	for d, s := range scores {
		if s > 0 {
			ref = append(ref, sc{d, s})
		}
	}
	sort.Slice(ref, func(i, j int) bool {
		if ref[i].s != ref[j].s {
			return ref[i].s > ref[j].s
		}
		return ref[i].d < ref[j].d
	})

	const k = 10
	got := ix.Query(q, k)
	if len(got) != k {
		t.Fatalf("Query returned %d hits, want %d", len(got), k)
	}
	for i, h := range got {
		if h.Index != ref[i].d || h.Score != ref[i].s {
			t.Errorf("rank %d: got {doc %d, %.6f}, want {doc %d, %.6f}", i, h.Index, h.Score, ref[i].d, ref[i].s)
		}
	}
}

func TestQuery_tieBreakByID(t *testing.T) {
	// Two docs with identical single-term weight → identical score → must come
	// back in ascending-id order.
	docs := []SparseVec{
		{Terms: []uint32{3}, Weights: []float32{1}}, // doc 0
		{Terms: []uint32{9}, Weights: []float32{0}}, // doc 1: zero weight, never scores
		{Terms: []uint32{3}, Weights: []float32{1}}, // doc 2: ties doc 0
	}
	ix := New(docs)
	got := ix.Query(SparseVec{Terms: []uint32{3}, Weights: []float32{1}}, 10)
	if len(got) != 2 || got[0].Index != 0 || got[1].Index != 2 {
		t.Fatalf("tie order: got %+v, want docs [0 2] ascending", got)
	}
}

func TestQuery_kSemantics(t *testing.T) {
	docs := []SparseVec{
		{Terms: []uint32{1}, Weights: []float32{3}},
		{Terms: []uint32{1}, Weights: []float32{2}},
		{Terms: []uint32{1}, Weights: []float32{1}},
		{Terms: []uint32{2}, Weights: []float32{5}}, // term 2: not in query → score 0
	}
	ix := New(docs)
	q := SparseVec{Terms: []uint32{1}, Weights: []float32{1}}

	if got := ix.Query(q, 0); len(got) != 0 {
		t.Errorf("k=0: got %d hits, want 0", len(got))
	}
	if got := ix.Query(q, -1); len(got) != 3 { // every positive doc
		t.Errorf("k<0: got %d hits, want 3 (positives only, doc 3 excluded)", len(got))
	}
	if got := ix.Query(q, 100); len(got) != 3 { // k>positives clamps
		t.Errorf("k>positives: got %d hits, want 3", len(got))
	}
	got := ix.Query(q, 2)
	if len(got) != 2 || got[0].Index != 0 || got[1].Index != 1 {
		t.Errorf("k=2: got %+v, want docs [0 1]", got)
	}
}

func TestNew_dupQuerySumsAndLenMismatchSafe(t *testing.T) {
	ix := New([]SparseVec{{Terms: []uint32{4}, Weights: []float32{1}}})
	// Duplicate query term 4 (0.5 + 0.5) and a trailing term with no weight (the
	// shorter slice bounds the walk) must not panic and must sum to 1.0·1.0.
	q := SparseVec{Terms: []uint32{4, 4, 7}, Weights: []float32{0.5, 0.5}}
	got := ix.Query(q, 1)
	if len(got) != 1 || math.Abs(got[0].Score-1.0) > 1e-6 {
		t.Fatalf("dup/len-mismatch: got %+v, want doc 0 score ~1.0", got)
	}
}

// TestScores_deterministic is the regression for AUDIT #16: Scores summed each
// document's contributions in Go map-iteration order (randomized per call), and
// float64 addition isn't associative, so identical queries returned different
// scores across calls and ties flipped. The weights here are chosen so the sum is
// strongly order-dependent (1 + 1e16 - 1e16 is 0 or 1 depending on order), making
// the old map-range path visibly non-deterministic; the fix sums in a fixed order.
func TestScores_deterministic(t *testing.T) {
	ix := New([]SparseVec{{Terms: []uint32{1, 2, 3}, Weights: []float32{1, 1, 1}}}) // doc 0 in all three terms
	q := SparseVec{Terms: []uint32{1, 2, 3}, Weights: []float32{1, 1e16, -1e16}}
	first := ix.Scores(q)[0]
	for i := range 500 {
		if got := ix.Scores(q)[0]; got != first {
			t.Fatalf("Scores non-deterministic: call %d = %v, first = %v (float64 sum order varied)", i, got, first)
		}
	}
}

// TestQuery_touchedOrderMatchesDenseScan is the equivalence gate for the
// merge in accum.orderTouched (the item-44 port). Selecting over the touched
// set must return EXACTLY what a full-corpus scan returns — same documents in
// the same order — not merely an equally-defensible ranking.
//
// The risk is entirely in the tie-break: topk keeps the FIRST-SEEN member of a
// tie, so which document survives depends on push order. The dense scan pushes
// in ascending document order, and the touched set only reproduces that because
// orderTouched merges the per-term runs. Break the merge and this fails — which
// nothing else did: the pre-existing tests use corpora whose scores rarely tie,
// so removing the ordering entirely left every one of them green.
//
// The weights here are small integers so that many documents land on exactly
// the same float64 score.
func TestQuery_touchedOrderMatchesDenseScan(t *testing.T) {
	const nDocs, vocab = 4000, 40
	rng := rand.New(rand.NewPCG(44, 44))
	docs := make([]SparseVec, nDocs)
	for d := range docs {
		var v SparseVec
		for range 3 + rng.IntN(4) {
			v.Terms = append(v.Terms, uint32(rng.IntN(vocab)))
			v.Weights = append(v.Weights, float32(1+rng.IntN(3)))
		}
		docs[d] = v
	}
	ix := New(docs)

	for qi := range 40 {
		var q SparseVec
		for range 2 + rng.IntN(5) {
			q.Terms = append(q.Terms, uint32(rng.IntN(vocab)))
			q.Weights = append(q.Weights, float32(1+rng.IntN(2)))
		}
		dense := ix.Scores(q)
		// Reference selection: (score desc, doc asc) over the whole corpus.
		type kv struct {
			d int
			s float64
		}
		var all []kv
		for d, s := range dense {
			if s > 0 {
				all = append(all, kv{d, s})
			}
		}
		sort.SliceStable(all, func(i, j int) bool {
			if all[i].s != all[j].s {
				return all[i].s > all[j].s
			}
			return all[i].d < all[j].d
		})
		for _, k := range []int{1, 5, 10, 50} {
			want := all
			if k < len(want) {
				want = want[:k]
			}
			got := ix.Query(q, k)
			if len(got) != len(want) {
				t.Fatalf("query %d k=%d: %d hits, dense scan %d", qi, k, len(got), len(want))
			}
			for i := range want {
				if got[i].Index != want[i].d || got[i].Score != want[i].s {
					t.Fatalf("query %d k=%d hit %d: got {%d %v}, dense scan {%d %v}",
						qi, k, i, got[i].Index, got[i].Score, want[i].d, want[i].s)
				}
			}
		}
	}
}

// TestQuery_tiesActuallyOccur guards the test above from silently becoming
// vacuous: if the fixture stops producing tied scores, the ordering it checks
// is no longer being checked.
func TestQuery_tiesActuallyOccur(t *testing.T) {
	const nDocs, vocab = 4000, 40
	rng := rand.New(rand.NewPCG(44, 44))
	docs := make([]SparseVec, nDocs)
	for d := range docs {
		var v SparseVec
		for range 3 + rng.IntN(4) {
			v.Terms = append(v.Terms, uint32(rng.IntN(vocab)))
			v.Weights = append(v.Weights, float32(1+rng.IntN(3)))
		}
		docs[d] = v
	}
	ix := New(docs)
	q := SparseVec{Terms: []uint32{1, 2, 3}, Weights: []float32{1, 1, 1}}
	counts := map[float64]int{}
	tied := 0
	for _, s := range ix.Scores(q) {
		if s > 0 {
			counts[s]++
			if counts[s] == 2 {
				tied++
			}
		}
	}
	if tied == 0 {
		t.Fatal("fixture produced no tied scores — the tie-break path is untested")
	}
	t.Logf("%d tied score groups exercised", tied)
}

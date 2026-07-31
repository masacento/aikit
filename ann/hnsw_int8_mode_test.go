package ann

import (
	"math/rand/v2"
	"testing"
)

// TestHNSW_int8ModeRecall builds a REAL int8-mode index — h.int8 true, so sim
// and simIDs take the quantized paths — and gates its recall against the exact
// scan.
//
// It exists because nothing did. The recall tests above measure int8 fidelity by
// dequantizing and building an f32 index, which is a faithful model of int8
// RANKING and never executes ann/hnsw.go's int8 branches at all. The gap was
// found by mutation: dropping `qv.scale` from `sim` alone — the exact "latent
// trap" documented on that function, and a textbook loop-invariant hoist —
// passes the entire ann suite otherwise.
//
// That mutation is what this test is calibrated against. It breaks
// selectHeuristic's cross-comparison: `e.sim` loses a factor of qv.scale while
// the simIDs side keeps both scales, so the "is this candidate closer to the base
// than to an already-selected neighbour" test flips for almost every candidate
// and the graph loses most of its edges. Recall collapses; a floor anywhere near
// the true value catches it.
func TestHNSW_int8ModeRecall(t *testing.T) {
	const n, dim, k = 3000, 96, 10
	rng := rand.New(rand.NewPCG(8, 8))
	// Clustered, so the top-k is a real neighbourhood rather than a coin flip —
	// on structureless data every method scores alike and no floor is meaningful
	// (measuring-performance.md §1.27).
	vecs, centers := clusteredCorpus(rng, 60, n/60, dim)

	h := NewHNSW(Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 8, Int8: true})
	for _, v := range vecs {
		h.Add(v)
	}
	if !h.int8 {
		t.Fatal("index is not in int8 mode; this test would exercise the f32 path")
	}
	exact := New(vecs)

	var total float64
	qrng := rand.New(rand.NewPCG(9, 9))
	for i, c := range centers {
		q := make([]float32, dim)
		for d := range q {
			q[d] = c[d] + float32(0.05*qrng.NormFloat64())
		}
		truth := map[int]bool{}
		for _, hit := range exact.Query(q, k) {
			truth[hit.Index] = true
		}
		total += recallOfHits(h.Query(q, k), truth, k)
		_ = i
	}
	recall := total / float64(len(centers))
	t.Logf("int8-mode HNSW recall@%d over %d queries, %d vectors: %.4f", k, len(centers), n, recall)
	if recall < 0.90 {
		t.Errorf("int8-mode recall@%d = %.4f, want >= 0.90", k, recall)
	}
}

// recallOfHits is recallAtK for package-internal tests (the exported-API version
// lives in hnsw_int8_recall_test.go, which is package ann_test).
func recallOfHits(got []Hit, truth map[int]bool, k int) float64 {
	hit := 0
	for _, h := range got {
		if truth[h.Index] {
			hit++
		}
	}
	return float64(hit) / float64(k)
}

// TestHNSW_simAndSimIDsAgree gates the matching-units invariant documented on
// sim, by asserting the invariant itself rather than its downstream effect.
//
// The invariant: for a node built from `vec`, sim(prepare(vec), c) and
// simIDs(id, c) are the SAME NUMBER, because Add quantizes the query and the
// stored row from the same vector with the same pure function. selectHeuristic
// compares one against the other, so the day they diverge the graph changes.
//
// Asserting it directly is what works. The obvious gate — build an int8 index and
// watch recall — does NOT catch it: TestHNSW_int8ModeRecall above scores 0.978
// with `qv.scale` dropped from sim, because on L2-normalized vectors every
// per-vector scale is nearly the same number, so the mutation is close to a
// uniform rescale of one side and the resulting graph is still good on clustered
// data. That is the trap in miniature — the failure is small, plausible, and
// invisible in aggregate quality — and it is why this test compares the two
// functions instead of grading the index they build.
func TestHNSW_simAndSimIDsAgree(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 11))
	vecs, _ := clusteredCorpus(rng, 20, 15, 64)

	h := NewHNSW(Config{M: 8, EfConstruction: 64, Seed: 11, Int8: true})
	for _, v := range vecs {
		h.Add(v)
	}
	if !h.int8 {
		t.Fatal("index is not in int8 mode; the invariant is trivial on the f32 path")
	}

	checked := 0
	for id, v := range vecs {
		qv := h.prepare(v)
		// The scale equality the invariant rests on.
		if qv.scale != float64(h.scales[id]) {
			t.Fatalf("node %d: prepare scale %v != stored scale %v — Add's query and row "+
				"are no longer quantized identically", id, qv.scale, h.scales[id])
		}
		for _, c := range []int{(id + 1) % len(vecs), (id + 7) % len(vecs), (id + 101) % len(vecs)} {
			a, b := h.sim(qv, c), h.simIDs(id, c)
			if a != b {
				t.Fatalf("node %d vs %d: sim=%v simIDs=%v — selectHeuristic compares these "+
					"two directly, so they must be in the same units", id, c, a, b)
			}
			checked++
		}
	}
	t.Logf("%d (node, candidate) pairs agree exactly", checked)
}

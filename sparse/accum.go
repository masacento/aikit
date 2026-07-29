package sparse

import (
	"slices"
	"sync"
)

// accum mirrors bm25/accum.go: a pooled accumulator that makes a query cost
// O(postings touched) instead of O(corpus) — perf-campaign-2026-07-28.md finding A /
// item 11, whose measurement covered this package too (a 30-term SPLADE query over
// 22,185 postings: 654 µs, almost none of it scoring).
//
// Generation stamps rather than a "score != 0 means touched" test: a sparse dot can
// legitimately produce a zero contribution (a zero-weight posting, or contributions
// that cancel), so the zero test would be wrong here even today — not merely fragile.
type accum struct {
	scores  []float64
	stamp   []uint32
	gen     uint32
	touched []int32
}

var accumPool = sync.Pool{New: func() any { return new(accum) }}

func getAccum(n int) *accum {
	a := accumPool.Get().(*accum)
	if cap(a.scores) < n {
		a.scores = make([]float64, n)
		a.stamp = make([]uint32, n)
		a.gen = 0
	}
	a.scores = a.scores[:n]
	a.stamp = a.stamp[:n]
	a.touched = a.touched[:0]
	a.gen++
	if a.gen == 0 { // wrapped
		clear(a.stamp)
		a.gen = 1
	}
	return a
}

func putAccum(a *accum) { accumPool.Put(a) }

func (a *accum) add(doc int32, v float64) {
	if a.stamp[doc] != a.gen {
		a.stamp[doc] = a.gen
		a.scores[doc] = 0
		a.touched = append(a.touched, doc)
	}
	a.scores[doc] += v
}

// sortTouched puts the touched ids in ascending doc order — load-bearing, because
// topk retains the FIRST-SEEN member of a tie, so reproducing the dense scan's exact
// output requires the dense scan's visit order. See bm25/accum.go.
func (a *accum) sortTouched() { slices.Sort(a.touched) }

// scoreQuery accumulates the query's postings. The caller must putAccum the result.
//
// Duplicate query terms are folded in FIRST-APPEARANCE order, preserved exactly: this
// package promises deterministic scores, and ranging a map here once made the float64
// accumulation order random per call, so identical queries produced 0.6 vs
// 0.6000000000000001 and ties flipped (audit #16). The dedupe is a linear scan over
// the accumulated terms rather than a map — a SPLADE query is tens of terms, where the
// map's allocation and hashing cost more than the scan (item 11, second-order) — and a
// scan preserves first-appearance order by construction.
func (ix *Index) scoreQuery(q SparseVec) *accum {
	n := min(len(q.Terms), len(q.Weights))
	type termWeight struct {
		term uint32
		w    float64
	}
	order := make([]termWeight, 0, n)
	for i := range n {
		t := q.Terms[i]
		found := false
		for j := range order {
			if order[j].term == t {
				order[j].w += float64(q.Weights[i])
				found = true
				break
			}
		}
		if !found {
			order = append(order, termWeight{t, float64(q.Weights[i])})
		}
	}
	a := getAccum(ix.ndocs)
	for _, tw := range order {
		if tw.w == 0 {
			continue
		}
		for _, p := range ix.postings[tw.term] {
			a.add(p.doc, tw.w*float64(p.w))
		}
	}
	a.sortTouched()
	return a
}

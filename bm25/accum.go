package bm25

import (
	"slices"
	"sync"
)

// accum is a pooled scoring accumulator that makes a query cost O(postings touched)
// instead of O(corpus) — perf-campaign-2026-07-28.md finding A / item 11.
//
// The old shape allocated `make([]float64, N)` per query (a fresh memclr of the whole
// span on the large-object path), wrote a handful of slots, then RE-READ all N for
// selection. Measured at 200k docs with a 3-term query touching 2,335 postings:
// ~65% allocate+zero, ~35% selection sweep, and under 1% actual scoring — the query
// itself was lost in the noise.
//
// The dense array stays (random-access `+=` beats a map), but it is pooled and only
// the touched slots are cleared, and selection runs over the touched set.
//
// WHY GENERATION STAMPS rather than "scores[doc] != 0 means touched": today every
// BM25 contribution is strictly positive (the Lucene idf is > 0 for any known term,
// tf >= 1, denom > 0), so a zero test would work. It would also break SILENTLY the
// day a scorer admits a zero or negative contribution — the doc would score, never
// enter the touched set, and vanish from results. A stamp costs 4 bytes per doc and
// cannot be wrong.
type accum struct {
	scores  []float64
	stamp   []uint32
	gen     uint32
	touched []int32
}

var accumPool = sync.Pool{New: func() any { return new(accum) }}

// get returns an accumulator sized for n docs, with a fresh generation. Clearing is
// O(1): bumping gen invalidates every stamp at once.
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
	if a.gen == 0 { // wrapped: the only case where stamps must actually be cleared
		clear(a.stamp)
		a.gen = 1
	}
	return a
}

func putAccum(a *accum) { accumPool.Put(a) }

// add applies one posting's contribution, recording the doc on its first touch.
func (a *accum) add(doc int, v float64) {
	if a.stamp[doc] != a.gen {
		a.stamp[doc] = a.gen
		a.scores[doc] = 0
		a.touched = append(a.touched, int32(doc))
	}
	a.scores[doc] += v
}

// sortTouched puts the touched ids in ascending doc order.
//
// This is not cosmetic. topk's selector keeps the FIRST-SEEN item of a tie (a tied
// newcomer is rejected at capacity), so which member of a tie survives depends on
// push order. The old code pushed in ascending doc order because it ranged over the
// dense array; visiting the touched set in the same order reproduces the identical
// result set, not merely an equally-valid one.
func (a *accum) sortTouched() { slices.Sort(a.touched) }

// scoreQuery accumulates every (deduped) query term's postings and returns the
// accumulator. The caller must putAccum it.
func (ix *Index) scoreQuery(query []string) *accum {
	a := getAccum(ix.N())
	for i, term := range query {
		if dupTerm(query, i) {
			continue // term contributes once; tf is per-document, not per-query
		}
		idf := ix.idf(term)
		if idf == 0 {
			continue
		}
		// K1/B are exported and may be tuned by the caller before querying, so
		// they cannot be folded into the postings at Build (item 10's
		// "precompute the impact" would bake them in). Hoisting them out of the
		// posting loop is free and keeps them live.
		k1, bb := ix.K1, ix.B
		k1p1 := k1 + 1
		for _, p := range ix.postings[term] {
			tf := float64(p.tf)
			denom := tf + k1*(1-bb+bb*ix.norm[p.doc])
			a.add(int(p.doc), idf*(tf*k1p1)/denom)
		}
	}
	a.sortTouched()
	return a
}

// dupTerm reports whether query[i] appeared earlier in query. A linear scan over the
// preceding terms replaces the per-query `seen` map: queries are tens of terms, where
// the map's allocation and hashing cost more than the scan (item 11, second-order).
func dupTerm(query []string, i int) bool {
	for j := range i {
		if query[j] == query[i] {
			return true
		}
	}
	return false
}

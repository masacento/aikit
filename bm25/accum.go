package bm25

import "sync"

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
	// runs holds the start offset in `touched` of each contributing term's
	// first-touch run, plus a terminating len(touched). Each run is ascending
	// by construction (Build appends postings in document order and `add`
	// records a doc on first touch only), so ordering `touched` is a MERGE of
	// sorted runs, not a sort — see orderTouched (item 44).
	runs  []int32
	merge []int32 // scratch destination for the merge
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
	a.runs = a.runs[:0]
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

// beginRun marks the start of a new term's postings in `touched`.
func (a *accum) beginRun() { a.runs = append(a.runs, int32(len(a.touched))) }

// orderTouched puts the touched ids in ascending doc order.
//
// This is not cosmetic. topk's selector keeps the FIRST-SEEN item of a tie (a tied
// newcomer is rejected at capacity), so which member of a tie survives depends on
// push order. The old code pushed in ascending doc order because it ranged over the
// dense array; visiting the touched set in the same order reproduces the identical
// result set, not merely an equally-valid one.
//
// It used to be a full slices.Sort — O(T log T), and measured at ~18% of a common
// 200k-doc query once item 11 had taken the posting scan off the critical path
// (item 44). But `touched` is never arbitrary: each term appends its first-touch
// docs in ascending order, so the slice is a concatenation of Q ascending runs for
// a Q-term query. Merging them is O(T log Q), and for the single-term case — one
// run, already sorted — it is free.
//
// The output is byte-for-byte what slices.Sort produced: both yield the ascending
// permutation, and the ids are distinct (a doc enters `touched` exactly once), so
// there is no tie for a merge to order differently.
func (a *accum) orderTouched() {
	// Close the final run.
	a.runs = append(a.runs, int32(len(a.touched)))
	// Drop empty runs (a term whose postings were all already touched).
	dst := a.runs[:0]
	for i := 0; i+1 < len(a.runs); i++ {
		if a.runs[i] < a.runs[i+1] {
			dst = append(dst, a.runs[i])
		}
	}
	dst = append(dst, int32(len(a.touched)))
	a.runs = dst

	nRuns := len(a.runs) - 1
	if nRuns <= 1 {
		return // one run (or none): already ascending
	}
	if nRuns == 2 {
		a.merge = mergeTwo(a.merge[:0], a.touched[a.runs[0]:a.runs[1]], a.touched[a.runs[1]:a.runs[2]])
		a.touched, a.merge = a.merge, a.touched
		return
	}
	// Few enough terms that pairwise merging beats a heap, and Q is bounded by
	// the query length in practice.
	for nRuns > 1 {
		a.merge = a.merge[:0]
		out := a.runs[:0]
		i := 0
		for ; i+1 < nRuns; i += 2 {
			out = append(out, int32(len(a.merge)))
			a.merge = mergeTwo(a.merge, a.touched[a.runs[i]:a.runs[i+1]], a.touched[a.runs[i+1]:a.runs[i+2]])
		}
		if i < nRuns { // odd run out: carry it forward unchanged
			out = append(out, int32(len(a.merge)))
			a.merge = append(a.merge, a.touched[a.runs[i]:a.runs[i+1]]...)
		}
		out = append(out, int32(len(a.merge)))
		a.touched, a.merge = a.merge, a.touched
		a.runs = out
		nRuns = len(a.runs) - 1
	}
}

// mergeTwo appends the ascending merge of two ascending slices to dst.
func mergeTwo(dst, x, y []int32) []int32 {
	i, j := 0, 0
	for i < len(x) && j < len(y) {
		if x[i] <= y[j] {
			dst = append(dst, x[i])
			i++
		} else {
			dst = append(dst, y[j])
			j++
		}
	}
	dst = append(dst, x[i:]...)
	return append(dst, y[j:]...)
}

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
		a.beginRun()
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
	a.orderTouched()
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

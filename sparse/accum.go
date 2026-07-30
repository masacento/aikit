package sparse

import "sync"

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
	// runs holds the start offset in `touched` of each contributing term's
	// first-touch run, plus a terminating len(touched) — see orderTouched.
	runs  []int32
	merge []int32 // scratch destination for the merge
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
	a.runs = a.runs[:0]
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

// beginRun marks the start of a new term's postings in `touched`.
func (a *accum) beginRun() { a.runs = append(a.runs, int32(len(a.touched))) }

// orderTouched puts the touched ids in ascending doc order — load-bearing,
// because topk retains the FIRST-SEEN member of a tie, so reproducing the dense
// scan's exact output requires the dense scan's visit order.
//
// It used to be a full slices.Sort. That is O(T log T), and on a 30-term SPLADE
// query touching ~9,200 documents it was the single largest cost in the query —
// larger than scoring all 9,421 postings. bm25 fixed exactly this in item 44
// and this package was left behind; item 39's benchmarks put it back in view.
//
// `touched` is never arbitrary: each term appends its first-touch documents in
// ascending order (New appends postings in document order, and `add` records a
// document on first touch only), so the slice is a concatenation of Q ascending
// runs for a Q-term query. Merging them is O(T log Q), and for a single term —
// one run, already sorted — it is free.
//
// The output is byte-for-byte what slices.Sort produced: both yield the
// ascending permutation, and the ids are distinct (a document enters `touched`
// exactly once), so there is no tie for a merge to order differently.
func (a *accum) orderTouched() {
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
		return
	}
	if nRuns == 2 {
		a.merge = mergeTwo(a.merge[:0], a.touched[a.runs[0]:a.runs[1]], a.touched[a.runs[1]:a.runs[2]])
		a.touched, a.merge = a.merge, a.touched
		return
	}
	// Pairwise merging: Q is the query length, small enough that this beats a
	// heap.
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

// termWeight is one distinct query term and its folded weight.
type termWeight struct {
	term uint32
	w    float64
}

// foldQuery collapses duplicate query terms into dst, summing their weights and
// preserving FIRST-APPEARANCE order.
//
// That order is not cosmetic and this is the single implementation of it, used
// by both the exhaustive scan and the pruning path (item 39). This package
// promises deterministic scores; ranging a map here once made the float64
// accumulation order random per call, so identical queries produced 0.6 vs
// 0.6000000000000001 and ties flipped (audit #16). Two implementations of the
// fold would reintroduce exactly that divergence between the two paths, which
// is why the pruning path calls this rather than repeating it.
//
// The dedupe is a linear scan rather than a map: a SPLADE query is tens of
// terms, where the map's allocation and hashing cost more than the scan (item
// 11, second-order), and a scan preserves first-appearance order by
// construction.
func foldQuery(q SparseVec, dst []termWeight) []termWeight {
	n := min(len(q.Terms), len(q.Weights))
	for i := range n {
		t := q.Terms[i]
		found := false
		for j := range dst {
			if dst[j].term == t {
				dst[j].w += float64(q.Weights[i])
				found = true
				break
			}
		}
		if !found {
			dst = append(dst, termWeight{t, float64(q.Weights[i])})
		}
	}
	return dst
}

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
	order := foldQuery(q, nil)
	a := getAccum(ix.ndocs)
	for _, tw := range order {
		if tw.w == 0 {
			continue
		}
		a.beginRun()
		for _, p := range ix.postings[tw.term] {
			a.add(p.doc, tw.w*float64(p.w))
		}
	}
	a.orderTouched()
	return a
}

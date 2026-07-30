// Package fuse combines multiple ranked result lists into one — the
// reusable primitive behind hybrid search (e.g. blending a BM25 lexical
// ranking with a dense/ANN ranking). It is a leaf: like topk, it
// depends only on the standard library and knows nothing about bm25,
// ann, or embeddings. The wiring that produces the rankings and the
// policy for how to weight them is the consumer's job (glue lives in
// ken); fuse just does the math.
//
// The default algorithm is Reciprocal Rank Fusion (Cormack, Clarke &
// Büttcher, SIGIR 2009): each list contributes 1/(k+rank) to every item it
// ranks, summed across lists. RRF is rank-based, not score-based, which is
// its whole point — it fuses a BM25 score (unbounded, log-ish) and a cosine
// similarity (bounded [-1,1]) without needing to normalize the two
// incomparable score scales. Only the *order* of each list matters.
//
// When the per-list scores ARE calibrated (cosine sims, BM25 within one
// corpus), RSF (Relative Score Fusion, rsf.go) is the alternative: it min-max
// normalizes each list's scores to [0,1] and sums them, so it keeps how much
// better one hit is than the next — information RRF discards. Pick RRF for
// robustness to incomparable/noisy scores, RSF when magnitude is meaningful.
//
// Keys are generic and comparable, so callers fuse on whatever id they
// index by — an int chunk index (ann.Hit.Index, bm25.Result.Doc) or a
// string document id. The Keys helper extracts the id slice from a typed
// result list in one line:
//
//	dense  := fuse.Keys(annHits,    func(h ann.Hit) int    { return h.Index })
//	lexical := fuse.Keys(bm25Hits,  func(r bm25.Result) int { return r.Doc })
//	fused  := fuse.RRF(60, dense, lexical)
package fuse

import "slices"

// DefaultK is the rank-fusion constant from the original RRF paper. It
// damps the influence of top ranks: larger k makes the fusion flatter
// (later ranks matter relatively more), smaller k sharpens it toward the
// very top of each list. 60 is the well-tested default; pass it
// explicitly so the knob is always visible at the call site.
const DefaultK = 60.0

// Result is one fused item, highest Score first.
type Result[K comparable] struct {
	Key   K
	Score float64
}

// RRF fuses rankings (each ordered best-first, rank 1 = element 0) by
// unweighted reciprocal rank fusion with constant k. Equivalent to
// RRFWeighted with every weight 1.
//
// Returns items by descending fused score. Ties are broken by
// first-appearance order across the input rankings (scanning ranking 0
// first, then 1, …), so the result is deterministic for a given input.
// k must be > 0 (it's the denominator); RRF panics otherwise, matching
// the "programmer error, not runtime condition" convention.
func RRF[K comparable](k float64, rankings ...[]K) []Result[K] {
	return RRFWeighted(k, nil, rankings...)
}

// RRFWeighted is RRF with a per-ranking weight, the practical knob for
// hybrid search: weight the dense and lexical lists differently (the "α
// blend" in retrieval terms) without having to normalize their raw
// scores. Item contribution from ranking r is weights[r]/(k+rank).
//
// weights may be nil (all 1) or must have one entry per ranking; a
// length mismatch panics. A weight of 0 disables that ranking's
// contribution but it still participates in first-appearance tie order.
func RRFWeighted[K comparable](k float64, weights []float64, rankings ...[]K) []Result[K] {
	if k <= 0 {
		panic("fuse: RRF k must be > 0")
	}
	if weights != nil && len(weights) != len(rankings) {
		panic("fuse: len(weights) must equal number of rankings")
	}

	// Accumulate fused score per key, in first-appearance order so ties resolve
	// deterministically.

	// FIRST-SEEN ORDER IS NOW POSITIONAL, WHICH IS THE WHOLE CHANGE.
	//
	// This used to carry a second map from key to first-appearance index and consult
	// it inside the sort comparator — a map lookup per comparison, so O(n log n) of
	// them, which at k=1000 was most of the call. Accumulating into `out` directly,
	// in first-appearance order, makes that ordering the slice's own positions: a
	// STABLE sort on score alone then breaks ties by first appearance, identically,
	// with no lookup at all.
	//
	// Three consequences, all measured in A5: one map instead of two, the
	// key-to-index map replacing the score map rather than joining it; no second
	// pass to project the map into a slice; and slices.SortStableFunc in place of
	// sort.SliceStable, which sorted through reflect.Swapper.
	//
	// The stability is load-bearing now in a way it was not before. Previously the
	// comparator was a total order (first-seen is unique per key) and stability was
	// irrelevant; now the tie-break lives in the element positions, so a non-stable
	// sort would reorder tied keys. Do not "optimize" SortStableFunc to SortFunc.
	//
	// The maps are presized to the total input length — the upper bound on
	// distinct keys. Rankings overlap by construction (fusing disjoint lists is
	// not what RRF is for), so this over-allocates, which costs one larger
	// bucket array against the several rehashes an unsized map performs while
	// growing.
	total := 0
	for _, ranking := range rankings {
		total += len(ranking)
	}
	pos := make(map[K]int, total)
	out := make([]Result[K], 0, total)

	for r, ranking := range rankings {
		w := 1.0
		if weights != nil {
			w = weights[r]
		}
		for rank0, key := range ranking {
			i, ok := pos[key]
			if !ok {
				i = len(out)
				pos[key] = i
				out = append(out, Result[K]{Key: key})
			}
			// rank is 1-based: element 0 is rank 1.
			out[i].Score += w / (k + float64(rank0+1))
		}
	}

	slices.SortStableFunc(out, func(a, b Result[K]) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		}
		return 0
	})
	return out
}

// Keys projects a typed result slice down to its id slice, preserving
// order — the one-line adapter from []ann.Hit / []bm25.Result (or any
// custom result type) to the []K that RRF consumes.
func Keys[T any, K comparable](items []T, key func(T) K) []K {
	out := make([]K, len(items))
	for i, it := range items {
		out[i] = key(it)
	}
	return out
}

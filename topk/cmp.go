package topk

import "cmp"

// Cmp orders by descending score, then ascending id — the standard tie-break
// for a ranked result: sort by relevance, and resolve equal scores
// deterministically by id rather than depending on sort stability or input
// order. aikit's top-k consumers (ann's Hit/cand, bm25's Result, sparse's
// Hit) each independently reimplemented this over their own result types;
// this is the one implementation. It takes the id/score pair rather than a
// whole struct, so a caller's result type doesn't need to share a type or
// interface with anyone else's — just call it from a one-line wrapper:
//
//	func hitCmp(a, b Hit) int { return topk.Cmp(a.Index, b.Index, a.Score, b.Score) }
func Cmp[T cmp.Ordered](aID, bID T, aScore, bScore float64) int {
	switch {
	case aScore > bScore:
		return -1
	case aScore < bScore:
		return 1
	case aID < bID:
		return -1
	case aID > bID:
		return 1
	}
	return 0
}

// ItemCmp is Cmp specialized for ItemWithScore — the shape Selector.Result()
// returns — matching slices.SortFunc's signature directly, so callers pass
// topk.ItemCmp with no wrapper needed.
func ItemCmp[T cmp.Ordered](a, b ItemWithScore[T]) int {
	return Cmp(a.Item, b.Item, a.Score, b.Score)
}

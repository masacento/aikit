package ner

import (
	"sort"
	"strings"
)

// gliner2_decode.go — thresholding and overlap resolution, ported from
// gliner2/inference/{candidate_decoder,overlap}.py and the entity branch of
// BoundaryExtractor._decode_entities (engine.py). Half-open token spans,
// per-query resolution, confidence-descending output.

// g2Scored is one thresholded candidate: (probability, start, end) in WORD
// coordinates — RawSpan/Tuple[float,int,int] in the reference.
type g2Scored struct {
	Score      float32
	Start, End int
	Index      int // original order, for tie-breaking
}

type g2QuerySpan struct {
	Query int
	Span  g2Scored
}

// selectGrouped resolves overlaps within each query and, unless multi-label
// output was requested, keeps only the highest-scoring label for an exact span.
// Ties retain the earlier schema label, matching declaration-order output.
func selectGrouped(grouped [][]g2Scored, policy string, multiLabel bool) []g2QuerySpan {
	var all []g2QuerySpan
	for q, scored := range grouped {
		for _, span := range resolveSpans(scored, policy) {
			all = append(all, g2QuerySpan{Query: q, Span: span})
		}
	}
	if multiLabel {
		return all
	}
	best := make(map[[2]int]int, len(all))
	for i, candidate := range all {
		key := [2]int{candidate.Span.Start, candidate.Span.End}
		if prev, ok := best[key]; !ok || candidate.Span.Score > all[prev].Span.Score {
			best[key] = i
		}
	}
	out := make([]g2QuerySpan, 0, len(best))
	for i, candidate := range all {
		key := [2]int{candidate.Span.Start, candidate.Span.End}
		if best[key] == i {
			out = append(out, candidate)
		}
	}
	return out
}

// thresholdCandidates is _group_scored_candidates for B=1: sigmoid(pair_logits /
// pair_temperature), keep probs >= threshold per query. Abstention (a null
// probability above the threshold) empties the query's candidates.
func (g *GLiNER2) thresholdCandidates(m *g2Marginals, pool g2Pool, scores []float32, threshold float64, nullLogits []float32) [][]g2Scored {
	Q := m.Q
	C := len(pool.Indices)
	invT := 1 / g.pairTemperature()
	out := make([][]g2Scored, Q)
	for q := range Q {
		if len(nullLogits) > q && sigmoid(nullLogits[q]) > float32(g.head.cfg.AbstentionThreshold) {
			continue // the query abstains: no entities of this type
		}
		for c := range C {
			if !pool.Valid[c] {
				continue
			}
			prob := sigmoid(scores[c*Q+q] * float32(invT))
			if float64(prob) >= threshold {
				out[q] = append(out[q], g2Scored{
					Score: prob,
					Start: pool.Indices[c][0],
					End:   pool.Indices[c][1],
					Index: c,
				})
			}
		}
	}
	return out
}

func (g *GLiNER2) pairTemperature() float64 {
	if g.head.cfg.PairTemperature == 0 {
		return 1
	}
	return g.head.cfg.PairTemperature
}

// resolveSpans dispatches on the canonical overlap policy ("flat" and "nested"
// are what the public Opts expose). resolveFlat is overlap.py resolve_overlaps
// with the "disallow" (flat) policy: the maximum-total-score non-overlapping
// subset via weighted interval scheduling, with the reference's deterministic
// tie-breaks.
func resolveSpans(scored []g2Scored, policy string) []g2Scored {
	if len(scored) == 0 {
		return nil
	}
	// Rank, then collapse exact-boundary duplicates to their best rank.
	ranked := make([]g2Scored, len(scored))
	copy(ranked, scored)
	sort.SliceStable(ranked, func(a, b int) bool { return rankLess(ranked[a], ranked[b]) })
	distinct := ranked[:0:0]
	seen := map[[2]int]bool{}
	for _, s := range ranked {
		b := [2]int{s.Start, s.End}
		if seen[b] {
			continue
		}
		seen[b] = true
		distinct = append(distinct, s)
	}

	switch policy {
	case "nested":
		kept := distinct[:0:0]
		for _, cand := range distinct {
			crossing := false
			for _, ex := range kept {
				overlaps := cand.Start < ex.End && ex.Start < cand.End
				contains := (cand.Start <= ex.Start && ex.End <= cand.End) ||
					(ex.Start <= cand.Start && cand.End <= ex.End)
				if overlaps && !contains {
					crossing = true
					break
				}
			}
			if !crossing {
				kept = append(kept, cand)
			}
		}
		return kept
	case "allow":
		return distinct
	}

	// "flat": weighted interval scheduling. Ties prefer the lexicographically
	// better confidence/start/end ranking (overlap.py:148-196).
	byEnd := make([]g2Scored, len(distinct))
	copy(byEnd, distinct)
	sort.SliceStable(byEnd, func(a, b int) bool {
		x, y := byEnd[a], byEnd[b]
		if x.End != y.End {
			return x.End < y.End
		}
		if x.Start != y.Start {
			return x.Start < y.Start
		}
		if x.Score != y.Score {
			return x.Score > y.Score
		}
		return x.Index < y.Index
	})
	ends := make([]int, len(byEnd))
	for i, s := range byEnd {
		ends[i] = s.End
	}
	type selection struct {
		score float64
		items []int
	}
	best := make([]selection, len(byEnd)+1)
	best[0] = selection{score: 0, items: nil}
	for i, item := range byEnd {
		// predecessors[i] = last interval ending at or before item.Start.
		lo, hi := 0, i // bisect_right(ends, item.Start, 0, i)
		for lo < hi {
			mid := (lo + hi) / 2
			if ends[mid] <= item.Start {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		pred := lo - 1

		withItem := append(append([]int{}, best[pred+1].items...), i)
		withScore := best[pred+1].score + float64(item.Score)
		without := best[i]
		switch {
		case withScore > without.score:
			best[i+1] = selection{withScore, withItem}
		case withScore < without.score:
			best[i+1] = without
		case len(withItem) > len(without.items):
			best[i+1] = selection{withScore, withItem}
		case len(withItem) < len(without.items):
			best[i+1] = without
		default:
			wk, wo := selectionKey(withItem, byEnd), selectionKey(without.items, byEnd)
			if keyLess(wk, wo) {
				best[i+1] = selection{withScore, withItem}
			} else {
				best[i+1] = without
			}
		}
	}

	selected := make([]g2Scored, 0, len(best[len(byEnd)].items))
	for _, i := range best[len(byEnd)].items {
		selected = append(selected, byEnd[i])
	}
	sort.SliceStable(selected, func(a, b int) bool { return rankLess(selected[a], selected[b]) })
	return selected
}

// rankLess is rank_key: (-score, start, end, original index).
func rankLess(a, b g2Scored) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	if a.Start != b.Start {
		return a.Start < b.Start
	}
	if a.End != b.End {
		return a.End < b.End
	}
	return a.Index < b.Index
}

// selectionKey is overlap.py's selection_key: the rank keys of the selection's
// members, themselves sorted by rank key, compared lexicographically.
func selectionKey(items []int, byEnd []g2Scored) [][4]float64 {
	rows := make([]g2Scored, len(items))
	for i, ix := range items {
		rows[i] = byEnd[ix]
	}
	sort.SliceStable(rows, func(a, b int) bool { return rankLess(rows[a], rows[b]) })
	key := make([][4]float64, len(rows))
	for i, r := range rows {
		key[i] = [4]float64{-float64(r.Score), float64(r.Start), float64(r.End), float64(r.Index)}
	}
	return key
}

func keyLess(a, b [][4]float64) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return a[i][j] < b[i][j]
			}
		}
	}
	return len(a) < len(b)
}

// trimSurface mirrors the reference's `surface = text[char_start:char_end].strip()`.
func trimSurface(s string) string {
	return strings.TrimFunc(s, isPySpace)
}

package ner

import (
	"math"
	"sort"
)

// gliner2_pool.go — the shared document candidate pool (DocumentCandidatePool)
// and the pooled reranker (SharedPoolScorer), ported from gliner2/models/boundary/
// pool.py with the attention stacks the checkpoint disables (see
// gliner2_boundary.go). Internal ordering is candidate-major [C,Q] exactly like
// the reference; internal [B,C,Q] tensors collapse to [C] / [C,Q] for B=1.
//
// Determinism is load-bearing: every ordering in this file must reproduce
// torch.argsort(stable=True) — value-descending (or ascending, as noted) with
// ties broken by ORIGINAL index, never by a sort that shuffles equal keys.

// g2Pool is PooledCandidates for B=1: query-agnostic span candidates.
type g2Pool struct {
	Indices  [][2]int  // [C] half-open (start, end) boundary indices
	Valid    []bool    // [C]
	Compat   []float32 // [C] selected_compat (zeroed where invalid)
	Proposal []float32 // [C] selected_score (g2MaskLogit where invalid)
}

// buildPool is DocumentCandidatePool.forward: a query-agnostic union of the
// top boundary marginals, one Cartesian pairing pass, a per-query quota
// reservation, then dedup + capacity truncation.
func (h *g2Head) buildPool(m *g2Marginals, bs g2BoundaryStates) g2Pool {
	N, d := m.N, h.cfg.BoundaryDim
	Q := m.Q
	topK := h.cfg.PoolBoundaryTopK

	// Query-conditioned endpoint union. q_boundary is all-true here (every
	// query is valid), so the floor only applies where the boundary is invalid.
	unionStart := make([]float32, N)
	unionEnd := make([]float32, N)
	for i := range N {
		best := float32(g2MaskLogit)
		for q := range Q {
			if v := m.Start[q*N+i]; v > best {
				best = v
			}
		}
		unionStart[i] = best
		best = float32(g2MaskLogit)
		for q := range Q {
			if v := m.End[q*N+i]; v > best {
				best = v
			}
		}
		unionEnd[i] = best
	}
	// select_top_boundaries: masked stable sort, descending, ties by index.
	// Every boundary is valid for B=1, so validity is all-true.
	starts := stableTopKDesc(unionStart, topK)
	ends := stableTopKDesc(unionEnd, topK)

	// One query-agnostic Cartesian pass: for each selected start, all ends.
	ks, ke := len(starts), len(ends)
	P := ks * ke
	pairS := make([]int, P)
	pairE := make([]int, P)
	pairValid := make([]bool, P)
	for si := range ks {
		for ei := range ke {
			p := si*ke + ei
			pairS[p], pairE[p] = starts[si], ends[ei]
			pairValid[p] = ends[ei] > starts[si]
		}
	}

	startAll := h.poolStartProj.apply(bs.States, N)
	endAll := h.poolEndProj.apply(bs.States, N)
	dotd := func(a, b []float32) float32 {
		var s float32
		for i := range a {
			s += a[i] * b[i]
		}
		return float32(float64(s) / math.Sqrt(float64(d)))
	}
	compat := make([]float32, P)
	ups := make([]float32, P) // union_pair_score
	for p := range P {
		compat[p] = dotd(startAll[pairS[p]*d:(pairS[p]+1)*d], endAll[pairE[p]*d:(pairE[p]+1)*d])
		ups[p] = compat[p] + unionStart[pairS[p]] + unionEnd[pairE[p]]
	}

	// Quota: reserve each query's strongest pairs ahead of the global fill.
	// quota_scores sit in a huge priority band (−MASK_LOGIT·0.5 = +5000) with a
	// small rank bonus so quota order itself is deterministic.
	quota := h.cfg.MinPoolPerQuery
	if quota > P {
		quota = P
	}
	all := make([]g2PoolEntry, 0, P+Q*quota)
	if quota > 0 {
		for q := range Q {
			perQuery := make([]float32, P)
			for p := range P {
				v := compat[p] + m.Start[q*N+pairS[p]] + m.End[q*N+pairE[p]]
				if !pairValid[p] {
					v = g2MaskLogit
				}
				perQuery[p] = v
			}
			ranked := stableTopKDesc(perQuery, quota)
			for r, p := range ranked {
				all = append(all, g2PoolEntry{
					key:   int64(pairS[p])*int64(N) + int64(pairE[p]),
					score: float32(-g2MaskLogit)*0.5 + float32(quota-r),
					valid: pairValid[p],
				})
			}
		}
	}
	for p := range P {
		all = append(all, g2PoolEntry{
			key:   int64(pairS[p])*int64(N) + int64(pairE[p]),
			score: ups[p],
			valid: pairValid[p],
		})
	}

	keys, valid := h.dedupePool(all, N, h.cfg.PoolSize)
	C := len(keys)

	// Recompute the differentiable scores only for retained candidates.
	pool := g2Pool{
		Indices:  make([][2]int, C),
		Valid:    valid,
		Compat:   make([]float32, C),
		Proposal: make([]float32, C),
	}
	for c := range C {
		s, e := int(keys[c]/int64(N)), int(keys[c]%int64(N))
		if !valid[c] {
			s, e = 0, 0
		}
		pool.Indices[c] = [2]int{s, e}
		var cp float32
		if valid[c] {
			cp = dotd(startAll[s*d:(s+1)*d], endAll[e*d:(e+1)*d])
			pool.Proposal[c] = cp + unionStart[s] + unionEnd[e]
		} else {
			pool.Proposal[c] = g2MaskLogit
		}
		pool.Compat[c] = cp
	}
	return pool
}

// g2PoolEntry is one (key, priority) row fed to the pool dedup. key = s*N+e.
type g2PoolEntry struct {
	key   int64
	score float32
	valid bool
}

// dedupePool is _deduplicate_pool: invalid entries sink to a shared sentinel
// key, a stable score-desc sort puts each key's best occurrence first, a stable
// key-asc sort groups duplicates so first-of-run wins, and the survivors are
// truncated to capacity by masked score again. Returned keys are zeroed where
// invalid and the slice is padded to capacity.
func (h *g2Head) dedupePool(all []g2PoolEntry, n, capacity int) ([]int64, []bool) {
	invalidKey := int64(n) * int64(n)
	keys := make([]int64, len(all))
	scores := make([]float32, len(all))
	valid := make([]bool, len(all))
	for i, e := range all {
		if e.valid {
			keys[i], scores[i], valid[i] = e.key, e.score, true
		} else {
			keys[i], scores[i], valid[i] = invalidKey, g2MaskLogit, false
		}
	}

	// Score-desc (stable), then key-asc (stable): within equal keys the first
	// row is the one with the best score.
	reorder := func(order []int) {
		k2 := make([]int64, len(order))
		s2 := make([]float32, len(order))
		v2 := make([]bool, len(order))
		for i, o := range order {
			k2[i], s2[i], v2[i] = keys[o], scores[o], valid[o]
		}
		keys, scores, valid = k2, s2, v2
	}
	reorder(stableSortIdx(len(all), func(a, b int) bool { return scores[a] > scores[b] }))
	reorder(stableSortIdx(len(all), func(a, b int) bool { return keys[a] < keys[b] }))

	keep := make([]bool, len(all))
	for i := range keep {
		keep[i] = valid[i] && (i == 0 || keys[i] != keys[i-1])
	}

	masked := make([]float32, len(all))
	for i := range masked {
		if keep[i] {
			masked[i] = scores[i]
		} else {
			masked[i] = g2MaskLogit
		}
	}
	order := stableTopKDesc(masked, len(masked))
	if capacity < len(order) {
		order = order[:capacity]
	}
	outKeys := make([]int64, capacity)
	outValid := make([]bool, capacity)
	for i, o := range order {
		outKeys[i], outValid[i] = keys[o], keep[o]
	}
	for i := range outKeys {
		if !outValid[i] {
			outKeys[i] = 0
		}
	}
	return outKeys, outValid
}

// stableSortIdx returns [0,n) sorted by less, preserving original order on ties.
func stableSortIdx(n int, less func(a, b int) bool) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return less(idx[a], idx[b]) })
	return idx
}

// scorePool is SharedPoolScorer.forward: candidate features computed once,
// scored against every query, then the marginals and inside evidence added.
// Returns [C,Q] logits (candidate-major, the reference's internal order; the
// transpose happens at the gather sites).
func (h *g2Head) scorePool(m *g2Marginals, bs g2BoundaryStates, text []float32, L int, queries []float32, Q int, pool g2Pool) []float32 {
	C := len(pool.Indices)
	pd := h.cfg.PairDim
	N := m.N

	startAll := h.scStartProj.apply(bs.States, N) // [N, pair]
	endAll := h.scEndProj.apply(bs.States, N)

	candidate := make([]float32, C*pd)
	for c := range C {
		s, e := pool.Indices[c][0], pool.Indices[c][1]
		copy(candidate[c*pd:(c+1)*pd], startAll[s*pd:(s+1)*pd])
		for i := range pd {
			candidate[c*pd+i] += endAll[e*pd+i]
		}
		// Length features [log1p(len), len/text_len, 1/sqrt(len)].
		length := float64(e - s)
		if length < 1 {
			length = 1
		}
		tl := float64(L)
		if tl < 1 {
			tl = 1
		}
		feats := []float32{
			float32(math.Log1p(length)),
			float32(length / tl),
			float32(1 / math.Sqrt(length)),
		}
		lr := h.lengthProj.apply(feats, 1)
		for i := range pd {
			candidate[c*pd+i] += lr[i]
		}
		pr := h.priorProj.apply([]float32{pool.Compat[c]}, 1)
		for i := range pd {
			candidate[c*pd+i] += pr[i]
		}
	}

	// Span content: prefix-sum mean pooling of the value-projected text states.
	if h.cfg.EnableSpanContent {
		cd := h.cfg.ContentDim
		values := h.contentValProj.apply(text, L) // [L, content]
		prefix := make([]float32, (L+1)*cd)
		for l := range L {
			for j := range cd {
				prefix[(l+1)*cd+j] = prefix[l*cd+j] + values[l*cd+j]
			}
		}
		for c := range C {
			s, e := pool.Indices[c][0], pool.Indices[c][1]
			length := e - s
			if length < 1 {
				length = 1
			}
			pooled := make([]float32, cd)
			for j := range cd {
				pooled[j] = (prefix[e*cd+j] - prefix[s*cd+j]) / float32(length)
			}
			h.contentLN.apply(pooled, 1, cd)
			pc := h.contentProj.apply(pooled, 1)
			for i := range pd {
				candidate[c*pd+i] += pc[i]
			}
		}
	}

	h.candNorm.apply(candidate, C, pd)
	for c := range C {
		if !pool.Valid[c] {
			for i := range pd {
				candidate[c*pd+i] = 0
			}
		}
	}

	query := h.queryProj.apply(queries, Q) // [Q, pair]

	// score[c,q] = candidate[c]·query[q]/√pair  (candidate-major [C,Q]).
	score := make([]float32, C*Q)
	scale := 1 / math.Sqrt(float64(pd))
	for c := range C {
		for q := range Q {
			var s float64
			for i := range pd {
				s += float64(candidate[c*pd+i]) * float64(query[q*pd+i])
			}
			score[c*Q+q] = float32(s * scale)
		}
	}

	// FiLM: per-query scale/shift of each candidate, then the small MLP head.
	gammaBeta := h.film.apply(query, Q) // [Q, 2*pair]
	cond := make([]float32, pd)
	for c := range C {
		for q := range Q {
			gb := gammaBeta[q*2*pd : (q+1)*2*pd]
			for i := range pd {
				cond[i] = candidate[c*pd+i]*(1+gb[i]) + gb[pd+i]
			}
			f := h.filmOut0.apply(cond, 1)
			for i := range f {
				f[i] = gelu(f[i])
			}
			score[c*Q+q] += h.filmOut3.apply(f, 1)[0]
		}
	}

	// Marginals + inside evidence, gathered per candidate. The reference's
	// clamp(max=N-1) on gather indices never engages for a valid pool.
	for c := range C {
		s, e := pool.Indices[c][0], pool.Indices[c][1]
		for q := range Q {
			score[c*Q+q] += m.Start[q*N+s] + m.End[q*N+e]
			if h.cfg.UseInsideEvidence {
				interval := m.Prefix[q*N+e] - m.Prefix[q*N+s] + m.Mean[q]*float32(e-s)
				length := e - s
				if length < 1 {
					length = 1
				}
				score[c*Q+q] += interval / float32(math.Sqrt(float64(length)))
			}
		}
	}

	// Mask invalid candidates (query masks are all-true for B=1 with real labels).
	for c := range C {
		if !pool.Valid[c] {
			for q := range Q {
				score[c*Q+q] = g2MaskLogit
			}
		}
	}
	return score
}

// stableTopKDesc returns the indices of the k largest values, value-descending
// with ties broken by original index (torch stable argsort semantics).
func stableTopKDesc(vals []float32, k int) []int {
	idx := make([]int, len(vals))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return vals[idx[a]] > vals[idx[b]] })
	if k < len(idx) {
		idx = idx[:k]
	}
	return idx
}

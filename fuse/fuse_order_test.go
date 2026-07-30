package fuse

import (
	"math/rand"
	"sort"
	"testing"
)

// A5's differential gate. The accumulation was restructured — one map instead of
// two, first-appearance order carried by the slice's own positions instead of by
// a map consulted from the comparator — and the claim is that the output is
// UNCHANGED: same keys, same scores, same order.
//
// The tie-break is the whole risk. Ties in RRF are not exotic: two lists that
// rank the same pair of items in swapped positions produce exactly equal fused
// scores, which is why first-appearance order exists at all. A reference
// implementation of the previous algorithm is the only honest way to check it,
// so both are written out below.

// rrfReference is the pre-A5 RRFWeighted, character for character in its
// structure: two maps, the second consulted from a reflect-based stable sort.
func rrfReference[K comparable](k float64, weights []float64, rankings ...[]K) []Result[K] {
	scores := make(map[K]float64)
	firstSeen := make(map[K]int)
	order := 0
	for r, ranking := range rankings {
		w := 1.0
		if weights != nil {
			w = weights[r]
		}
		for rank0, key := range ranking {
			if _, ok := firstSeen[key]; !ok {
				firstSeen[key] = order
				order++
			}
			scores[key] += w / (k + float64(rank0+1))
		}
	}
	out := make([]Result[K], 0, len(scores))
	for key, s := range scores {
		out = append(out, Result[K]{Key: key, Score: s})
	}
	sort.SliceStable(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return firstSeen[out[a].Key] < firstSeen[out[b].Key]
	})
	return out
}

// rsfReference is the pre-A5 RSFWeighted.
func rsfReference[K comparable](weights []float64, rankings ...[]Scored[K]) []Result[K] {
	scores := make(map[K]float64)
	firstSeen := make(map[K]int)
	order := 0
	for r, ranking := range rankings {
		if len(ranking) == 0 {
			continue
		}
		w := 1.0
		if weights != nil {
			w = weights[r]
		}
		lo, hi := ranking[0].Score, ranking[0].Score
		for _, s := range ranking {
			lo = min(lo, s.Score)
			hi = max(hi, s.Score)
		}
		span := hi - lo
		for _, s := range ranking {
			if _, ok := firstSeen[s.Key]; !ok {
				firstSeen[s.Key] = order
				order++
			}
			norm := 1.0
			if span > 0 {
				norm = (s.Score - lo) / span
			}
			scores[s.Key] += w * norm
		}
	}
	out := make([]Result[K], 0, len(scores))
	for key, s := range scores {
		out = append(out, Result[K]{Key: key, Score: s})
	}
	sort.SliceStable(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return firstSeen[out[a].Key] < firstSeen[out[b].Key]
	})
	return out
}

// tieHeavyRankings deliberately manufactures equal fused scores: with two lists
// over the same small key set, any pair of keys whose ranks are swapped between
// the lists fuses to the identical score. The narrower the key space, the more
// ties, so `keys` is kept small on purpose.
func tieHeavyRankings(rng *rand.Rand, lists, n, keys int) [][]int {
	out := make([][]int, lists)
	for l := range out {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = rng.Intn(keys)
		}
		out[l] = ids
	}
	return out
}

func TestRRF_matchesPreA5(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	ties := 0
	for range 400 {
		lists := 1 + rng.Intn(4)
		n := rng.Intn(40)
		rs := tieHeavyRankings(rng, lists, n, 1+rng.Intn(12))
		var weights []float64
		if rng.Intn(2) == 0 {
			weights = make([]float64, lists)
			for i := range weights {
				weights[i] = float64(rng.Intn(4)) // includes 0, which must still order
			}
		}
		k := float64(1 + rng.Intn(80))

		got := RRFWeighted(k, weights, rs...)
		want := rrfReference(k, weights, rs...)
		if len(got) != len(want) {
			t.Fatalf("k=%v weights=%v: %d results, reference %d", k, weights, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("k=%v weights=%v result %d: %+v, reference %+v", k, weights, i, got[i], want[i])
			}
			if i > 0 && want[i].Score == want[i-1].Score {
				ties++
			}
		}
	}
	if ties == 0 {
		t.Fatal("fixture produced no tied scores — the tie-break path is untested")
	}
	t.Logf("%d tied adjacent pairs exercised", ties)
}

func TestRSF_matchesPreA5(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	ties := 0
	for range 400 {
		lists := 1 + rng.Intn(4)
		rs := make([][]Scored[int], lists)
		for l := range rs {
			n := rng.Intn(30)
			rs[l] = make([]Scored[int], n)
			for i := range rs[l] {
				// A small score alphabet makes equal spans and equal
				// normalized values common, which is where ties come from.
				rs[l][i] = Scored[int]{Key: rng.Intn(10), Score: float64(rng.Intn(4))}
			}
		}
		var weights []float64
		if rng.Intn(2) == 0 {
			weights = make([]float64, lists)
			for i := range weights {
				weights[i] = float64(rng.Intn(3))
			}
		}
		got := RSFWeighted(weights, rs...)
		want := rsfReference(weights, rs...)
		if len(got) != len(want) {
			t.Fatalf("weights=%v: %d results, reference %d", weights, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("weights=%v result %d: %+v, reference %+v", weights, i, got[i], want[i])
			}
			if i > 0 && want[i].Score == want[i-1].Score {
				ties++
			}
		}
	}
	if ties == 0 {
		t.Fatal("fixture produced no tied scores — the tie-break path is untested")
	}
	t.Logf("%d tied adjacent pairs exercised", ties)
}

// TestRRF_tieBreakSurvivesAnUnstableSort is the gate the randomized test above
// could not be: swapping SortStableFunc for SortFunc passed it, because its key
// spaces were under a dozen and Go's pdqsort uses insertion sort — which is
// stable — below n=12. A stability requirement tested only on inputs that never
// reach the unstable path is not tested at all.
//
// Both cases here are large enough to reach pdqsort proper AND tied enough that
// order is decided entirely by the tie-break:
//
//   - all weights zero: every fused score is 0, so the output must be exactly
//     first-appearance order;
//   - two rankings that are reverses of each other: rank i in one and n-1-i in
//     the other, so keys i and n-1-i fuse to identical scores, giving n/2 tied
//     pairs spread across the whole list.
func TestRRF_tieBreakSurvivesAnUnstableSort(t *testing.T) {
	const n = 64 // comfortably past pdqsort's insertion-sort cutoff

	t.Run("all scores tied", func(t *testing.T) {
		keys := make([]int, n)
		for i := range keys {
			keys[i] = i * 7 % n // a first-appearance order that is not sorted order
		}
		got := RRFWeighted(DefaultK, []float64{0, 0}, keys, keys)
		if len(got) != n {
			t.Fatalf("got %d results, want %d", len(got), n)
		}
		seen := map[int]bool{}
		want := make([]int, 0, n)
		for _, k := range keys {
			if !seen[k] {
				seen[k] = true
				want = append(want, k)
			}
		}
		for i := range want {
			if got[i].Score != 0 {
				t.Fatalf("result %d has score %v; the fixture must produce all-zero scores", i, got[i].Score)
			}
			if got[i].Key != want[i] {
				t.Fatalf("result %d: key %d, want %d (first-appearance order)", i, got[i].Key, want[i])
			}
		}
	})

	t.Run("reversed rankings", func(t *testing.T) {
		fwd := make([]int, n)
		rev := make([]int, n)
		for i := range fwd {
			fwd[i] = i
			rev[i] = n - 1 - i
		}
		got := RRF(DefaultK, fwd, rev)
		want := rrfReference(DefaultK, nil, fwd, rev)
		tied := 0
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("result %d: %+v, reference %+v", i, got[i], want[i])
			}
			if i > 0 && want[i].Score == want[i-1].Score {
				tied++
			}
		}
		if tied < n/4 {
			t.Fatalf("only %d tied adjacent pairs; the fixture is not tie-heavy enough to test stability", tied)
		}
		t.Logf("%d tied adjacent pairs over %d results", tied, len(want))
	})
}

// TestRSF_tieBreakSurvivesAnUnstableSort is the RSF twin, and it took two
// attempts. An ALL-tied fixture does not test stability: pdqsort's
// already-ordered fast path leaves such an input untouched, so the non-stable
// mutant passed. What is needed is PARTIAL ties spread through a range of
// distinct scores, which forces real partitioning.
//
// Both lists score key i by i/2, so keys 2j and 2j+1 normalize identically and
// fuse to the same value while different j do not — 32 tied pairs among 32
// distinct score levels.
func TestRSF_tieBreakSurvivesAnUnstableSort(t *testing.T) {
	const n = 64
	a := make([]Scored[int], n)
	b := make([]Scored[int], n)
	for i := range a {
		a[i] = Scored[int]{Key: i, Score: float64(i / 2)}
		b[i] = Scored[int]{Key: n - 1 - i, Score: float64((n - 1 - i) / 2)}
	}
	got := RSF(a, b)
	want := rsfReference(nil, a, b)
	if len(got) != len(want) {
		t.Fatalf("got %d results, reference %d", len(got), len(want))
	}
	tied := 0
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("result %d: %+v, reference %+v", i, got[i], want[i])
		}
		if i > 0 && want[i].Score == want[i-1].Score {
			tied++
		}
	}
	if tied < n/4 {
		t.Fatalf("only %d tied adjacent pairs; the fixture is not tie-heavy enough to test stability", tied)
	}
	if want[0].Score == want[len(want)-1].Score {
		t.Fatal("every score is tied; pdqsort's ordered-input path skips the sort and stability goes untested")
	}
	t.Logf("%d tied adjacent pairs over %d results, %v..%v", tied, len(want), want[0].Score, want[len(want)-1].Score)
}

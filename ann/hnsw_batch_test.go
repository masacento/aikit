package ann

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// simPristine is the pre-item-15 per-neighbour score: one linalg.Dot per
// candidate. Kept as the oracle.
func (h *HNSW) simPristine(qv queryVec, id int) float64 {
	if h.int8 {
		return float64(linalg.DotI8(qv.q8, h.code(id))) * qv.scale * float64(h.scales[id])
	}
	return float64(linalg.Dot(qv.f32, h.vecs[id]))
}

// TestHNSW_batchedScoringMatchesPristine is the gate for item 15. Batching
// through Dot8x4 is NOT bit-identical — the 8-row kernel accumulates in a
// different order, so a score can move by ~1 float32 ULP. HNSW's traversal is
// threshold-driven (results.items[0].sim gates both exploration and collection),
// so a moved score could in principle change which nodes are visited and
// therefore what is returned.
//
// This checks it does not, across dims, corpus sizes and ef values: the batched
// scorer must agree with the per-candidate one to within a couple of ULP on
// every gathered group, including groups that are not multiples of 8.
func TestHNSW_batchedScoringMatchesPristine(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	for _, d := range []int{64, 65, 256, 768} {
		for _, n := range []int{9, 500, 3000} {
			vecs := make([][]float32, n)
			for i := range vecs {
				v := make([]float32, d)
				var norm float64
				for j := range v {
					v[j] = float32(rng.NormFloat64())
					norm += float64(v[j]) * float64(v[j])
				}
				inv := float32(1 / math.Sqrt(norm))
				for j := range v {
					v[j] *= inv
				}
				vecs[i] = v
			}
			h := &HNSW{vecs: vecs, dim: d}
			qv := queryVec{f32: vecs[0]}

			// Group sizes that straddle the 8-wide kernel in both directions.
			for _, gsz := range []int{0, 1, 7, 8, 9, 16, 17, 31, n} {
				if gsz > n {
					continue
				}
				ids := make([]int, gsz)
				for i := range ids {
					ids[i] = rng.Intn(n)
				}
				got := h.scoreInto(qv, ids, nil)
				if len(got) != len(ids) {
					t.Fatalf("d=%d n=%d gsz=%d: got %d scores, want %d", d, n, gsz, len(got), len(ids))
				}
				const ulp = 1.0 / (1 << 23)
				for i, id := range ids {
					want := h.simPristine(qv, id)
					if diff := math.Abs(got[i] - want); diff > 4*ulp {
						t.Fatalf("d=%d n=%d gsz=%d idx=%d id=%d: batched %v, per-candidate %v (Δ=%g, %.1f ULP)",
							d, n, gsz, i, id, got[i], want, diff, diff/ulp)
					}
				}
			}
		}
	}
}

// TestHNSW_batchedQueryMatchesPristine is the real gate for item 15's claim
// that the two-phase rewrite is ORDER-PRESERVING. Batching moves scores by ~1
// ULP and, more importantly, restructures the loop that feeds the evolving
// results.items[0].sim threshold — so the only convincing check is the full
// ranked result list against the pristine scorer on the SAME graph.
//
// Checking only the top hit is not enough: an earlier version of this test did
// that and passed a mutant that reversed the push order entirely.
func TestHNSW_batchedQueryMatchesPristine(t *testing.T) {
	rng := rand.New(rand.NewSource(150))
	var totalHits, idxDiffs int
	var worstDelta float64
	for _, d := range []int{64, 256, 768} {
		for _, n := range []int{500, 5000} {
			vecs := make([][]float32, n)
			for i := range vecs {
				v := make([]float32, d)
				var norm float64
				for j := range v {
					v[j] = float32(rng.NormFloat64())
					norm += float64(v[j]) * float64(v[j])
				}
				inv := float32(1 / math.Sqrt(norm))
				for j := range v {
					v[j] *= inv
				}
				vecs[i] = v
			}
			// Build ONCE, with the pristine scorer, so both query paths traverse
			// an identical graph and the only variable is the scoring loop.
			h := BuildHNSW(vecs, Config{})
			h.scoreUnbatched = true

			for _, ef := range []int{16, 64, 200} {
				for range 25 {
					q := vecs[rng.Intn(n)]

					h.scoreUnbatched = true
					want := h.QueryEf(q, 10, ef)
					h.scoreUnbatched = false
					got := h.QueryEf(q, 10, ef)

					if len(got) != len(want) {
						t.Fatalf("d=%d n=%d ef=%d: %d hits batched, %d pristine", d, n, ef, len(got), len(want))
					}
					for i := range want {
						totalHits++
						if got[i].Index != want[i].Index {
							idxDiffs++
						}
						if delta := math.Abs(got[i].Score - want[i].Score); delta > worstDelta {
							worstDelta = delta
						}
					}
				}
			}
		}
	}
	if totalHits == 0 {
		t.Fatal("no hits compared — this test proves nothing")
	}
	if idxDiffs != 0 {
		t.Errorf("%d of %d ranked hits differ between the batched and pristine scorers — "+
			"the two-phase rewrite is supposed to preserve push order", idxDiffs, totalHits)
	}
	const ulp = 1.0 / (1 << 23)
	if worstDelta > 4*ulp {
		t.Errorf("worst |Δscore| %g (%.1f ULP) exceeds the ~1 ULP the 8-row kernel should cost",
			worstDelta, worstDelta/ulp)
	}
	t.Logf("%d ranked hits, %d index differences, max |Δscore| %g (%.2f ULP)",
		totalHits, idxDiffs, worstDelta, worstDelta/ulp)
}

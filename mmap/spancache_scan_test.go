package mmap

import "testing"

// TestSpanCache_cyclicScanHitRate is the gate for perf-campaign item 9, and the
// measurement that justified changing the eviction policy.
//
// The only demand signal in the kit is a sequential scan: FlatI8's paged query
// walks blocks 0,1,2,… on every call. That is the textbook cyclic-scan
// pathology for LRU — the member evicted to make room is precisely the one
// needed next time round — and it is far worse than "some misses". Measured
// before the change, over ten passes of a 64-block working set:
//
//	budget  8/64 blocks   hit rate 0.0%
//	budget 32/64          hit rate 0.0%
//	budget 63/64          hit rate 0.0%   ← 98% of the data resident, still 0%
//	budget 64/64          hit rate 90.0%
//
// A cache holding 98% of the working set that hits 0% of the time is not a
// tuning problem, it is the wrong policy. Evicting the most-recently-touched
// member instead pins a stable prefix, so the hit rate tracks
// budget/working-set — the best any policy can do without knowing the loop
// length.
//
// The floors below are set at ~85% of that ceiling. They exist to catch a
// reversion to LRU, which would drive every one of them to zero.
func TestSpanCache_cyclicScanHitRate(t *testing.T) {
	const nBlocks, blockBytes = 64, 1 << 16
	for _, tc := range []struct {
		budgetBlocks int
		minHitRate   float64
	}{
		{8, 0.09},  // ceiling 12.5%
		{32, 0.42}, // ceiling 50%
		{63, 0.80}, // ceiling ~98%
	} {
		fa := &fakeAdvise{}
		c := NewSpanCache[int](int64(tc.budgetBlocks) * blockBytes)
		c.advise = fa.advise
		for b := range nBlocks {
			c.Add(b, [][]byte{span(blockBytes)})
		}
		for range 10 { // ten full passes, as a paged query loop does
			for b := range nBlocks {
				c.Touch(b)
			}
		}
		h, m, e := c.Stats()
		rate := float64(h) / float64(h+m)
		t.Logf("budget=%2d/%d blocks: hits=%4d misses=%4d evictions=%4d -> hit rate %.1f%%",
			tc.budgetBlocks, nBlocks, h, m, e, 100*rate)
		if rate < tc.minHitRate {
			t.Errorf("budget=%d/%d: hit rate %.1f%% below the %.0f%% floor — has eviction "+
				"reverted to LRU? A cyclic scan under LRU hits 0%%.",
				tc.budgetBlocks, nBlocks, 100*rate, 100*tc.minHitRate)
		}
		if c.Resident() > c.Budget() {
			t.Errorf("budget=%d: resident %d exceeds budget %d", tc.budgetBlocks, c.Resident(), c.Budget())
		}
	}
}

// TestSpanCache_randomAccessStillCaches checks the policy change did not make
// non-scan access pathological in the other direction: with the working set
// inside the budget nothing is ever evicted, so every repeat is a hit whatever
// the eviction order is.
func TestSpanCache_randomAccessStillCaches(t *testing.T) {
	fa := &fakeAdvise{}
	const n = 16
	c := NewSpanCache[int](int64(n) * 100)
	c.advise = fa.advise
	for i := range n {
		c.Add(i, [][]byte{span(100)})
	}
	for range 5 {
		for i := range n {
			c.Touch(i)
		}
	}
	h, m, e := c.Stats()
	if e != 0 {
		t.Errorf("evictions = %d, want 0 — the working set fits the budget", e)
	}
	if want := int64(n * 4); h != want {
		t.Errorf("hits = %d, want %d (first pass misses, four repeat passes hit)", h, want)
	}
	if m != int64(n) {
		t.Errorf("misses = %d, want %d (one cold miss per member)", m, n)
	}
}

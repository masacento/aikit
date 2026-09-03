package linalg

import "testing"

// TestSpanRows_emptyRowRangeVisitsNoColumns pins the guard that w8a8Span's
// leftover-strip call depends on: an empty ROW range must visit no columns at
// all, not walk the column range doing per-column setup for zero rows.
//
// This is a performance property, so the interesting question is how to gate it
// WITHOUT a timing assertion. The answer used here is to make the walk itself
// observable: the weight slice passed in is deliberately too short for the column
// range, so a loop that constructs bQ[j*K:j*K+K] panics on the first column and a
// function that returns early cannot. No clock, no flake.
//
// The history is why this exists. When the S-01b tile declined (M<4 — which is
// every decode call, the most-executed path in the system), w8a8Span still made
// its leftover-column call with a zero-height row range, and that call walked all
// N columns building slices and loading scales for no rows. Every correctness
// test passed, because the results were right. tools/perfgate caught it as an
// 8.00-15.53% regression on the M=1 shapes against v1.31.0 — which is exactly the
// class of defect that gate exists for, and exactly the class this file's own A/B
// harnesses could not see, since they only ever measured M>=4 where the tile
// engages.
func TestSpanRows_emptyRowRangeVisitsNoColumns(t *testing.T) {
	const K, N = 64, 128
	// One column's worth of weights against a 128-column range: any column walk
	// runs off the end.
	short8 := make([]int8, K)
	shortNibbles := make([]byte, K/2)
	scales := make([]float32, N)
	dst := make([]float32, N)
	aq := make([]int8, K)
	aScales := []float32{1}

	t.Run("w8a8", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("w8a8SpanRows walked the column range for an empty row range: %v", r)
			}
		}()
		w8a8SpanRows(aq, aScales, short8, scales, dst, K, N, 0, 0, 0, N)
	})

	t.Run("w4a8", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("w4a8SpanRows walked the column range for an empty row range: %v", r)
			}
		}()
		nGroups, bpr := groupsFor(K, 32)
		w4a8SpanRows(aq, aScales, shortNibbles, scales, dst, K, N, 32, nGroups, bpr, 0, 0, 0, N)
	})

	// The guard must not have turned a real row range into a no-op: the same call
	// with one row must still write every column.
	t.Run("nonEmptyStillWorks", func(t *testing.T) {
		full := make([]int8, K*N)
		for i := range full {
			full[i] = 1
		}
		for i := range aq {
			aq[i] = 2
		}
		for i := range scales {
			scales[i] = 0.5
		}
		for i := range dst {
			dst[i] = -1
		}
		w8a8SpanRows(aq, aScales, full, scales, dst, K, N, 0, 1, 0, N)
		for j := range N {
			if dst[j] == -1 {
				t.Fatalf("column %d was not written — the empty-range guard is too eager", j)
			}
		}
	})
}

package encoder

import (
	"slices"
	"testing"
)

// TestLengthSortedOrder covers the dispatch-order helper extracted from
// EncodeBatch (docs/internal/task-code-health.md §4.4). It needs no checkpoint,
// which is the point: every EncodeBatch test that would otherwise exercise this
// is gated on testdata/encoder-model and skips on a machine without it, so the
// extraction landed with no running coverage at all until this file.
func TestLengthSortedOrder(t *testing.T) {
	t.Run("ascending by length", func(t *testing.T) {
		lens := []int{40, 10, 30, 20}
		got := lengthSortedOrder(lens, len(lens))
		if want := []int{1, 3, 2, 0}; !slices.Equal(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	})

	// Stability is load-bearing, not incidental: EncodeBatch scatters results
	// back by index, and callers rely on equal-length inputs keeping their
	// original relative order for run-to-run determinism.
	t.Run("stable across equal lengths", func(t *testing.T) {
		lens := []int{5, 5, 5, 5, 5}
		got := lengthSortedOrder(lens, len(lens))
		if want := []int{0, 1, 2, 3, 4}; !slices.Equal(got, want) {
			t.Errorf("equal lengths must keep caller order: got %v, want %v", got, want)
		}
	})

	t.Run("is a permutation of 0..n-1", func(t *testing.T) {
		lens := []int{7, 2, 9, 2, 7, 1, 9, 3}
		got := lengthSortedOrder(lens, len(lens))
		seen := make([]bool, len(lens))
		for _, ix := range got {
			if ix < 0 || ix >= len(lens) || seen[ix] {
				t.Fatalf("not a permutation: %v", got)
			}
			seen[ix] = true
		}
		for i := 1; i < len(got); i++ {
			if lens[got[i-1]] > lens[got[i]] {
				t.Errorf("not ascending at %d: %v over lens %v", i, got, lens)
			}
		}
	})

	t.Run("empty", func(t *testing.T) {
		if got := lengthSortedOrder(nil, 0); len(got) != 0 {
			t.Errorf("got %v, want empty", got)
		}
	})
}

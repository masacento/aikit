package bm25

import (
	"reflect"
	"testing"
)

func batchTestIndex() *Index {
	return Build([][]string{
		{"the", "quick", "brown", "fox"},
		{"the", "lazy", "dog"},
		{"quick", "fox", "jumps"},
		{"brown", "dog", "runs"},
		{"the", "fox", "and", "the", "dog"},
	})
}

func TestTopKBatch_matchesSerialTopK(t *testing.T) {
	ix := batchTestIndex()
	queries := [][]string{
		{"quick", "fox"},
		{"the", "dog"},
		{"brown"},
		{"nonexistent"},
		{"jumps", "runs"},
	}
	got := ix.TopKBatch(queries, 3, 0)
	if len(got) != len(queries) {
		t.Fatalf("TopKBatch returned %d results, want %d", len(got), len(queries))
	}
	for i, q := range queries {
		want := ix.TopK(q, 3)
		if !reflect.DeepEqual(got[i], want) {
			t.Errorf("query %d (%v): TopKBatch = %v, want %v (serial TopK)", i, q, got[i], want)
		}
	}
}

func TestTopKBatch_orderIsCallerOrderNotCompletionOrder(t *testing.T) {
	ix := batchTestIndex()
	// More queries than likely goroutines, several repeats, so a race in the
	// output-slot indexing (writing to the wrong i) would show up as a
	// mismatch against the serial reference at some position.
	queries := make([][]string, 50)
	for i := range queries {
		switch i % 4 {
		case 0:
			queries[i] = []string{"quick", "fox"}
		case 1:
			queries[i] = []string{"the", "dog"}
		case 2:
			queries[i] = []string{"brown"}
		case 3:
			queries[i] = nil
		}
	}
	got := ix.TopKBatch(queries, 2, 0)
	for i, q := range queries {
		want := ix.TopK(q, 2)
		if !reflect.DeepEqual(got[i], want) {
			t.Fatalf("query %d: TopKBatch = %v, want %v", i, got[i], want)
		}
	}
}

func TestTopKBatch_concurrencyDoesNotChangeResult(t *testing.T) {
	ix := batchTestIndex()
	queries := [][]string{{"quick", "fox"}, {"the", "dog"}, {"brown"}, {"jumps"}}
	base := ix.TopKBatch(queries, 3, 0)
	for _, c := range []int{1, 2, 100} {
		got := ix.TopKBatch(queries, 3, c)
		if !reflect.DeepEqual(got, base) {
			t.Errorf("concurrency=%d: TopKBatch = %v, want %v (concurrency=0)", c, got, base)
		}
	}
}

func TestTopKBatch_empty(t *testing.T) {
	ix := batchTestIndex()
	if got := ix.TopKBatch(nil, 3, 0); got != nil {
		t.Errorf("TopKBatch(no queries) = %v, want nil", got)
	}
}

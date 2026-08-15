package sparse

import (
	"reflect"
	"testing"
)

func batchTestIndex() *Index {
	return New([]SparseVec{
		{Terms: []uint32{1, 2, 3}, Weights: []float32{1, 0.5, 0.2}},
		{Terms: []uint32{2, 4}, Weights: []float32{0.8, 0.3}},
		{Terms: []uint32{1, 5}, Weights: []float32{0.6, 0.9}},
		{Terms: []uint32{3, 4, 5}, Weights: []float32{0.4, 0.4, 0.4}},
		{Terms: []uint32{1, 2, 3, 4, 5}, Weights: []float32{0.1, 0.1, 0.1, 0.1, 0.1}},
	})
}

func TestQueryBatch_matchesSerialQuery(t *testing.T) {
	ix := batchTestIndex()
	queries := []SparseVec{
		{Terms: []uint32{1, 2}, Weights: []float32{1, 1}},
		{Terms: []uint32{4}, Weights: []float32{1}},
		{Terms: []uint32{5, 3}, Weights: []float32{0.7, 0.3}},
		{Terms: []uint32{99}, Weights: []float32{1}}, // unknown term
		{},
	}
	got := ix.QueryBatch(queries, 3, 0)
	if len(got) != len(queries) {
		t.Fatalf("QueryBatch returned %d results, want %d", len(got), len(queries))
	}
	for i, q := range queries {
		want := ix.Query(q, 3)
		if !reflect.DeepEqual(got[i], want) {
			t.Errorf("query %d (%v): QueryBatch = %v, want %v (serial Query)", i, q, got[i], want)
		}
	}
}

func TestQueryBatch_orderIsCallerOrderNotCompletionOrder(t *testing.T) {
	ix := batchTestIndex()
	base := []SparseVec{
		{Terms: []uint32{1, 2}, Weights: []float32{1, 1}},
		{Terms: []uint32{4}, Weights: []float32{1}},
		{Terms: []uint32{5, 3}, Weights: []float32{0.7, 0.3}},
		{},
	}
	queries := make([]SparseVec, 50)
	for i := range queries {
		queries[i] = base[i%len(base)]
	}
	got := ix.QueryBatch(queries, 2, 0)
	for i, q := range queries {
		want := ix.Query(q, 2)
		if !reflect.DeepEqual(got[i], want) {
			t.Fatalf("query %d: QueryBatch = %v, want %v", i, got[i], want)
		}
	}
}

func TestQueryBatch_concurrencyDoesNotChangeResult(t *testing.T) {
	ix := batchTestIndex()
	queries := []SparseVec{
		{Terms: []uint32{1, 2}, Weights: []float32{1, 1}},
		{Terms: []uint32{4}, Weights: []float32{1}},
		{Terms: []uint32{5, 3}, Weights: []float32{0.7, 0.3}},
	}
	base := ix.QueryBatch(queries, 3, 0)
	for _, c := range []int{1, 2, 100} {
		got := ix.QueryBatch(queries, 3, c)
		if !reflect.DeepEqual(got, base) {
			t.Errorf("concurrency=%d: QueryBatch = %v, want %v (concurrency=0)", c, got, base)
		}
	}
}

func TestQueryBatch_empty(t *testing.T) {
	ix := batchTestIndex()
	if got := ix.QueryBatch(nil, 3, 0); got != nil {
		t.Errorf("QueryBatch(no queries) = %v, want nil", got)
	}
}

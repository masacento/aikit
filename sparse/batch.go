package sparse

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// QueryBatch runs Query for many queries concurrently, returning each query's
// result in the CALLER'S order (not completion order). concurrency <= 0 means
// NumCPU.
//
// Query's own scratch (scoreQuery's accum.Accum) is sync.Pool-backed per call,
// so concurrent Query calls over the same Index were already safe; this just
// amortizes the goroutine dispatch (the same work-stealing shape
// encoder.CrossEncoder.ScoreBatch and late.ScoreBatch use for their own
// per-item work) across many queries, instead of every bulk caller — an eval
// harness scoring thousands of queries, say — hand-rolling the same loop.
// ann.FlatI8.QueryBatch and bm25.Index.TopKBatch are this package's
// counterparts on the dense and lexical sides.
func (ix *Index) QueryBatch(queries []SparseVec, k int, concurrency int) [][]Hit {
	if len(queries) == 0 {
		return nil
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	concurrency = min(concurrency, len(queries))

	out := make([][]Hit, len(queries))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			for {
				i := int(next.Add(1)) - 1
				if i >= len(queries) {
					return
				}
				out[i] = ix.Query(queries[i], k)
			}
		})
	}
	wg.Wait()
	return out
}

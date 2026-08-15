package bm25

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// TopKBatch runs TopK for many queries concurrently, returning each query's
// result in the CALLER'S order (not completion order). concurrency <= 0 means
// NumCPU.
//
// TopK's own scratch — scoreQuery's accum.Accum, topKWAND's wandState — is
// sync.Pool-backed per call, so concurrent TopK calls over the same Index were
// already safe; this just amortizes the goroutine dispatch (the same
// work-stealing shape encoder.CrossEncoder.ScoreBatch and late.ScoreBatch use
// for their own per-item work) across many queries, instead of every bulk
// caller — an eval harness scoring thousands of queries, say — hand-rolling
// the same loop. ann.FlatI8.QueryBatch is this package's counterpart on the
// dense side.
func (ix *Index) TopKBatch(queries [][]string, k int, concurrency int) [][]Result {
	if len(queries) == 0 {
		return nil
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	concurrency = min(concurrency, len(queries))

	out := make([][]Result, len(queries))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			for {
				i := int(next.Add(1)) - 1
				if i >= len(queries) {
					return
				}
				out[i] = ix.TopK(queries[i], k)
			}
		})
	}
	wg.Wait()
	return out
}

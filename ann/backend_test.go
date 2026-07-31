package ann

import (
	"fmt"
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// fakeBatchIndex is an I8BatchIndex that scores on the CPU, so the device
// batched path can be exercised without a device.
//
// It exists because that path is otherwise UNREACHABLE in the default build —
// f.gpu is nil unless a gpu build registers a backend — which is why §4.7's
// per-call M×N allocation survived this long: no test and no benchmark ever ran
// the line. calls counts dispatches so a test can prove the path was taken
// rather than assume it.
type fakeBatchIndex struct {
	f     *FlatI8
	calls int
}

func (b *fakeBatchIndex) Score(q []float32, dst []float32) error {
	linalg.MatmulBTW8A8(q, b.f.bq, b.f.scales, dst, 1, b.f.dim, b.f.n)
	return nil
}
func (b *fakeBatchIndex) Close() error { return nil }
func (b *fakeBatchIndex) ScoreBatch(queries [][]float32, dst []float32) error {
	b.calls++
	for m, q := range queries {
		linalg.MatmulBTW8A8(q, b.f.bq, b.f.scales, dst[m*b.f.n:(m+1)*b.f.n], 1, b.f.dim, b.f.n)
	}
	return nil
}

// TestQueryBatch_deviceBatchedPathMatchesPerQuery gates §4.7's change: pooling
// the M×N score matrix must not alter a single hit, and the batched path must
// actually be the one running.
func TestQueryBatch_deviceBatchedPathMatchesPerQuery(t *testing.T) {
	const n, d, k = 2000, 64, 10
	vecs := unitVecs(n, d, 47)
	f := NewFlatI8(vecs)
	queries := unitVecs(24, d, 470)

	want := make([][]Hit, len(queries))
	for i, q := range queries {
		want[i] = f.Query(q, k)
	}

	fb := &fakeBatchIndex{f: f}
	f.gpu = fb
	defer func() { f.gpu = nil }()

	// Twice, so the second call reuses a pooled buffer the first one filled.
	for round := range 2 {
		got := f.QueryBatch(queries, k)
		if fb.calls != round+1 {
			t.Fatalf("round %d: ScoreBatch called %d times; the batched path did not run", round, fb.calls)
		}
		for i := range want {
			if len(got[i]) != len(want[i]) {
				t.Fatalf("round %d query %d: %d hits vs %d", round, i, len(got[i]), len(want[i]))
			}
			for j := range want[i] {
				if got[i][j] != want[i][j] {
					t.Fatalf("round %d query %d hit %d: %+v vs per-query %+v", round, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
}

// TestQueryBatch_pooledScratchIsInert poisons the pooled score matrix before a
// call. ScoreBatch assigns every element it is given, so stale contents must be
// invisible — and if that contract is ever broken, the symptom is a plausible
// wrong ranking rather than a crash.
func TestQueryBatch_pooledScratchIsInert(t *testing.T) {
	const n, d, k = 1500, 48, 10
	vecs := unitVecs(n, d, 48)
	f := NewFlatI8(vecs)
	queries := unitVecs(8, d, 480)
	fb := &fakeBatchIndex{f: f}
	f.gpu = fb
	defer func() { f.gpu = nil }()

	want := f.QueryBatch(queries, k)

	sc := batchScratchPool.Get().(*batchScratch)
	for i := range sc.dst {
		sc.dst[i] = float32(math.Inf(1))
	}
	batchScratchPool.Put(sc)

	got := f.QueryBatch(queries, k)
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("query %d hit %d after a poisoned pool: %+v, want %+v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

// BenchmarkQueryBatch_deviceBatched prices §4.7 through the fake backend.
//
// WHAT IT CAN AND CANNOT SHOW. The scoring here is the CPU kernel, not a device
// GEMM, so the TIME is not a device number and must not be read as one — a real
// backend's dispatch would dominate. What transfers is the allocation: the M×N
// host score matrix is the same size and lifetime whatever fills it, and that is
// the whole of the finding.
func BenchmarkQueryBatch_deviceBatched(b *testing.B) {
	const n, d, k = 50_000, 256, 10
	f := NewFlatI8(unitVecs(n, d, 407))
	fb := &fakeBatchIndex{f: f}
	f.gpu = fb
	defer func() { f.gpu = nil }()
	for _, m := range []int{8, 64, 256} {
		queries := unitVecs(m, d, int64(4070+m))
		b.Run(fmt.Sprintf("m%d", m), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkBatchHits = f.QueryBatch(queries, k)
			}
		})
	}
}

var sinkBatchHits [][]Hit

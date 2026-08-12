//go:build linux

package anncuda

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// randUnit builds n random L2-normalized dim-vectors (the FlatI8 invariant).
func randUnit(rng *rand.Rand, n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		var s float64
		for d := range v {
			v[d] = float32(rng.NormFloat64())
			s += float64(v[d]) * float64(v[d])
		}
		inv := float32(1 / math.Sqrt(s))
		for d := range v {
			v[d] *= inv
		}
		out[i] = v
	}
	return out
}

// TestCUDAGEMV_parityWithCPU is the Phase-1b gate — the CUDA twin of annmetal's
// TestMetalGEMV_parityWithCPU: an ann.FlatI8 scored on the GPU (EnableGPU) must
// return exactly the CPU FlatI8's top-k — same indices in the same order, scores
// within the int8/float tolerance — over many random queries. The int8 dot is
// exact integer arithmetic, so the rankings are identical, not merely close.
// Break-it-first below proves the equality check is not vacuous.
func TestCUDAGEMV_parityWithCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(1))
	const n, dim, k = 2000, 96, 10
	vecs := randUnit(rng, n, dim)

	cpu := ann.NewFlatI8(vecs)
	gpu := ann.NewFlatI8(vecs)
	if err := gpu.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer gpu.Close()
	if !gpu.GPUEnabled() {
		t.Fatal("GPUEnabled false after EnableGPU")
	}
	t.Logf("backend=%q n=%d dim=%d k=%d", ann.BackendName(), n, dim, k)

	queries := randUnit(rng, 200, dim)
	worstScoreΔ := 0.0
	for qi, q := range queries {
		want := cpu.Query(q, k)
		got := gpu.Query(q, k)
		if len(got) != len(want) {
			t.Fatalf("q%d: got %d hits, want %d", qi, len(got), len(want))
		}
		for i := range want {
			if got[i].Index != want[i].Index {
				t.Fatalf("q%d rank %d: GPU index %d != CPU index %d (top-k diverged)", qi, i, got[i].Index, want[i].Index)
			}
			if d := math.Abs(got[i].Score - want[i].Score); d > worstScoreΔ {
				worstScoreΔ = d
			}
		}
	}
	t.Logf("GPU≡CPU top-%d over %d queries: worst score Δ %.3e", k, len(queries), worstScoreΔ)
	if worstScoreΔ > 1e-3 {
		t.Errorf("worst score Δ %.3e too large — GPU rescale diverges from CPU", worstScoreΔ)
	}

	// Break-it-first: the equality gate must be able to FAIL. Score the SAME GPU
	// index but compare against a DIFFERENT index's top-k — it must diverge, proving
	// the per-query comparison is discriminating (not "any two rankings match").
	other := ann.NewFlatI8(randUnit(rng, n, dim))
	diverged := 0
	for _, q := range queries {
		g := gpu.Query(q, k)
		o := other.Query(q, k)
		for i := range g {
			if g[i].Index != o[i].Index {
				diverged++
				break
			}
		}
	}
	t.Logf("break-it-first: %d/%d queries' top-k differ from an unrelated index", diverged, len(queries))
	if diverged == 0 {
		t.Error("break-it-first vacuous: GPU top-k matched an unrelated index on every query")
	}
}

// Compile-time proof that the CUDA index is a batch index — so FlatI8.QueryBatch
// takes the batched-GEMM path (ScoreBatch), never the per-query fallback.
var _ ann.I8BatchIndex = (*cudaI8Index)(nil)

// TestCUDAGEMM_batchParityWithCPU is the batched gate (Phase 2's shape, on CUDA):
// scoring a batch of queries as one int8 GEMM on the GPU must return exactly the
// CPU FlatI8's per-query top-k for every query in the batch — same indices, scores
// within tolerance. The int8 dot is exact, so the rankings are identical.
func TestCUDAGEMM_batchParityWithCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(7))
	const n, dim, k, M = 3000, 128, 10, 16
	vecs := randUnit(rng, n, dim)

	cpu := ann.NewFlatI8(vecs)
	gpu := ann.NewFlatI8(vecs)
	if err := gpu.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer gpu.Close()

	queries := randUnit(rng, M, dim)
	batch := gpu.QueryBatch(queries, k)
	if len(batch) != M {
		t.Fatalf("QueryBatch returned %d rows, want %d", len(batch), M)
	}
	worst := 0.0
	for m, q := range queries {
		want := cpu.Query(q, k)
		got := batch[m]
		if len(got) != len(want) {
			t.Fatalf("q%d: %d hits, want %d", m, len(got), len(want))
		}
		for i := range want {
			if got[i].Index != want[i].Index {
				t.Fatalf("q%d rank %d: GPU-batch index %d != CPU index %d (GEMM diverged)", m, i, got[i].Index, want[i].Index)
			}
			if d := math.Abs(got[i].Score - want[i].Score); d > worst {
				worst = d
			}
		}
	}
	t.Logf("GPU-batch(M=%d) ≡ CPU per-query top-%d: worst score Δ %.3e", M, k, worst)
	if worst > 1e-3 {
		t.Errorf("worst score Δ %.3e too large", worst)
	}

	// Vacuity: distinct queries must give distinct top hits (the batch is not
	// collapsing every row to the same answer).
	distinct := false
	for m := 1; m < M; m++ {
		if len(batch[m]) > 0 && len(batch[0]) > 0 && batch[m][0].Index != batch[0][0].Index {
			distinct = true
			break
		}
	}
	if !distinct {
		t.Error("break-it-first vacuous: every batch row shared the same top hit")
	}
}

// TestCUDA_shapesOffBlockBoundary pushes shapes that are NOT multiples of the
// 256-thread block through both kernels — the CUDA-specific geometry the Metal
// path never sees, since dispatchThreads launches exactly n where a CUDA launch
// rounds up to whole blocks. It gates that no row is dropped or misindexed when
// the grid overhangs N (or M*N).
//
// It does NOT gate the kernels' bounds checks, and mutation testing is how we
// know: deleting `if (j >= *N) return;` from gemv_w8a8 leaves every test here
// PASSING. The overhanging threads write just past out[N], which lands in the
// allocation's alignment slack — silent corruption, not a wrong answer. The gate
// that does catch it is TestCUDA_tailBlockGuard in the device layer, which
// allocates a sentinel canary past the output and checks it survives (deleting
// the guard there fails 3 assertions). Guard changes belong to that test.
func TestCUDA_shapesOffBlockBoundary(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(11))
	const k = 5
	// n values straddling the 256-block boundary; dims that are not multiples of 4
	// (so the K loop's tail is exercised too).
	for _, shape := range []struct{ n, dim, m int }{
		{1, 7, 1},     // single row, single query
		{255, 33, 3},  // one short of a block
		{256, 64, 1},  // exactly one block
		{257, 17, 2},  // one past a block
		{1023, 31, 7}, // large, prime-ish
	} {
		vecs := randUnit(rng, shape.n, shape.dim)
		cpu := ann.NewFlatI8(vecs)
		gpu := ann.NewFlatI8(vecs)
		if err := gpu.EnableGPU(); err != nil {
			t.Fatalf("n=%d dim=%d EnableGPU: %v", shape.n, shape.dim, err)
		}
		queries := randUnit(rng, shape.m, shape.dim)

		// Single-query path.
		for qi, q := range queries {
			want, got := cpu.Query(q, k), gpu.Query(q, k)
			if len(got) != len(want) {
				t.Fatalf("n=%d dim=%d q%d: %d hits, want %d", shape.n, shape.dim, qi, len(got), len(want))
			}
			for i := range want {
				if got[i].Index != want[i].Index {
					t.Fatalf("n=%d dim=%d q%d rank %d: GEMV index %d != CPU %d", shape.n, shape.dim, qi, i, got[i].Index, want[i].Index)
				}
			}
		}
		// Batched path (M*N is the launch bound there).
		batch := gpu.QueryBatch(queries, k)
		for m, q := range queries {
			want := cpu.Query(q, k)
			if len(batch[m]) != len(want) {
				t.Fatalf("n=%d dim=%d M=%d q%d: %d hits, want %d", shape.n, shape.dim, shape.m, m, len(batch[m]), len(want))
			}
			for i := range want {
				if batch[m][i].Index != want[i].Index {
					t.Fatalf("n=%d dim=%d M=%d q%d rank %d: GEMM index %d != CPU %d", shape.n, shape.dim, shape.m, m, i, batch[m][i].Index, want[i].Index)
				}
			}
		}
		gpu.Close()
	}
	t.Log("GEMV+GEMM ≡ CPU across 5 shapes straddling the 256-thread block boundary")
}

// TestCUDA_concurrentScore runs many goroutines against one GPU index. A CUDA
// context is thread-affine, and a per-call dispatch from a migrating goroutine is
// the exact crash class that hit the Metal path through NSAutoreleasePool. Nothing
// here pins a thread because gocudrv's Context owns a LockOSThread'd executor and
// funnels every driver call through it (gpu/cuda.go, "Thread affinity") — this
// test is what holds that claim honest, and also covers the per-index mutex that
// guards the shared scratch buffers.
func TestCUDA_concurrentScore(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(23))
	const n, dim, k, workers, iters = 512, 48, 5, 8, 25
	vecs := randUnit(rng, n, dim)
	cpu := ann.NewFlatI8(vecs)
	gpu := ann.NewFlatI8(vecs)
	if err := gpu.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer gpu.Close()

	queries := randUnit(rng, workers, dim)
	want := make([][]ann.Hit, workers)
	for i, q := range queries {
		want[i] = cpu.Query(q, k)
	}

	errs := make(chan string, workers)
	done := make(chan struct{})
	for w := range workers {
		go func() {
			defer func() { done <- struct{}{} }()
			for range iters {
				got := gpu.Query(queries[w], k)
				if len(got) != len(want[w]) {
					errs <- "hit count changed under concurrency"
					return
				}
				for i := range want[w] {
					if got[i].Index != want[w][i].Index {
						errs <- "top-k diverged under concurrency"
						return
					}
				}
			}
		}()
	}
	for range workers {
		<-done
	}
	close(errs)
	for e := range errs {
		t.Error(e)
	}
	t.Logf("%d goroutines × %d queries: top-%d stable and CPU-identical", workers, iters, k)
}

// --- on-device top-k (ann.I8TopKIndex) ---

// Compile-time proof the CUDA index offers the top-k capability, so
// FlatI8.QueryBatch prefers it over score-then-host-select.
var _ ann.I8TopKIndex = (*cudaI8Index)(nil)

// TestCUDATopK_matchesCPU gates the device selection against the CPU's own top-k over
// an index large enough to clear topkMinN. The hits must be IDENTICAL — same indices in
// the same order — not merely a plausible set: the kernel reimplements topHits's
// ordering, so anything else means the two disagree about what "top-k" is.
func TestCUDATopK_matchesCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(101))
	const n, dim, k, M = 60_000, 64, 10, 8 // n > topkMinN so the device path engages
	vecs := randUnit(rng, n, dim)
	cpu := ann.NewFlatI8(vecs)
	g := ann.NewFlatI8(vecs)
	if err := g.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer g.Close()

	queries := randUnit(rng, M, dim)
	got := g.QueryBatch(queries, k)
	for m, q := range queries {
		want := cpu.Query(q, k)
		if len(got[m]) != len(want) {
			t.Fatalf("q%d: %d hits, want %d", m, len(got[m]), len(want))
		}
		for i := range want {
			if got[m][i].Index != want[i].Index {
				t.Fatalf("q%d rank %d: device top-k index %d != CPU %d", m, i, got[m][i].Index, want[i].Index)
			}
			if math.Abs(got[m][i].Score-want[i].Score) > 1e-3 {
				t.Fatalf("q%d rank %d: score %v vs CPU %v", m, i, got[m][i].Score, want[i].Score)
			}
		}
	}
	t.Logf("device top-k ≡ CPU top-%d over %d queries at n=%d", k, M, n)
}

// TestCUDATopK_tieBreak is the sharp one. With EVERY score equal, the ONLY thing
// deciding the answer is the tie-break, so the result must be exactly [0..k-1] —
// index ASC. A kernel that reduced on score alone would return an arbitrary k here and
// still look correct on random data, which is precisely how a wrong tie-break survives.
func TestCUDATopK_tieBreak(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered")
	}
	const n, dim, k, M = 60_000, 32, 12, 3
	// Every vector identical ⇒ every score identical for any query.
	vecs := make([][]float32, n)
	base := make([]float32, dim)
	for d := range base {
		base[d] = float32(1.0 / math.Sqrt(float64(dim)))
	}
	for i := range vecs {
		vecs[i] = base
	}
	g := ann.NewFlatI8(vecs)
	if err := g.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer g.Close()

	queries := make([][]float32, M)
	for m := range queries {
		queries[m] = base
	}
	got := g.QueryBatch(queries, k)
	for m := range M {
		if len(got[m]) != k {
			t.Fatalf("q%d: %d hits, want %d", m, len(got[m]), k)
		}
		for i := range k {
			if got[m][i].Index != i {
				t.Fatalf("q%d rank %d: index %d, want %d — all scores are equal, so the tie-break (index ASC) is the whole answer",
					m, i, got[m][i].Index, i)
			}
		}
	}
	t.Log("all-equal scores ⇒ top-k is [0..k-1]: tie-break matches topHits")
}

// newTestIndex builds a backend + index directly, so the decline paths can be
// exercised on the concrete type. Reaching them through FlatI8 is not possible — the
// index is unexported there — and asserting nothing would make this test vacuous.
func newTestIndex(t *testing.T, vecs [][]float32) *cudaI8Index {
	t.Helper()
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	lib, err := dev.CompileLibrary(w8a8PTX)
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("CompileLibrary: %v", err)
	}
	gemv, err := dev.NewComputePipeline(lib, "gemv_w8a8")
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("gemv: %v", err)
	}
	topk, err := dev.NewComputePipeline(lib, "topk_rows")
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("topk: %v", err)
	}
	vit, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("NewViT: %v", err)
	}
	b := &cudaBackend{dev: dev, q: dev.NewCommandQueue(), gemv: gemv, gemmT: vit.GEMMW8A8Tiled, topk: topk}
	t.Cleanup(dev.ReleaseObjects)

	// Quantize with the package's own quantizeRowInt8 — byte-identical to what
	// FlatI8 stores, so the index under test holds the same codes it would in
	// production.
	n, dim := len(vecs), len(vecs[0])
	bq := make([]int8, n*dim)
	scales := make([]float32, n)
	for i, v := range vecs {
		scales[i] = quantizeRowInt8(v, bq[i*dim:(i+1)*dim])
	}
	idx, err := b.NewI8Index(bq, scales, n, dim)
	if err != nil {
		t.Fatalf("NewI8Index: %v", err)
	}
	return idx.(*cudaI8Index)
}

// TestCUDATopK_declines pins the fallback contract: outside the range the kernel
// implements, TopKBatch must decline so QueryBatch takes the scoring path — and the
// answer must still be right.
func TestCUDATopK_declines(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered")
	}
	rng := rand.New(rand.NewSource(102))
	const dim, k, small = 48, 5, 2_000
	vecs := randUnit(rng, small, dim)
	idx := newTestIndex(t, vecs)
	defer idx.Close()

	q := [][]float32{randUnit(rng, 1, dim)[0]}
	if _, err := idx.TopKBatch(q, k); err == nil {
		t.Errorf("TopKBatch should decline at n=%d < topkMinN=%d", small, topkMinN)
	}
	if _, err := idx.TopKBatch(q, small); err == nil {
		t.Error("TopKBatch should decline for k >= n (topHits's full-sort path)")
	}
	if _, err := idx.TopKBatch(q, 0); err == nil {
		t.Error("TopKBatch should decline for k <= 0")
	}

	// A declining index must leave QueryBatch's answer identical to the CPU's.
	cpu := ann.NewFlatI8(vecs)
	g := ann.NewFlatI8(vecs)
	if err := g.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer g.Close()
	queries := randUnit(rng, 4, dim)
	got := g.QueryBatch(queries, k)
	for m, qq := range queries {
		want := cpu.Query(qq, k)
		for i := range want {
			if got[m][i].Index != want[i].Index {
				t.Fatalf("q%d rank %d after decline: %d != CPU %d", m, i, got[m][i].Index, want[i].Index)
			}
		}
	}
	t.Log("declines below topkMinN, for k>=n and k<=0; the fallback answer is unchanged")
}

// TestCUDAGEMM_batchM1MatchesQuery covers the ONE batch width the tiled GEMM never
// sees. gemmTileMinM is 2, so M=1 is the only batch that took the naive
// thread-per-(query,row) kernel — and is now routed to the warp-per-row GEMV
// instead (backend.go launchGEMM). TestCUDAGEMM_batchParityWithCPU runs at M=16 and
// therefore exercises the tiled path only; without this, the reroute ships ungated.
//
// The assertion is that a 1-query batch equals the single-query Query path exactly,
// which is both the natural contract and the thing the reroute could break: the two
// kernels take their arguments in a different order, so a mis-bound buffer would
// still produce plausible-looking scores.
func TestCUDAGEMM_batchM1MatchesQuery(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(7))
	const n, dim, k = 5000, 128, 10
	vecs := randUnit(rng, n, dim)

	idx := ann.NewFlatI8(vecs)
	if err := idx.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer idx.Close()

	for qi, q := range randUnit(rng, 25, dim) {
		want := idx.Query(q, k)
		got := idx.QueryBatch([][]float32{q}, k)
		if len(got) != 1 {
			t.Fatalf("QueryBatch(M=1) returned %d result sets", len(got))
		}
		if len(got[0]) != len(want) {
			t.Fatalf("query %d: batch returned %d hits, Query returned %d", qi, len(got[0]), len(want))
		}
		for i := range want {
			if got[0][i].Index != want[i].Index || got[0][i].Score != want[i].Score {
				t.Fatalf("query %d hit %d: batch {%d, %v} != Query {%d, %v}",
					qi, i, got[0][i].Index, got[0][i].Score, want[i].Index, want[i].Score)
			}
		}
	}
	t.Logf("M=1 batch ≡ Query over 25 queries, exactly (n=%d dim=%d k=%d)", n, dim, k)
}

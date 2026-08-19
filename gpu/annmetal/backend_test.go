//go:build darwin

package annmetal

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/ann"
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

// TestMetalGEMV_parityWithCPU is the Phase-1 gate: an ann.FlatI8 scored on the GPU
// (EnableGPU) must return exactly the CPU FlatI8's top-k — same indices in the same
// order, scores within the int8/float tolerance — over many random queries. The
// int8 dot is exact integer arithmetic, so the rankings are identical, not merely
// close. Break-it-first below proves the equality check is not vacuous.
func TestMetalGEMV_parityWithCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no Metal backend registered (no GPU?)")
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

// TestMetalGEMV_offBoundaryShapes is the guard for the SIMD-group-per-row GEMV's new
// failure mode: a lane's row is j = gid/W, so a K that isn't a multiple of 4 (scalar
// fallback, not the char4 path), a partial last threadgroup (N×W not a multiple of the
// threadgroup, i.e. N%8≠0 at W=32/tg=256), and N below one SIMD-group all have to land on
// exactly the right rows. The int32 dot is exact regardless of lane order, so parity is
// worst Δ == 0 at every shape, not merely close. The aligned parity test above (n=2000,
// dim=96) exercises none of these — both divide evenly.
func TestMetalGEMV_offBoundaryShapes(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no Metal backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(7))
	Ns := []int{1, 7, 31, 32, 33, 63, 100, 257, 1000}      // tiny, ==W, and N%8≠0 (partial last tg)
	Ks := []int{1, 3, 4, 5, 7, 96, 97, 255, 256, 257, 768} // char4 (K%4==0) and scalar (K%4≠0)
	for _, N := range Ns {
		for _, K := range Ks {
			vecs := randUnit(rng, N, K)
			cpu := ann.NewFlatI8(vecs)
			gpu := ann.NewFlatI8(vecs)
			if err := gpu.EnableGPU(); err != nil {
				t.Fatalf("N=%d K=%d EnableGPU: %v", N, K, err)
			}
			k := min(10, N)
			worst := 0.0
			for _, q := range randUnit(rng, 20, K) {
				want := cpu.Query(q, k)
				got := gpu.Query(q, k)
				if len(got) != len(want) {
					gpu.Close()
					t.Fatalf("N=%d K=%d: got %d hits, want %d", N, K, len(got), len(want))
				}
				for i := range want {
					if got[i].Index != want[i].Index {
						gpu.Close()
						t.Fatalf("N=%d K=%d rank %d: GPU idx %d != CPU idx %d (top-k diverged)", N, K, i, got[i].Index, want[i].Index)
					}
					if d := math.Abs(got[i].Score - want[i].Score); d > worst {
						worst = d
					}
				}
			}
			gpu.Close()
			if worst != 0 {
				t.Errorf("N=%d K=%d: worst score Δ %.3e, want 0 (int32 dot is exact regardless of lane order)", N, K, worst)
			}
		}
	}
	t.Logf("off-boundary GEMV parity: %d×%d shapes, every GPU≡CPU exact", len(Ns), len(Ks))
}

// TestMetalTopK_randomParityWithCPU gates the on-device top-k (topk_rows) on RANDOM,
// all-distinct scores at n ≥ topkMinN — the regime where QueryBatch actually takes the
// device path (TopKBatch), not the host fallback. The all-tie test above proves the
// tie-break; this proves the parallel tree merge picks the same k largest the CPU does,
// in the same order, with worst score Δ 0 (int32 dot ⇒ exact). Without an n ≥ topkMinN
// case, the device merge has no random-data gate at all — TestMetalGEMM_batchParityWithCPU
// runs at n=3000 < topkMinN and so exercises the fallback, never the kernel.
func TestMetalTopK_randomParityWithCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no Metal backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(11))
	const n, dim, k, M = 120_000, 96, 10, 32 // n ≥ topkMinN (100k) → device top-k path
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
				t.Fatalf("q%d rank %d: device top-k index %d != CPU %d (tree merge diverged)", m, i, got[i].Index, want[i].Index)
			}
			if d := math.Abs(got[i].Score - want[i].Score); d > worst {
				worst = d
			}
		}
	}
	t.Logf("device top-k (n=%d≥topkMinN, M=%d) ≡ CPU top-%d: worst score Δ %.3e", n, M, k, worst)
	if worst != 0 {
		t.Errorf("worst score Δ %.3e, want 0 (int32 dot is exact; the merge must not perturb it)", worst)
	}
}

// Compile-time proof that the Metal index is a batch index — so FlatI8.QueryBatch
// takes the batched-GEMM path (ScoreBatch), never the per-query fallback.
var _ ann.I8BatchIndex = (*metalI8Index)(nil)

// TestMetalGEMM_batchParityWithCPU is the Phase-2 gate: scoring a batch of queries
// as one int8 GEMM on the GPU must return exactly the CPU FlatI8's per-query top-k
// for every query in the batch — same indices, scores within tolerance. The int8
// dot is exact, so the rankings are identical.
func TestMetalGEMM_batchParityWithCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no Metal backend registered (no GPU?)")
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

// TestMetalTopK_tieBreakAndFallback gates the on-device top-k selection (TopKBatch, reached
// via QueryBatch): its (score-desc, index-asc) order must match topHits EXACTLY, including on
// ties, and k above the kernel cap must fall back to the full-matrix path and still be correct.
func TestMetalTopK_tieBreakAndFallback(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no Metal backend registered (no GPU?)")
	}
	const n, dim = 100_000, 64 // ≥ topkMinN so k=10 takes the DEVICE top-k path
	// All rows identical → every score ties → the top-k is decided PURELY by the index-asc
	// tie-break, so it must be exactly [0,1,...,k-1]. This is the case a wrong tie-break fails.
	one := make([]float32, dim)
	for i := range one {
		one[i] = 1
	}
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = append([]float32(nil), one...)
	}
	g := ann.NewFlatI8(vecs)
	if err := g.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer g.Close()

	q := [][]float32{append([]float32(nil), one...)}
	for _, k := range []int{10, 20} { // 10 ≤ topkMaxK (device path); 20 > cap (fallback path)
		got := g.QueryBatch(q, k)[0]
		if len(got) != k {
			t.Fatalf("k=%d: got %d hits", k, len(got))
		}
		for i := range k {
			if got[i].Index != i {
				t.Fatalf("k=%d rank %d: index %d, want %d — tie-break (or fallback) diverges from topHits index-asc", k, i, got[i].Index, i)
			}
		}
		t.Logf("k=%d: all-tie top-k == [0..k-1] (index-asc tie-break holds%s)", k, map[bool]string{true: ", via device", false: ", via fallback"}[k <= 16])
	}
}

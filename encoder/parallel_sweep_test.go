package encoder

import (
	"math/rand"
	"runtime"
	"testing"
)

// The parallel-axis sweep. perf-campaign §7.12/§7.13 both ended with the same
// suspicion from two different models: the row-split path underperforms on
// amd64, where parallelThreshold's tuning table was measured on an 8-core M1
// Pro. This sweep is the arbiter.
//
// The structural asymmetry it is testing: a ROW split gives every worker the
// whole of b, so b is streamed `workers` times; a COLUMN split partitions b, so
// it is streamed once. In a transformer linear, a is [L,K] activations (small)
// and b is [N,K] weights (large) — 9.4 MB for one fc11 — so the row split
// multiplies the dominant memory traffic by the worker count.
//
// It also has a second, independent handicap: matmulBTBlockedIntoParallel caps
// workers at (M+31)/32, so M=91 gets THREE workers on a 16-thread box.
//
// Weights come from a 12-deep bank rotated per iteration, so no run gets to
// stream the same b out of L3 repeatedly — the mistake that would make every
// parallel variant look good.
type sweepShape struct {
	name    string
	M, K, N int
}

func benchAxis(b *testing.B, sh sweepShape, fn func(a, w, dst []float32, M, K, N int)) {
	rng := rand.New(rand.NewSource(int64(sh.M * sh.N)))
	const bank = 12 // one per transformer layer
	a := randF32(rng, sh.M*sh.K)
	ws := make([][]float32, bank)
	for i := range ws {
		ws[i] = randF32(rng, sh.N*sh.K)
	}
	dst := make([]float32, sh.M*sh.N)
	i := 0
	b.ResetTimer()
	for b.Loop() {
		fn(a, ws[i%bank], dst, sh.M, sh.K, sh.N)
		i++
	}
}

func BenchmarkParallelAxis(b *testing.B) {
	shapes := []sweepShape{
		// BERT-base / GTE trunk linears at representative sequence lengths.
		{"fc11_L22", 22, 768, 3072},
		{"fc11_L91", 91, 768, 3072},
		{"fc11_L357", 357, 768, 3072},
		{"fc2_L91", 91, 3072, 768},
		{"fc2_L357", 357, 3072, 768},
		{"qkv_L91", 91, 768, 2304},
		{"gte_upgate_L175", 175, 768, 6144},
		{"gte_upgate_L690", 690, 768, 6144},
		{"gte_down_L690", 690, 3072, 768},
	}
	for _, sh := range shapes {
		b.Run(sh.name+"/serial", func(b *testing.B) {
			benchAxis(b, sh, matmulBTBlockedInto)
		})
		b.Run(sh.name+"/rows", func(b *testing.B) {
			b.ReportMetric(float64(rowWorkersFor(sh.M)), "workers")
			benchAxis(b, sh, matmulBTBlockedIntoParallel)
		})
		b.Run(sh.name+"/cols", func(b *testing.B) {
			benchAxis(b, sh, func(a, w, dst []float32, M, K, N int) {
				matmulBTColsInto(a, w, dst, M, K, N)
			})
		})
	}
}

// rowWorkersFor mirrors matmulBTBlockedIntoParallel's worker calculation, so the
// sweep can report how much parallelism the row split actually gets.
func rowWorkersFor(M int) int {
	w := runtime.NumCPU()
	if maxByRows := (M + minRowsPerWorker - 1) / minRowsPerWorker; w > maxByRows {
		w = maxByRows
	}
	return w
}

package bench

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/ann"
)

func metricsCorpus(n, d int) ([][]float32, [][]float32) {
	rng := rand.New(rand.NewSource(1))
	unit := func() []float32 {
		v := make([]float32, d)
		var norm float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		inv := float32(1 / math.Sqrt(norm))
		for j := range v {
			v[j] *= inv
		}
		return v
	}
	corpus := make([][]float32, n)
	for i := range corpus {
		corpus[i] = unit()
	}
	queries := make([][]float32, 16)
	for i := range queries {
		queries[i] = unit()
	}
	return corpus, queries
}

// TestHarness_reportsAllocsAndQPS gates perf-campaign item 1's harness half.
// Latency alone was not enough to see several campaign results: item 8 removed
// 12.7 MiB per call for no time change, item 4 cut allocation 99% for −5%, and
// item 16 found throughput plateauing at a memory-bandwidth ceiling that serial
// latency over-predicts. So the harness now reports allocation and CONCURRENT
// throughput too, and this checks those fields are populated rather than
// silently zero.
func TestHarness_reportsAllocsAndQPS(t *testing.T) {
	corpus, queries := metricsCorpus(2000, 64)
	results := Run(corpus, queries, 10, ann.Config{})
	if len(results) == 0 {
		t.Fatal("no results")
	}
	for _, r := range results {
		if r.QPS <= 0 {
			t.Errorf("%s: QPS = %v, want > 0 — the concurrent pass did not run", r.Name, r.QPS)
		}
		if r.Concurrency < 1 {
			t.Errorf("%s: Concurrency = %d, want >= 1", r.Name, r.Concurrency)
		}
		if r.BytesPerQuery <= 0 {
			t.Errorf("%s: BytesPerQuery = %v, want > 0 — every query allocates its hit slice",
				r.Name, r.BytesPerQuery)
		}
		if r.AllocsPerQuery <= 0 {
			t.Errorf("%s: AllocsPerQuery = %v, want > 0", r.Name, r.AllocsPerQuery)
		}
		// Throughput must not be a restatement of serial latency: with several
		// cores it should beat 1/Mean, and it must not exceed cores/Mean.
		if r.Mean > 0 {
			serial := 1000 / r.Mean // queries/s from mean ms
			if ceil := serial * float64(r.Concurrency) * 1.5; r.QPS > ceil {
				t.Errorf("%s: QPS %.0f exceeds %.0f (cores × serial rate × 1.5) — implausible",
					r.Name, r.QPS, ceil)
			}
		}
		t.Logf("%-7s p50=%.3fms QPS×%d=%.0f allocs/q=%.0f B/q=%.0f",
			r.Name, r.P50, r.Concurrency, r.QPS, r.AllocsPerQuery, r.BytesPerQuery)
	}
	if tbl := Table(results); !strings.Contains(tbl, "QPS") || !strings.Contains(tbl, "allocs/q") {
		t.Errorf("Table omits the new columns:\n%s", tbl)
	}
}

// TestHarness_warmupExcluded checks the warm-up runs before timing. Without it
// the first query's cold-start lands in the latency distribution, which shows up
// worst at p99.
func TestHarness_warmupExcluded(t *testing.T) {
	corpus, queries := metricsCorpus(1000, 64)
	results := Run(corpus, queries, 10, ann.Config{})
	for _, r := range results {
		if r.P99 <= 0 || r.P50 <= 0 {
			t.Fatalf("%s: degenerate latencies p50=%v p99=%v", r.Name, r.P50, r.P99)
		}
		// A cold first call is typically orders of magnitude slower than steady
		// state; with warm-up excluded the spread stays bounded.
		if r.P99 > r.P50*100 {
			t.Errorf("%s: p99 %.4fms is >100× p50 %.4fms — is the cold first query "+
				"still inside the measured loop?", r.Name, r.P99, r.P50)
		}
	}
}

package encoder

import (
	"math/rand"
	"os"
	"strings"
	"testing"
)

func ceRerankCorpus(n int, rng *rand.Rand) (string, []string) {
	words := strings.Fields("encoding json unmarshal struct reflection buffer reader " +
		"context deadline cancel goroutine channel select mutex atomic pointer slice map")
	query := "how do i unmarshal json into a generic struct in go"
	docs := make([]string, n)
	for i := range docs {
		nw := 20 + rng.Intn(180)
		var b strings.Builder
		for j := range nw {
			b.WriteString(words[(i*7+j)%len(words)])
			b.WriteByte(' ')
		}
		docs[i] = b.String()
	}
	return query, docs
}

func loadBenchCE(tb testing.TB) *CrossEncoder {
	tb.Helper()
	const dir = "../testdata/crossencoder-model"
	if _, err := os.Stat(dir); err != nil {
		tb.Skip("testdata/crossencoder-model/ not present; see scripts/README.md")
	}
	ce, err := LoadCrossEncoder(dir)
	if err != nil {
		tb.Fatalf("LoadCrossEncoder: %v", err)
	}
	return ce
}

// BenchmarkRerankN50 is the arbiter for item 28: one query against 50
// documents, the shape a reranker actually runs. The serial arm is what a caller
// writes today with no batch API.
func BenchmarkRerankN50(b *testing.B) {
	ce := loadBenchCE(b)
	defer func() { _ = ce.Close() }()

	rng := rand.New(rand.NewSource(28))
	query, docs := ceRerankCorpus(50, rng)

	b.Run("serial_Score", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for _, d := range docs {
				s, err := ce.Score(query, d)
				if err != nil {
					b.Fatal(err)
				}
				sinkScore = s
			}
		}
	})
	b.Run("ScoreBatch", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			v, err := ce.ScoreBatch(query, docs, 0)
			if err != nil {
				b.Fatal(err)
			}
			sinkScores = v
		}
	})
}

var sinkScores []float32

// TestScoreBatch_matchesScore is the correctness gate for item 28. ScoreBatch
// reorders the work (longest pair first) and runs several forwards at once, so
// the risks are a scatter-back bug and any load-dependence in the forward. Both
// would show as a mismatch against a plain Score loop.
//
// Exact equality: nothing about batching may change a score. Concurrency raises
// the in-flight count, which makes the intra-op matmul gate decline — and that
// path is bit-identical by construction (gated in parallel_cols_test), so it
// must be bit-identical here too.
func TestScoreBatch_matchesScore(t *testing.T) {
	ce := loadBenchCE(t)
	defer func() { _ = ce.Close() }()

	rng := rand.New(rand.NewSource(280))
	query, docs := ceRerankCorpus(12, rng)
	// A pair that overflows maxSeq, so the longest_first trim runs; and an empty
	// document, the degenerate end of the same path.
	docs = append(docs, strings.Repeat("overflow token ", 5000), "")

	want := make([]float32, len(docs))
	for i, d := range docs {
		s, err := ce.Score(query, d)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = s
	}

	for _, conc := range []int{1, 3, 0} { // 0 = NumCPU
		got, err := ce.ScoreBatch(query, docs, conc)
		if err != nil {
			t.Fatalf("ScoreBatch: %v", err)
		}
		if len(got) != len(want) {
			t.Fatalf("conc=%d: %d scores, want %d", conc, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("conc=%d doc %d: ScoreBatch %v, Score %v — batching must not "+
					"change a score, and results must land in the caller's order",
					conc, i, got[i], want[i])
			}
		}
	}
	if out, err := ce.ScoreBatch(query, nil, 0); err != nil || out != nil {
		t.Errorf("ScoreBatch with no documents = (%v, %v), want (nil, nil)", out, err)
	}
}

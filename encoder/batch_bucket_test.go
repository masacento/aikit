package encoder

import (
	"math/rand"
	"os"
	"strings"
	"testing"
)

// batchTexts builds a deliberately RAGGED corpus — the case item 14 is about.
// Uniform lengths hide the padding waste entirely.
func batchTextsMax(n, maxWords int, rng *rand.Rand) []string {
	words := strings.Fields("func parse json into a generic struct with reflection and " +
		"encoding decode marshal unmarshal buffer reader writer context cancel deadline")
	out := make([]string, n)
	for i := range out {
		// Lmax/mean ≈ 2, so ~50% of linear-layer FLOPs land on pad.
		nw := 3 + rng.Intn(maxWords)
		var b strings.Builder
		for j := range nw {
			b.WriteString(words[(i+j)%len(words)])
			b.WriteByte(' ')
		}
		out[i] = b.String()
	}
	return out
}

// TestEncodeBatch_matchesEncodeRagged is the property item 14 must preserve, on
// the input shape item 14 is ABOUT. batch_test.go already checks four short
// texts of similar length; length-bucketing reorders which sequences share a
// forward, and only a ragged corpus exercises that.
//
// The property: batching
// changes only which sequences share a forward, and linalg guarantees
// M-invariance, so every embedding must be BIT-IDENTICAL to the same text
// encoded alone. (Before item 27 removed the naive/blocked threshold this was
// not true; the threshold was the last source of batch-vs-single divergence.)
func TestEncodeBatch_matchesEncodeRagged(t *testing.T) {
	const dir = "../testdata/encoder-model"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("testdata/encoder-model/ not present; see scripts/README.md")
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = m.Close() }()

	rng := rand.New(rand.NewSource(14))
	// 12 texts of ≤120 words: enough raggedness to exercise bucketing without
	// making this a two-minute test (the 24×400-word version took 131 s).
	texts := batchTextsMax(12, 120, rng)
	isQ := make([]bool, len(texts))
	for i := range isQ {
		isQ[i] = i%3 == 0
	}

	for _, conc := range []int{1, 3} {
		got, err := m.EncodeBatch(texts, isQ, conc)
		if err != nil {
			t.Fatalf("EncodeBatch: %v", err)
		}
		var diffs int
		for i := range texts {
			want, err := m.Encode(texts[i], isQ[i])
			if err != nil {
				t.Fatal(err)
			}
			if len(got[i]) != len(want) {
				t.Fatalf("conc=%d text %d: dim %d vs %d", conc, i, len(got[i]), len(want))
			}
			for j := range want {
				if got[i][j] != want[j] {
					diffs++
					break
				}
			}
		}
		if diffs != 0 {
			t.Errorf("concurrency=%d: %d/%d embeddings differ from a solo Encode — "+
				"batching must be bit-identical (linalg M-invariance; pad rows are never read)",
				conc, diffs, len(texts))
		}
	}
}

// TestBucketByLength pins the three bounds, because two of them exist for
// reasons that are easy to regress away.
func TestBucketByLength(t *testing.T) {
	lensOf := func(l ...int) []int { return l }
	orderOf := func(lens []int) []int {
		o := make([]int, len(lens))
		for i := range o {
			o[i] = i
		}
		return o // callers below supply already-ascending lengths
	}
	t.Run("token budget caps B*Lmax", func(t *testing.T) {
		lens := lensOf(1024, 1024, 1024, 1024, 1024, 1024)
		got := bucketByLength(orderOf(lens), lens, 1000)
		for _, b := range got {
			if len(b)*1024 > batchTokenBudget {
				t.Errorf("bucket of %d × 1024 tokens exceeds the %d budget", len(b), batchTokenBudget)
			}
		}
	})
	t.Run("worker cap keeps units dispatchable", func(t *testing.T) {
		lens := make([]int, 8)
		for i := range lens {
			lens[i] = 30 // short: the token budget alone would allow one bucket
		}
		got := bucketByLength(orderOf(lens), lens, 1)
		if len(got) != 8 {
			t.Errorf("got %d buckets, want 8 — without the per-worker cap a small "+
				"corpus of short texts collapses into one bucket and only one worker runs", len(got))
		}
	})
	t.Run("every index appears exactly once", func(t *testing.T) {
		rng := rand.New(rand.NewSource(3))
		lens := make([]int, 200)
		for i := range lens {
			lens[i] = 1 + rng.Intn(600)
		}
		order := orderOf(lens)
		got := bucketByLength(order, lens, 7)
		seen := map[int]int{}
		for _, b := range got {
			if len(b) == 0 {
				t.Fatal("empty bucket")
			}
			for _, i := range b {
				seen[i]++
			}
		}
		for i := range lens {
			if seen[i] != 1 {
				t.Fatalf("index %d appears %d times across buckets, want 1", i, seen[i])
			}
		}
	})
	t.Run("a single oversized sequence still gets a bucket", func(t *testing.T) {
		lens := lensOf(batchTokenBudget * 4)
		got := bucketByLength(orderOf(lens), lens, 8)
		if len(got) != 1 || len(got[0]) != 1 {
			t.Errorf("got %v, want one bucket holding the one sequence — a sequence "+
				"larger than the whole budget must not be dropped", got)
		}
	})
}

// BenchmarkEncodeBatchRagged is the arbiter for item 14.
//
// It must run with len(texts) > concurrency, or there is nothing to measure: at
// B=1 every sequence is its own forward and no padding exists in EITHER version.
// The old partition wasted work only inside a multi-sequence chunk, so the
// benchmark fixes concurrency below the corpus size to force B>1 — with
// concurrency=NumCPU and 16 texts the first version of this benchmark measured
// two identical B=1 paths and reported, correctly, no difference.
//
// A UNIFORM batch of the same size is the control: bucketing should neither help
// nor hurt when every sequence is already the same length.
func BenchmarkEncodeBatchRagged(b *testing.B) {
	const dir = "../testdata/encoder-model"
	if _, err := os.Stat(dir); err != nil {
		b.Skip("testdata/encoder-model/ not present; see scripts/README.md")
	}
	m, err := Load(dir)
	if err != nil {
		b.Fatalf("Load: %v", err)
	}
	defer func() { _ = m.Close() }()

	const n, conc = 32, 4 // B = 8 per forward
	rng := rand.New(rand.NewSource(140))
	ragged := batchTextsMax(n, 100, rng)
	uniform := make([]string, n)
	for i := range uniform {
		uniform[i] = ragged[len(ragged)/2]
	}
	for _, tc := range []struct {
		name  string
		texts []string
	}{
		{"ragged", ragged},
		{"uniform", uniform},
	} {
		isQ := make([]bool, len(tc.texts))
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				v, err := m.EncodeBatch(tc.texts, isQ, conc)
				if err != nil {
					b.Fatal(err)
				}
				sinkBatch = v
			}
		})
	}
}

var sinkBatch [][]float32

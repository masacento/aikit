package bm25

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// loadRealSource reads a real Go source file from this repo — non-trivial in
// size and full of camelCase / snake_case / digit-bearing identifiers, which is
// what the tokenizer's split paths exist for.
//
// It used to point at `../search/index.go`, a leftover `ken` path. aikit has no
// search/ directory, so every benchmark in this file SKIPPED, and skipping is
// green: `bm25.Tokenize` — the hottest indexing function in the package — and
// `bm25.TopK` had zero live benchmark coverage while looking fine
// (perf-campaign item 1).
//
// It now FAILS rather than skips. The fixture is a file checked into this
// repository; if it is missing, the benchmark is broken and should say so
// instead of quietly reporting nothing.
func loadRealSource(b *testing.B) string {
	b.Helper()
	path := filepath.Join("..", "ann", "hnsw.go")
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatalf("in-repo fixture %s is missing: %v", path, err)
	}
	return string(data)
}

func BenchmarkTokenize(b *testing.B) {
	src := loadRealSource(b)
	b.SetBytes(int64(len(src)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Tokenize(src)
	}
}

// buildBenchCorpus builds a synthetic N-document corpus from a single
// real source file by splitting it into roughly equal line-count chunks.
// Each "document" is the chunk's lines re-joined with newlines, then
// tokenized. Deterministic; same input always produces the same corpus.
func buildBenchCorpus(src string, numDocs int) [][]string {
	lines := strings.Split(src, "\n")
	if numDocs <= 0 {
		numDocs = 1
	}
	per := max(len(lines)/numDocs, 1)
	docs := make([][]string, 0, numDocs)
	for i := 0; i < numDocs && i*per < len(lines); i++ {
		end := min((i+1)*per, len(lines))
		chunk := strings.Join(lines[i*per:end], "\n")
		docs = append(docs, Tokenize(chunk))
	}
	return docs
}

func BenchmarkScore(b *testing.B) {
	src := loadRealSource(b)
	// Table-driven: small / medium corpus. The 1000-doc size is the
	// briefing's prediction-#2-falsification target; the 100-doc size
	// provides a fast inner-loop variant.
	cases := []struct {
		name string
		n    int
	}{
		{"N100", 100},
		{"N1000", 1000},
	}
	query := Tokenize("index search build chunks")
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			corpus := buildBenchCorpus(src, tc.n)
			if len(corpus) == 0 {
				b.Skipf("corpus build produced 0 docs for N=%d", tc.n)
			}
			ix := Build(corpus)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = ix.TopK(query, 10)
			}
		})
	}
}

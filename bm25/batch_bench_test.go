package bm25

import (
	"os"
	"testing"
)

// realSource reads a real in-repo Go file. The existing loadRealSource points at
// ../search/index.go — ken's layout, absent here — so those benchmarks silently
// SKIP. This one uses a file that exists, so the bm25 numbers are actually
// measured rather than reported as "no data".
func realSource(tb testing.TB) string {
	tb.Helper()
	for _, p := range []string{"../encoder/bert.go", "../ann/hnsw.go", "../linalg/linalg.go"} {
		if data, err := os.ReadFile(p); err == nil {
			return string(data)
		}
	}
	tb.Skip("no in-repo source file found")
	return ""
}

func BenchmarkTokenizeReal(b *testing.B) {
	src := realSource(b)
	b.SetBytes(int64(len(src)))
	b.ReportAllocs()
	for b.Loop() {
		sinkToks = Tokenize(src)
	}
}

func BenchmarkBuildReal(b *testing.B) {
	docs := buildBenchCorpus(realSource(b), 1000)
	b.ReportAllocs()
	for b.Loop() {
		sinkIndex = Build(docs)
	}
}

// BenchmarkScoreReal exercises the posting scan — items 10 and 29's target. The
// query mixes a very common term (scans a long posting list) with rarer ones.
func BenchmarkScoreReal(b *testing.B) {
	docs := buildBenchCorpus(realSource(b), 2000)
	ix := Build(docs)
	for _, tc := range []struct {
		name  string
		query []string
	}{
		{"common", []string{"the", "b", "if", "err"}},
		{"mixed", []string{"encoder", "layer", "weights", "attention", "b"}},
		{"rare", []string{"positionoffset", "roberta"}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkRes = ix.TopK(tc.query, 10)
			}
		})
	}
}

var (
	sinkToks  []string
	sinkIndex *Index
	sinkRes   []Result
)

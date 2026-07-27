package bm25

import (
	"fmt"
	"testing"
)

func benchDocs(nDocs, nTok, vocab int) [][]string {
	docs := make([][]string, nDocs)
	for d := range docs {
		toks := make([]string, nTok)
		for i := range toks {
			toks[i] = fmt.Sprintf("t%d", (d*7+i*13)%vocab)
		}
		docs[d] = toks
	}
	return docs
}

func BenchmarkBuild(b *testing.B) {
	docs := benchDocs(2000, 300, 4000)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = Build(docs)
	}
}

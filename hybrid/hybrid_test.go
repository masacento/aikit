package hybrid

import (
	"reflect"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/fuse"
)

func testCorpus() (*ann.Flat, *bm25.Index) {
	// Three unit vectors at increasing distance from the query direction (1,0);
	// deliberately NOT in the same relevance order as the lexical side, so a
	// bug that used only one signal (instead of fusing both) would show up as
	// a wrong Retriever.Query order.
	vecs := [][]float32{
		{0, 1},         // doc 0: dense-irrelevant (90° away)
		{0.9995, 0.03}, // doc 1: dense-near
		{0.7, 0.71},    // doc 2: dense-middling
	}
	docs := [][]string{
		{"fox", "jumps"},        // doc 0: lexical match
		{"lazy", "dog"},         // doc 1: no lexical match
		{"quick", "fox", "run"}, // doc 2: lexical match
	}
	return ann.New(vecs), bm25.Build(docs)
}

func TestRetriever_matchesHandWiredFuse(t *testing.T) {
	dense, lexical := testCorpus()
	r := New(dense, lexical)

	queryVec := []float32{1, 0}
	queryTokens := []string{"fox"}

	got := r.Query(queryVec, queryTokens, 10)

	den := dense.Query(queryVec, 10)
	lex := lexical.TopK(queryTokens, 10)
	want := fuse.RRF(fuse.DefaultK,
		fuse.Keys(den, func(h ann.Hit) int { return h.Index }),
		fuse.Keys(lex, func(res bm25.Result) int { return res.Doc }),
	)

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Retriever.Query = %v, want %v (hand-wired fuse.RRF)", got, want)
	}
	// Sanity: both signals actually contributed — doc 0 (lexical-only) and
	// doc 1 (dense-only) must both appear, or this test would pass vacuously
	// even if Query silently dropped one signal.
	seen := map[int]bool{}
	for _, res := range got {
		seen[res.Key] = true
	}
	if !seen[0] || !seen[1] {
		t.Fatalf("expected both dense-only (1) and lexical-only (0) docs in the fused result, got %v", got)
	}
}

func TestRetriever_emptyShortlist(t *testing.T) {
	dense, lexical := testCorpus()
	r := New(dense, lexical)
	if got := r.Query([]float32{1, 0}, []string{"fox"}, 0); len(got) != 0 {
		t.Errorf("Query(shortlist=0) = %v, want empty", got)
	}
}

func TestRetriever_noLexicalMatch(t *testing.T) {
	dense, lexical := testCorpus()
	r := New(dense, lexical)
	// A query with no BM25 hits must still return the dense ranking, not
	// error or drop everything (RRF fuses an empty lexical list fine).
	got := r.Query([]float32{1, 0}, []string{"nonexistent"}, 10)
	if len(got) == 0 {
		t.Fatal("Query with no lexical match returned nothing, want the dense-only ranking")
	}
	if got[0].Key != 1 { // doc 1 is the closest dense match
		t.Errorf("top result = doc %d, want 1 (dense-nearest)", got[0].Key)
	}
}

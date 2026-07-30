package bm25

import (
	"math"
	"strings"
	"unsafe"
)

// BM25 defaults. These are the Lucene / bm25s defaults; ken's DESIGN.md
// Stage 1 validates ranking against semble's SearchMode.BM25, so the exact
// variant is pinned to what bm25s uses by default (the Lucene IDF, see
// query.go).
const (
	DefaultK1 = 1.5
	DefaultB  = 0.75
)

// posting is deliberately 8 bytes, not 16 (perf-campaign item 29). The scoring
// scan is a linear walk over posting lists and is memory-bound, so the struct
// width IS the scan cost: int/int made it 16 B where sparse's equivalent has
// always been 8. int32 is ample — Build rejects a corpus that would overflow it.
type posting struct {
	doc int32
	tf  int32
}

// sizeofPosting is asserted by TestPostingWidth; see posting's comment.
const sizeofPosting = unsafe.Sizeof(posting{})

// Index is an immutable BM25 inverted index over a fixed document set.
// Documents are referenced by their position in the slice passed to Build.
type Index struct {
	K1, B  float64
	docLen []int
	// norm[d] is docLen[d]/avgdl, precomputed at Build. The scoring loop used to
	// divide per POSTING; this makes it a load (item 10). Computed with exactly
	// the expression the loop used, so scores are bit-identical.
	norm     []float64
	avgdl    float64
	postings map[string][]posting
	df       map[string]int
}

// Build constructs the index from already-tokenized documents (use
// Tokenize). docs[i] is document i's token stream; empty docs are allowed
// and simply score zero.
func Build(docs [][]string) *Index {
	// posting packs doc and tf into int32s (item 29). Both bounds are far beyond
	// any real corpus, but a silent truncation here would corrupt the index, so
	// it panics rather than wraps.
	if len(docs) > math.MaxInt32 {
		panic("bm25: corpus exceeds 2^31-1 documents")
	}
	ix := &Index{
		K1:       DefaultK1,
		B:        DefaultB,
		docLen:   make([]int, len(docs)),
		postings: make(map[string][]posting),
		df:       make(map[string]int),
	}
	var total int
	// One term-frequency map reused across all documents (clear per doc) instead of
	// a fresh make per document — at ~378k chunks that churned multiple GB of
	// short-lived map storage and 378k map-header allocations (audit #20). Output
	// is identical: each doc appends exactly one posting per term and docs are
	// processed in order, so postings[term] is doc-ordered regardless of the inner
	// map-iteration order.
	tf := make(map[string]int, 512)
	for d, toks := range docs {
		ix.docLen[d] = len(toks)
		total += len(toks)
		clear(tf)
		for _, t := range toks {
			tf[t]++
		}
		for term, f := range tf {
			// Intern the retained key (task-perf-memoization §1b). toks[i] is a view
			// into the caller's document text / the tokenizer's per-call arena, so using
			// it directly as a map key would pin that whole backing array for the Index's
			// lifetime — a 356-file index retained 4.79 MB while its distinct terms were
			// 0.11 MB. On a term's FIRST occurrence we store a compact strings.Clone; every
			// later occurrence matches that existing key (Go leaves the stored key header
			// untouched on value update), so no view is ever retained. Byte-identical keys
			// ⇒ identical scores. Single-threaded here, so no lock — unlike a Tokenize-side
			// interner, this never touches the concurrent hot path.
			key := term
			if _, seen := ix.df[key]; !seen {
				key = strings.Clone(term)
			}
			ix.postings[key] = append(ix.postings[key], posting{doc: int32(d), tf: int32(f)})
			ix.df[key]++
		}
	}
	if len(docs) > 0 {
		ix.avgdl = float64(total) / float64(len(docs))
	}
	// Precompute the length normalization per document (item 10). The expression
	// is character-for-character the one the scoring loop used, so every score is
	// bit-identical; the loop just stops dividing once per posting.
	ix.norm = make([]float64, len(docs))
	if ix.avgdl > 0 {
		for d := range ix.norm {
			ix.norm[d] = float64(ix.docLen[d]) / ix.avgdl
		}
	}
	return ix
}

// N is the number of indexed documents.
func (ix *Index) N() int { return len(ix.docLen) }

// Package late is ColBERT-style late-interaction (MaxSim) reranking: instead of
// pooling a query or document down to one vector (a bi-encoder, encoder's usual
// path) or scoring a (query, document) pair jointly in one forward
// (encoder.CrossEncoder), late interaction keeps every token's own
// contextualized vector and lets each query token independently find its
// best-matching document token:
//
//	MaxSim(q, d) = Σ_i max_j cos(q_i, d_j)
//
// # Inference-optional
//
// Like sparse, this package is the SCORING half: MaxSim / ScoreBatch / Index
// operate on PRE-COMPUTED per-token vectors, produced out of band or in-process
// via encoder.Model.EncodeTokens / EncodeTokensWithIDs — built for exactly this,
// see their doc comments. Every row of every matrix must be L2-normalized
// (embed.L2Normalize); MaxSim treats a dot product as cosine similarity, the
// same unit-vector contract ann.Flat and embed.Encode use.
//
// # A reranker, not a first-stage retriever
//
// A document's token matrix is L tokens × D floats — roughly L× the footprint
// of its single pooled vector. Cheap to compute (EncodeTokens is one forward,
// no more expensive than Encode) but too large to hold for a whole corpus the
// way ann/bm25/sparse do. Index is sized for a SHORTLIST — the same role
// encoder.CrossEncoder plays in examples/rag — not a corpus-scale index. Get the
// shortlist from a dense+lexical fuse.RRF first, then rerank it here; see
// examples/colbert.
package late

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

// Hit is one scored document, highest Score (MaxSim) first. The field name
// Index matches ann.Hit/sparse.Hit, so a MaxSim ranking composes with fuse.Keys
// the same way — though as a shortlist reranker it typically stands as the
// FINAL stage rather than one more leg to fuse (see encoder.CrossEncoder's
// identical role in examples/rag).
type Hit struct {
	Index int
	Score float64
}

// MaxSim scores query against doc: for every query token vector, its highest
// cosine similarity (a plain dot product — every row must be L2-normalized)
// against any document token vector, summed over query tokens.
//
// An empty query scores 0 (no terms to sum). An empty doc gives every query
// token a max of 0 (no candidate to match it against), not an error — the
// natural result of an empty inner-max loop, the same "degenerate input,
// defined output" posture as embed.L2Normalize's zero-vector case.
func MaxSim(query, doc [][]float32) float64 {
	var sum float64
	for _, q := range query {
		var best float32
		for j, d := range doc {
			// j==0 unconditionally takes the first candidate: cosine similarity
			// can be negative, so seeding best at 0 would wrongly floor a doc
			// whose every token is a poor match at "no match" instead of its
			// true (negative) best.
			if s := linalg.Dot(q, d); j == 0 || s > best {
				best = s
			}
		}
		sum += float64(best)
	}
	return sum
}

// ScoreBatch scores one query against many document token-matrices, returning
// MaxSim(query, docs[i]) per document in the caller's order. concurrency <= 0
// means NumCPU.
//
// Mirrors encoder.CrossEncoder.ScoreBatch's document-parallel work-stealing
// dispatch, minus that function's longest-first sort: a cross-encoder pays a
// full BERT forward per pair, cheap enough per-pair here (plain dot products,
// no forward pass) that a work-stealing split already keeps every core busy
// without needing to pre-balance the queue.
func ScoreBatch(query [][]float32, docs [][][]float32, concurrency int) []float64 {
	if len(docs) == 0 {
		return nil
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	concurrency = min(concurrency, len(docs))

	out := make([]float64, len(docs))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			for {
				i := int(next.Add(1)) - 1
				if i >= len(docs) {
					return
				}
				out[i] = MaxSim(query, docs[i])
			}
		})
	}
	wg.Wait()
	return out
}

// Index is a MaxSim reranker over a fixed shortlist of document token-matrices
// — see the package doc for why this is sized for a shortlist, not a corpus.
// Matrices are used by reference, not copied.
type Index struct {
	docs [][][]float32
}

// New builds an Index over docs — one token-matrix per document, each row
// L2-normalized (see the package doc).
func New(docs [][][]float32) *Index {
	return &Index{docs: docs}
}

// Len returns the number of documents in the index.
func (ix *Index) Len() int { return len(ix.docs) }

// Query scores query against every document (ScoreBatch, so document-parallel)
// and returns the k highest-scoring, best first. k <= 0 or an empty index
// returns nil.
func (ix *Index) Query(query [][]float32, k int) []Hit {
	if k <= 0 || len(ix.docs) == 0 {
		return nil
	}
	scores := ScoreBatch(query, ix.docs, 0)
	sel := topk.New[int](k)
	for i, s := range scores {
		sel.Push(i, s)
	}
	res := sel.Result()
	hits := make([]Hit, len(res))
	for i, r := range res {
		hits[i] = Hit{Index: r.Item, Score: r.Score}
	}
	return hits
}

// Command colbert is the end-to-end ColBERT-style late-interaction rerank
// pipeline in one file:
//
//	chunk → embed → (ANN + BM25) → RRF fuse → MaxSim rerank → top-K
//
// Identical shortlist mechanism to examples/rag (same corpus, same dense+
// lexical fuse), swapping the final-stage reranker: rag uses
// encoder.CrossEncoder (one joint forward per (query, doc) pair); this uses
// late.Index — every candidate keeps its own per-token vectors
// (encoder.Model.EncodeTokens, built explicitly for this), and each query
// token independently finds its best-matching document token
// (late.MaxSim = Σ_i max_j cos(q_i, d_j)) instead of the pair being squashed
// through one shared forward. Run both examples on the same query to compare.
//
// It needs two local models (skipped-clean if absent, so `go build ./...`
// always compiles and `go run` without flags just prints guidance):
//
//	go run ./examples/colbert \
//	    --embed-model   testdata/model \
//	    --encoder-model testdata/encoder-model \
//	    --q "read a file line by line"
//
// --encoder-model wants a CodeRankEmbed-shaped checkpoint (encoder.Load) — see
// scripts/README.md's "Fetching testdata/encoder-model" section — because
// EncodeTokens needs a real contextualizing transformer forward, not
// Model2Vec's static per-token vectors (--embed-model powers only the
// first-stage dense retrieval, exactly as in examples/rag).
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/chunk"
	_ "github.com/townsendmerino/aikit/chunk/regex" // registers the "regex" chunker via init()
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/fuse"
	"github.com/townsendmerino/aikit/late"
)

// The same small code corpus examples/rag and examples/splade use, so all
// three examples are directly comparable on the same query.
var corpus = []struct{ name, src string }{
	{"readlines.go", "func readLines(path string) ([]string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\tdefer f.Close()\n\tvar lines []string\n\ts := bufio.NewScanner(f)\n\tfor s.Scan() {\n\t\tlines = append(lines, s.Text())\n\t}\n\treturn lines, s.Err()\n}"},
	{"json.go", "func parseConfig(b []byte) (*Config, error) {\n\tvar c Config\n\tif err := json.Unmarshal(b, &c); err != nil {\n\t\treturn nil, fmt.Errorf(\"parse config: %w\", err)\n\t}\n\treturn &c, nil\n}"},
	{"server.go", "func handler(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode(map[string]string{\"ok\": \"true\"})\n}"},
	{"math.go", "func fib(n int) int {\n\tif n < 2 {\n\t\treturn n\n\t}\n\treturn fib(n-1) + fib(n-2)\n}"},
	{"hash.go", "func sha256File(path string) (string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn \"\", err\n\t}\n\tdefer f.Close()\n\th := sha256.New()\n\tif _, err := io.Copy(h, f); err != nil {\n\t\treturn \"\", err\n\t}\n\treturn hex.EncodeToString(h.Sum(nil)), nil\n}"},
}

func main() {
	embedDir := flag.String("embed-model", "", "dir with a Model2Vec checkpoint for embed.Load (first-stage dense retrieval)")
	encoderDir := flag.String("encoder-model", "", "dir with a CodeRankEmbed-shaped checkpoint for encoder.Load (MaxSim rerank)")
	query := flag.String("q", "read a file line by line", "search query")
	shortlist := flag.Int("shortlist", 20, "candidates each retriever contributes to the fuse")
	rerankN := flag.Int("rerank", 8, "fused candidates to rerank with MaxSim")
	topN := flag.Int("n", 5, "results to show")
	flag.Parse()

	if *embedDir == "" || *encoderDir == "" {
		fmt.Println(`colbert — end-to-end aikit late-interaction (MaxSim) rerank example.

Needs two local models:
  --embed-model   <dir>   Model2Vec               (e.g. testdata/model)
  --encoder-model <dir>   CodeRankEmbed-shaped     (e.g. testdata/encoder-model)

Without them this just prints guidance; the pipeline code below is the point.`)
		return
	}

	em, err := embed.LoadFromFS(os.DirFS(*embedDir), ".")
	check(err, "load embed model")
	tm, err := encoder.Load(*encoderDir)
	check(err, "load encoder model")

	// 1) CHUNK — same as examples/rag.
	var chunks []chunk.Chunk
	for _, d := range corpus {
		cs, err := chunk.ChunkFile("regex", d.name, []byte(d.src), 60)
		check(err, "chunk "+d.name)
		chunks = append(chunks, cs...)
	}

	// 2) EMBED + index for dense retrieval, and the BM25 lexical index — the
	//    identical first stage examples/rag uses. This is deliberate: the point
	//    of this example is the rerank stage, so the shortlist mechanism it
	//    reranks should be the one already established, not a new one.
	chunkTexts := make([]string, len(chunks))
	for i, c := range chunks {
		chunkTexts[i] = c.Text
	}
	dense := ann.New(em.EncodeBatch(chunkTexts, 0))
	docs := make([][]string, len(chunks))
	for i, t := range chunkTexts {
		docs[i] = bm25.Tokenize(t)
	}
	lexical := bm25.Build(docs)

	// 3) RETRIEVE + FUSE.
	lexHits := lexical.TopK(bm25.Tokenize(*query), *shortlist)
	denHits := dense.Query(em.Encode(*query), *shortlist)
	fused := fuse.RRF(fuse.DefaultK,
		fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc }),
		fuse.Keys(denHits, func(h ann.Hit) int { return h.Index }),
	)

	// 4) RERANK the fused shortlist with MaxSim. Every candidate keeps its own
	//    per-token matrix (EncodeTokens — one forward per doc, same cost as
	//    Encode) instead of being squashed through a joint forward with the
	//    query; late.Index scores them all against the query's token matrix in
	//    parallel and returns the best.
	n := min(*rerankN, len(fused))
	cand := fused[:n]
	qVecs, err := tokenVecs(tm, *query, true)
	check(err, "encode query tokens")
	docVecs := make([][][]float32, n)
	for i, r := range cand {
		docVecs[i], err = tokenVecs(tm, chunks[r.Key].Text, false)
		check(err, "encode doc tokens for "+chunks[r.Key].File)
	}
	rix := late.New(docVecs)
	hits := rix.Query(qVecs, n)

	// 5) Final ranked output.
	fmt.Printf("query: %q\n\n", *query)
	show := min(*topN, len(hits))
	for rank, h := range hits[:show] {
		c := chunks[cand[h.Index].Key]
		fmt.Printf("%d. %.4f  %s:%d-%d\n     %s\n", rank+1, h.Score, c.File, c.StartLine, c.EndLine, firstLine(c.Text))
	}
}

// tokenVecs runs EncodeTokens and reshapes+normalizes the flat [L*D] result
// into late's [][]float32 shape — L2-normalizing each row, since EncodeTokens
// returns raw hidden states (the same reason examples/vision normalizes its
// mean-pooled image vector before indexing: late.MaxSim's dot product IS cosine
// similarity only over unit vectors).
func tokenVecs(m *encoder.Model, text string, isQuery bool) ([][]float32, error) {
	flat, l, err := m.EncodeTokens(text, isQuery)
	if err != nil {
		return nil, err
	}
	d := m.HiddenDim()
	rows := make([][]float32, l)
	for i := range rows {
		rows[i] = embed.L2Normalize(append([]float32(nil), flat[i*d:(i+1)*d]...))
	}
	return rows, nil
}

func firstLine(s string) string {
	for i, r := range s {
		if r == '\n' {
			return s[:i]
		}
	}
	return s
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "colbert: %s: %v\n", what, err)
		os.Exit(1)
	}
}

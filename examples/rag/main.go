// Command rag is the end-to-end aikit retrieval pipeline in one file:
//
//	chunk → embed → (ANN + BM25) → RRF fuse → cross-encoder rerank → top-K
//
// It indexes a tiny in-memory corpus, runs a hybrid (lexical + dense) search,
// fuses the two rankings with reciprocal-rank fusion, and reranks the fused
// shortlist with a BERT cross-encoder for the final order — a joint
// (query, document) forward per candidate, not a cosine comparison of two
// separately-embedded vectors, which is why it typically outranks a
// bi-encoder rerank.
//
// It needs two local models (skipped-clean if absent, so `go build ./...`
// always compiles and `go run` without flags just prints guidance):
//
//	go run ./examples/rag \
//	    --embed-model  testdata/model \
//	    --rerank-model testdata/crossencoder-model \
//	    --q "read a file line by line"
//
// See the repo README (and scripts/README.md's "Fetching testdata/
// crossencoder-model" section) for how to fetch the models.
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
	"github.com/townsendmerino/aikit/topk"
)

// A small, deliberately varied corpus. Each entry is a "file"; the chunker
// splits it into indexable units. The query below should surface the
// file-reading snippet over the unrelated ones.
var corpus = []struct{ name, src string }{
	{"readlines.go", "func readLines(path string) ([]string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\tdefer f.Close()\n\tvar lines []string\n\ts := bufio.NewScanner(f)\n\tfor s.Scan() {\n\t\tlines = append(lines, s.Text())\n\t}\n\treturn lines, s.Err()\n}"},
	{"json.go", "func parseConfig(b []byte) (*Config, error) {\n\tvar c Config\n\tif err := json.Unmarshal(b, &c); err != nil {\n\t\treturn nil, fmt.Errorf(\"parse config: %w\", err)\n\t}\n\treturn &c, nil\n}"},
	{"server.go", "func handler(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode(map[string]string{\"ok\": \"true\"})\n}"},
	{"math.go", "func fib(n int) int {\n\tif n < 2 {\n\t\treturn n\n\t}\n\treturn fib(n-1) + fib(n-2)\n}"},
	{"hash.go", "func sha256File(path string) (string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn \"\", err\n\t}\n\tdefer f.Close()\n\th := sha256.New()\n\tif _, err := io.Copy(h, f); err != nil {\n\t\treturn \"\", err\n\t}\n\treturn hex.EncodeToString(h.Sum(nil)), nil\n}"},
}

func main() {
	embedDir := flag.String("embed-model", "", "dir with a Model2Vec checkpoint for embed.Load")
	rerankDir := flag.String("rerank-model", "", "dir with a BERT cross-encoder checkpoint (e.g. ms-marco-MiniLM-L-6-v2) for encoder.LoadCrossEncoder")
	query := flag.String("q", "read a file line by line", "search query")
	shortlist := flag.Int("shortlist", 20, "candidates each retriever contributes to the fuse")
	rerankN := flag.Int("rerank", 8, "fused candidates to rerank with the encoder")
	flag.Parse()

	if *embedDir == "" || *rerankDir == "" {
		fmt.Println(`rag — end-to-end aikit retrieval example.

Needs two local models:
  --embed-model  <dir>   Model2Vec        (e.g. testdata/model)
  --rerank-model <dir>   BERT cross-encoder (e.g. testdata/crossencoder-model)

Without them this just prints guidance; the pipeline code below is the point.`)
		return
	}

	em, err := embed.LoadFromFS(os.DirFS(*embedDir), ".")
	check(err, "load embed model")
	ce, err := encoder.LoadCrossEncoder(*rerankDir)
	check(err, "load rerank model")

	// 1) CHUNK — split each file into indexable units. One flat slice; its
	//    index is the shared id space for BM25 (Result.Doc) and ANN (Hit.Index).
	var chunks []chunk.Chunk
	for _, d := range corpus {
		cs, err := chunk.ChunkFile("regex", d.name, []byte(d.src), 60)
		check(err, "chunk "+d.name)
		chunks = append(chunks, cs...)
	}

	// 2) EMBED + index for dense (semantic) search. EncodeBatch fans the corpus
	//    out across cores and returns L2-normalized vectors in input order,
	//    which is ann's input contract. Bit-identical to a serial Encode loop —
	//    which is what this used to be, and what every caller wrote before
	//    EncodeBatch existed. Embedding is ~78% of an index run.
	chunkTexts := make([]string, len(chunks))
	for i, c := range chunks {
		chunkTexts[i] = c.Text
	}
	vecs := em.EncodeBatch(chunkTexts, 0) // 0 ⇒ NumCPU
	dense := ann.New(vecs)

	// 2b) Build the BM25 lexical index over the same chunks (same order).
	docs := make([][]string, len(chunks))
	for i, t := range chunkTexts {
		docs[i] = bm25.Tokenize(t)
	}
	lexical := bm25.Build(docs)

	// 3) RETRIEVE both ways, then FUSE the rankings. RRF is rank-based, so it
	//    blends BM25's unbounded scores with cosine [-1,1] without normalizing.
	lexHits := lexical.TopK(bm25.Tokenize(*query), *shortlist)
	denHits := dense.Query(em.Encode(*query), *shortlist)
	fused := fuse.RRF(fuse.DefaultK,
		fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc }),
		fuse.Keys(denHits, func(h ann.Hit) int { return h.Index }),
	)

	// 4) RERANK the fused shortlist with the cross-encoder: ScoreBatch scores
	//    each (query, document) pair JOINTLY in one forward — the trunk attends
	//    across query and document together, not a cosine comparison of two
	//    independently-embedded vectors — and dispatches the candidates
	//    document-parallel. Higher logit = more relevant.
	n := min(*rerankN, len(fused))
	cand := fused[:n]
	texts := make([]string, n)
	for i, r := range cand {
		texts[i] = chunks[r.Key].Text
	}
	scores, err := ce.ScoreBatch(*query, texts, 0)
	check(err, "cross-encoder rerank")

	sel := topk.New[int](n)
	for i, r := range cand {
		sel.Push(r.Key, float64(scores[i]))
	}

	// 5) Final ranked output.
	fmt.Printf("query: %q\n\n", *query)
	for rank, hit := range sel.Result() {
		c := chunks[hit.Item]
		fmt.Printf("%d. %.4f  %s:%d-%d\n     %s\n", rank+1, hit.Score, c.File, c.StartLine, c.EndLine, firstLine(c.Text))
	}
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
		fmt.Fprintf(os.Stderr, "rag: %s: %v\n", what, err)
		os.Exit(1)
	}
}

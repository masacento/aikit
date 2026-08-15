// Command splade is the end-to-end learned-sparse retrieval pipeline in one file:
//
//	chunk → SPLADE expand → sparse index → sparse query → top-K
//
// The README's capability matrix calls this aikit's headline differentiator —
// "the only one of these that ships the whole pipeline" — but until now nothing
// under examples/ actually showed it: sparse and encoder.SPLADE/LoadSPLADE are
// complete and parity-pinned, with no visible demo. This is that demo, standing
// alone the way sparse's own package doc frames it: the third retrieval signal
// alongside dense (ann) and lexical (bm25). See examples/rag for how a sparse
// ranking joins those two in a fused hybrid search — it composes the same way
// dense and lexical do, via fuse.Keys(sparseHits, func(h sparse.Hit) int {
// return h.Index }) — which this example deliberately does NOT build, to keep
// the sparse leg legible on its own.
//
// A SPLADE expansion projects each token's hidden state to vocabulary logits,
// applies log(1+ReLU), and max-pools over the sequence into a sparse
// term-weight vector — most of the vocabulary is exactly zero, a few hundred
// terms carry a learned positive weight. Query and document use the SAME
// expansion (no query/doc asymmetry, unlike CodeRankEmbed's bi-encoder), and
// the whole loop — BERT forward, masked-LM head, inverted-index scoring —
// runs in-process. No Python at query time.
//
// It needs one local model (skipped-clean if absent, so `go build ./...`
// always compiles and `go run` without flags just prints guidance):
//
//	go run ./examples/splade \
//	    --splade-model testdata/splade-model \
//	    --q "read a file line by line"
//
// See scripts/README.md's "Fetching testdata/splade-model" section (the
// checkpoint is naver/splade-cocondenser-ensembledistil) for how to fetch it.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/chunk"
	_ "github.com/townsendmerino/aikit/chunk/regex" // registers the "regex" chunker via init()
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/sparse"
)

// The same small, deliberately varied corpus examples/rag uses — so the two
// examples are directly comparable on the same query, one dense+lexical+
// cross-encoder, one learned-sparse alone.
var corpus = []struct{ name, src string }{
	{"readlines.go", "func readLines(path string) ([]string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\tdefer f.Close()\n\tvar lines []string\n\ts := bufio.NewScanner(f)\n\tfor s.Scan() {\n\t\tlines = append(lines, s.Text())\n\t}\n\treturn lines, s.Err()\n}"},
	{"json.go", "func parseConfig(b []byte) (*Config, error) {\n\tvar c Config\n\tif err := json.Unmarshal(b, &c); err != nil {\n\t\treturn nil, fmt.Errorf(\"parse config: %w\", err)\n\t}\n\treturn &c, nil\n}"},
	{"server.go", "func handler(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode(map[string]string{\"ok\": \"true\"})\n}"},
	{"math.go", "func fib(n int) int {\n\tif n < 2 {\n\t\treturn n\n\t}\n\treturn fib(n-1) + fib(n-2)\n}"},
	{"hash.go", "func sha256File(path string) (string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn \"\", err\n\t}\n\tdefer f.Close()\n\th := sha256.New()\n\tif _, err := io.Copy(h, f); err != nil {\n\t\treturn \"\", err\n\t}\n\treturn hex.EncodeToString(h.Sum(nil)), nil\n}"},
}

func main() {
	spladeDir := flag.String("splade-model", "", "dir with a SPLADE checkpoint (e.g. naver/splade-cocondenser-ensembledistil) for encoder.LoadSPLADE")
	query := flag.String("q", "read a file line by line", "search query")
	topK := flag.Int("k", 5, "results to return")
	flag.Parse()

	if *spladeDir == "" {
		fmt.Println(`splade — end-to-end learned-sparse (SPLADE) retrieval example.

Needs one local model:
  --splade-model <dir>   SPLADE (BertForMaskedLM)   (e.g. testdata/splade-model)

Without it this just prints guidance; the pipeline code below is the point.`)
		return
	}

	s, err := encoder.LoadSPLADE(*spladeDir)
	check(err, "load SPLADE model")

	// 1) CHUNK — split each file into indexable units. One flat slice; its
	//    index is the shared id space sparse.Hit.Index uses.
	var chunks []chunk.Chunk
	for _, d := range corpus {
		cs, err := chunk.ChunkFile("regex", d.name, []byte(d.src), 60)
		check(err, "chunk "+d.name)
		chunks = append(chunks, cs...)
	}

	// 2) EXPAND every chunk to a sparse term-weight vector via SPLADE — BERT
	//    forward + masked-LM head + log(1+ReLU) max-pool over the vocab, no
	//    Python at query time. The SAME Expand call handles queries and
	//    documents (no bi-encoder-style asymmetry).
	dvecs := make([]sparse.SparseVec, len(chunks))
	for i, c := range chunks {
		v, err := s.Expand(c.Text)
		check(err, "expand chunk "+c.File)
		dvecs[i] = v
	}

	// 3) INDEX — an inverted index over the sparse vectors, scored by sparse
	//    dot product at query time (only the query's non-zero terms are walked).
	ix := sparse.New(dvecs)

	// 4) QUERY — expand the query the same way, then retrieve.
	qvec, err := s.Expand(*query)
	check(err, "expand query")
	hits := ix.Query(qvec, *topK)

	// 5) Final ranked output. The expansion's own term count is worth showing:
	//    this is the "most of the vocabulary is zero" sparsity SPLADE's whole
	//    pitch rests on, not just the ranking it produces.
	fmt.Printf("query: %q  (expanded to %d nonzero terms)\n\n", *query, len(qvec.Terms))
	for rank, h := range hits {
		c := chunks[h.Index]
		fmt.Printf("%d. %.4f  %s:%d-%d\n     %s\n", rank+1, h.Score, c.File, c.StartLine, c.EndLine, firstLine(c.Text))
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
		fmt.Fprintf(os.Stderr, "splade: %s: %v\n", what, err)
		os.Exit(1)
	}
}

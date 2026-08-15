# rag — hybrid retrieval + cross-encoder rerank

The canonical end-to-end pipeline: chunk → embed → ANN + BM25 → RRF fuse →
cross-encoder rerank → top-K. See `main.go`'s doc comment for the full shape;
this is the "does it actually work" record.

## Run it

Needs two local models — a Model2Vec checkpoint and a BERT cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`; see `scripts/README.md`'s "Fetching
`testdata/crossencoder-model`" section):

```sh
go run ./examples/rag \
    --embed-model  testdata/model \
    --rerank-model testdata/crossencoder-model \
    --q "read a file line by line"
```

## Real output

```
query: "read a file line by line"

1. -2.2145  readlines.go:10-12
     		lines = append(lines, s.Text())
2. -9.6999  readlines.go:5-7
     	}
3. -9.8056  readlines.go:1-1
     func readLines(path string) ([]string, error) {
4. -11.3048  hash.go:8-9
     	if _, err := io.Copy(h, f); err != nil {
5. -11.3080  hash.go:2-4
     	f, err := os.Open(path)
6. -11.3498  readlines.go:2-4
     	f, err := os.Open(path)
7. -11.4049  readlines.go:8-9
     	s := bufio.NewScanner(f)
8. -11.4325  hash.go:1-1
     func sha256File(path string) (string, error)
```

Scores are the cross-encoder's raw relevance logit (unbounded, higher =
better) — not a probability, so don't read the sign as "irrelevant."

## Compare against the other rerankers

[`examples/colbert`](../colbert) reranks the identical fused shortlist with
ColBERT-style MaxSim instead of the cross-encoder — run both on the same `-q`
to compare final orderings.

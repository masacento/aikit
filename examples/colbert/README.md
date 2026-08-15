# colbert — ColBERT-style late-interaction (MaxSim) rerank

Identical shortlist mechanism to [`examples/rag`](../rag) (same corpus, same
dense+lexical fuse), swapping the final rerank stage: instead of
`encoder.CrossEncoder`'s one joint forward per `(query, doc)` pair, every
candidate keeps its own per-token vectors (`encoder.Model.EncodeTokens`, built
for exactly this) and `late.Index`'s MaxSim lets each query token
independently find its best-matching document token. See `main.go`'s doc
comment for the full shape.

## Run it

Needs two local models — a Model2Vec checkpoint (first-stage dense retrieval)
and a CodeRankEmbed-shaped checkpoint (`encoder.Load`, for `EncodeTokens` —
see `scripts/README.md`'s "Fetching `testdata/encoder-model`" section; a real
contextualizing transformer is required here, not Model2Vec's static
per-token vectors):

```sh
go run ./examples/colbert \
    --embed-model   testdata/model \
    --encoder-model testdata/encoder-model \
    --q "read a file line by line"
```

## Real output

```
query: "read a file line by line"

1. 8.2574  readlines.go:1-1
     func readLines(path string) ([]string, error) {
2. 7.2521  readlines.go:5-7
     	}
3. 6.6745  readlines.go:8-9
     	s := bufio.NewScanner(f)
4. 5.6019  readlines.go:10-12
     		lines = append(lines, s.Text())
5. 4.7608  hash.go:1-1
     func sha256File(path string) (string, error) {
```

Compare against [`examples/rag`](../rag)'s cross-encoder rerank on the same
query — both correctly surface `readlines.go`, but in a different order,
since the two rerankers are scoring the pair fundamentally differently.

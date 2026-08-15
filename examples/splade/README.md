# splade — learned-sparse retrieval, standalone

The third retrieval signal alongside dense (`ann`) and lexical (`bm25`):
chunk → SPLADE expand → sparse index → sparse query → top-K, entirely
in-process — no Python at query time. Deliberately does NOT fuse with dense or
lexical (that's what [`examples/rag`](../rag) shows); see `main.go`'s doc
comment for how it composes into a fused hybrid search when you want that.

## Run it

Needs one local model (`naver/splade-cocondenser-ensembledistil`; see
`scripts/README.md`'s "Fetching `testdata/splade-model`" section):

```sh
go run ./examples/splade \
    --splade-model testdata/splade-model \
    --q "read a file line by line"
```

## Real output

```
query: "read a file line by line"  (expanded to 29 nonzero terms)

1. 8.1637  readlines.go:10-12
     		lines = append(lines, s.Text())
2. 7.7904  readlines.go:1-1
     func readLines(path string) ([]string, error) {
3. 5.3383  readlines.go:5-7
     	}
4. 2.7383  server.go:2-2
     	w.Header().Set("Content-Type", "application/json")
5. 1.4976  json.go:3-3
     	if err := json.Unmarshal(b, &c); err != nil {
```

The "29 nonzero terms" is the point SPLADE's whole pitch rests on — most of
the model's vocabulary scored exactly zero for this query; only a few hundred
learned terms carry weight, which is what makes the inverted index cheap to
walk.

# Fork notes

This fork adds pure-Go named-entity recognition to
`github.com/townsendmerino/aikit`, based on upstream commit `1ca2ba9`. The Go
module path is unchanged.

## Added in this fork

- Experimental `ner` package: GLiNER2 extraction/classification and DistilBERT
  BIO token classification.
- DeBERTa-v2/v3 and DistilBERT encoder support.
- SentencePiece loading and WordPiece byte-offset tracking.
- Japanese, Chinese, and Korean segmentation, ported from Litsea.

See [`docs/ner.md`](docs/ner.md) and [`docs/cjk.md`](docs/cjk.md). Model
checkpoints remain local under the Git-ignored `testdata/` directories.

## Verification

```bash
GOWORK=off go test ./...
GOWORK=off go test -count=1 -v ./ner -run \
  'TestGLiNER2_(forward|parity|classify|promptLayout|tokenizer)$|TestTokenClassifier_parity$'
```

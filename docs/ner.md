# Named-entity recognition

The experimental `ner` package provides two pure-Go paths for extracting spans from
raw text. Both load local safetensors checkpoints and are parity-tested against the
corresponding Python implementations.

| Path | Model | Labels | Long input handling |
|---|---|---|---|
| `GLiNER2` | `fastino/gliner2.5-multi-v1` | Supplied with each request | Word-level cap from the checkpoint; DeBERTa relative positions extend beyond its configured baseline |
| `TokenClassifier` | A DistilBERT `ForTokenClassification` checkpoint | Read from `config.json` | Overlapping windows with first-window-wins deduplication |

All reported offsets are byte offsets into the original Go string, not Unicode code
point indices. A valid result always supports `text[result.Start:result.End]`.

## Checkpoints

Download the parity-pinned checkpoints into the ignored `testdata` directories:

```bash
uvx --from huggingface_hub hf download fastino/gliner2.5-multi-v1 \
  --local-dir testdata/gliner-multi-v2.5

uvx --from huggingface_hub hf download \
  AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs \
  --local-dir testdata/distilbert-secret-masker-v3.3a-rs
```

`LoadGLiNER2` expects `config.json`, `encoder_config/config.json`,
`model.safetensors`, and `tokenizer.json`. `LoadTokenClassifier` expects
`config.json`, `model.safetensors`, and `tokenizer.json` in one directory.

## Zero-shot extraction with GLiNER2

Entity types are prompt text rather than a label schema baked into the checkpoint:

```go
m, err := ner.LoadGLiNER2("testdata/gliner-multi-v2.5")
if err != nil {
    return err
}
defer m.Close()

text := "Barack Obama was born in Honolulu."
entities, err := m.Predict(text, []string{"person", "location"}, ner.Opts{
    Threshold: 0.5,
})
```

`Opts` controls decoding:

- `Threshold`: minimum span probability; zero selects the default `0.5`, and a
  negative value accepts every candidate.
- `Nested`: permits nested spans while still rejecting crossing spans.
- `MultiLabel`: permits the same boundaries to be returned under multiple labels.
  When false, the highest-scoring label wins.
- `CJKSplit`: segments CJK word runs before subword tokenization. It improves span
  granularity but deliberately differs from the reference splitter.

GLiNER2 also exposes classification over request-defined classes:

```go
results, err := m.Classify(
    "The board approved the merger.",
    "sentiment",
    []string{"positive", "negative", "neutral"},
    false,
)
```

The final argument selects multi-label sigmoid scoring when true and single-label
softmax scoring when false.

A runnable command is available under `examples/gliner2`:

```bash
go run ./examples/gliner2 \
  --model testdata/gliner-multi-v2.5 \
  --text "織田信長は日本の武将である" \
  --labels person
```

## BIO token classification

`TokenClassifier` reads `id2label` from the checkpoint configuration. Labels must
use `O`, `B-*`, and `I-*` roles. Mismatched `I-*` transitions start a new entity and
are counted as invalid transitions rather than merged into the preceding type.

```go
m, err := ner.LoadTokenClassifier(
    "testdata/distilbert-secret-masker-v3.3a-rs",
)
if err != nil {
    return err
}
defer m.Close()

text := "password = hunter2!"
entities, err := m.Predict(text, ner.TokenOpts{Tau: 0.99})
```

`TokenOpts.MaxLength` includes `[CLS]` and `[SEP]` and cannot exceed the DistilBERT
position table. `Stride` controls overlap between windows. Their zero values select
the checkpoint limit and a stride of 128. `Tau` filters complete decoded entities by
their mean entity-role probability; zero keeps every decoded entity.

Call `Pieces` to obtain the position-level predictions consumed by BIO decoding and
the number of tolerated invalid transitions.

The secret-masker example can report or replace detected spans:

```bash
echo 'password = hunter2!' | go run ./examples/secretmasker \
  --model testdata/distilbert-secret-masker-v3.3a-rs

go run ./examples/secretmasker \
  --model testdata/distilbert-secret-masker-v3.3a-rs \
  --mask-output clean.txt secrets.env
```

Model inference is probabilistic. Secret masking should complement deterministic
credential scanners and provider-side secret revocation, not replace them.

## CJK segmentation

The `cjk` package embeds Japanese, Chinese, and Korean segmentation models. GLiNER2
uses them only when `Opts.CJKSplit` is enabled. Bare Han text defaults to the Japanese
model; Hangul selects Korean, and kana selects Japanese. If model loading fails, NER
falls back to the reference whitespace splitter.

See [`docs/cjk.md`](cjk.md) for the Litsea port's API, scope, model formats, and
provenance. The embedded model licensing terms are recorded in `cjk/LICENSE`.

## Parity tests

Golden files are committed under `testdata/`; model weights are local and ignored.
The oracle generation details live in `testdata/README.md` and `scripts/oracle/`.

Run the checkpoint-backed gates without test-cache reuse:

```bash
GOWORK=off go test -count=1 -v ./ner -run \
  'TestGLiNER2_(forward|parity|classify|promptLayout|tokenizer)$|TestTokenClassifier_parity$'
```

The GLiNER2 gate separately checks tokenizer output, prompt layout, pooled states,
boundary logits, candidate pools, decoded entities, and classification. The token
classifier gate checks WordPiece offsets, overflow windowing, DistilBERT output, BIO
decoding, thresholds, and masked surfaces.

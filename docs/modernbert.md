# ModernBERT

The `encoder` package runs ModernBERT embedding models and cross-encoder
rerankers in pure Go. It supports the architecture's pre-norm blocks, RoPE,
alternating local/global attention, and GeGLU feed-forward layers.

| Path | Pinned model | Tokenizer | Output |
|---|---|---|---|
| `ModernBERT` | `hotchpotch/bekko-embedding-v1-a8m` | Metaspace BPE | 384-dimensional normalized embedding |
| `ModernBERT` | `cl-nagoya/ruri-v3-30m` | Unigram | 256-dimensional normalized embedding |
| `ModernBERTQ8` | Either embedding checkpoint | Same as the source model | Approximate normalized embedding with int8 projections |
| `ModernBERTCrossEncoder` | `cross-encoder/ettin-reranker-17m-v1` | Byte-level BPE | Relevance logit for a query/document pair |

## Checkpoints

Download the parity-pinned checkpoints into the ignored `testdata` directories:

```bash
uvx --from huggingface_hub hf download hotchpotch/bekko-embedding-v1-a8m \
  --local-dir testdata/bekko-embedding-v1-a8m

uvx --from huggingface_hub hf download cl-nagoya/ruri-v3-30m \
  --local-dir testdata/ruri-v3-30m

uvx --from huggingface_hub hf download cross-encoder/ettin-reranker-17m-v1 \
  --local-dir testdata/ettin-reranker-17m
```

Embedding models require `config.json`, `model.safetensors`, and
`tokenizer.json`. `sentence_bert_config.json` supplies an optional sequence-length
cap, while `1_Pooling/config.json` selects CLS or mean pooling.

The reranker is a sentence-transformers module chain. In addition to the trunk,
it requires the files under `1_Pooling`, `2_Dense`, `3_LayerNorm`, and
`4_Dense`; the final three modules form its classification head. The pinned
checkpoint also includes `modules.json`, which the loader validates when present.

## Embeddings

`Encode` tokenizes text, adds the model's special tokens, right-truncates it,
pools the final hidden states, and L2-normalizes the result:

```go
m, err := encoder.LoadModernBERT("testdata/bekko-embedding-v1-a8m")
if err != nil {
	return err
}
defer m.Close()

vector, err := m.Encode("Text to embed")
```

The effective sequence limit is the smaller of `max_position_embeddings` and
the positive `max_seq_length` in `sentence_bert_config.json`. Use `Embed(ids)`
when token IDs, including special tokens, are already available. `HiddenDim`
reports the output width.

## Int8 embeddings

`LoadModernBERTQ8` keeps LayerNorm and embedding-table weights in float32 while
storing the four large projection matrices per layer as per-row int8 weights:

```go
m, err := encoder.LoadModernBERTQ8("testdata/ruri-v3-30m")
if err != nil {
	return err
}
defer m.Close()

vector, err := m.Encode("Text to embed")
```

Float, float16, and bfloat16 checkpoints are quantized during loading. A
pre-quantized checkpoint may instead store I8 projection tensors with companion
`<tensor>.scale` tensors. The public encoding API is otherwise the same as
`ModernBERT`.

## Cross-encoder reranking

The reranker scores each query/document pair jointly. Higher values indicate
greater relevance:

```go
r, err := encoder.LoadModernBERTCrossEncoder(
	"testdata/ettin-reranker-17m",
)
if err != nil {
	return err
}
defer r.Close()

score, err := r.Score("What is ModernBERT?", "ModernBERT is an encoder model.")
scores, err := r.ScoreBatch("What is ModernBERT?", documents, 0)
```

`Score` returns label zero. `ScoreAll` returns every classification output, and
`Labels` reports their count. `ScoreBatch` preserves document order;
`concurrency <= 0` uses the available CPUs.

Pairs are framed as `[CLS] query [SEP] document [SEP]` without token-type IDs.
Long pairs use longest-first truncation, removing tokens from the longer side
until the model limit is met.

## Runnable example

The example loads bekko, ruri, and Ettin, then embeds and reranks one input:

```bash
go run ./examples/modernbert -text "ModernBERT uses rotary positions."
```

If a checkpoint directory is missing, the command prints the corresponding
download command.

## Parity tests

Golden outputs are committed under `testdata`; model weights are local and
ignored. Run the tokenizer, forward, quantization, reranking, and example gates
without test-cache reuse:

```bash
GOWORK=off go test -count=1 -v ./embed ./encoder ./examples/modernbert -run \
  'Test(SPBPE|BPE_ettin|ModernBERT|ModernBert|RuriParity|BekkoParity|EttinRerankerParity)'
```

The gates compare tokenizer IDs, hidden states, normalized embeddings, banded
and dense attention, float and Q8 output, reranker logits, pair truncation, and
ranking against the pinned references. Asset-backed tests skip when their local
checkpoint is absent.

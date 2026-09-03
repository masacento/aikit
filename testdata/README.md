# testdata

Golden fixtures for the Go test suite. The Python reference scripts that
produce them live under [`../scripts/`](../scripts/); the Go tests only
read these files, never write them.

## `golden.json`

Produced by `scripts/oracle/pin_inference.py`. 18 hand-picked cases. For each:

- input text
- WordPiece token strings and IDs (from HF tokenizer reference)
- per-token weights (from the `weights` tensor)
- ground-truth output vector (from `StaticModel.encode()`)
- three candidate pooling-recipe outputs (for debugging)

The empty-string and all-`[UNK]` rows have `ground_truth: null` and a
`degenerate_ground_truth` flag — the Go golden test asserts the
zero-vector contract for those directly rather than via cosine.

To regenerate (from repo root, with `.venv/` bootstrapped — `model2vec`,
`safetensors`, `tokenizers`, `huggingface_hub`, `numpy`):

```bash
.venv/bin/python scripts/oracle/pin_inference.py
cp ken_golden.json testdata/golden.json
```

(The script's own output filename is a pre-extraction leftover — it still
writes `ken_golden.json`, not `aikit_golden.json`; the `cp` step is not
optional.)

## `gliner_tokenizer_golden.json`

Produced by `scripts/oracle/pin_gliner_tokenizer.py` from `gliner-multi-v2.1/spm.model`.
The id-exact oracle for the raw-`spm.model` reader (`embed/tokenize_spm.go`), which
exists because the GLiNER / mDeBERTa-v3 checkpoints ship no `tokenizer.json`. 72
cases (whitespace handling, multilingual, NFKC folding, byte-fallback, split-digits,
piece-type discipline, added tokens) plus 8 word-level cases for `EncodeWords`.

The reference is `sentencepiece` itself — no torch or transformers needed:

```bash
.venv/bin/pip install sentencepiece
.venv/bin/python scripts/oracle/pin_gliner_tokenizer.py
go test ./embed -run TestSPMTokenizer
```

## `deberta_golden.json`

Produced by `scripts/oracle/pin_deberta.py` from `mdeberta-v3-base/`. The **per-layer**
forward-parity oracle for `encoder/deberta.go` — the embedding output plus all 12
layer outputs, for 9 cases including 262/482/512-token sequences (below ~128 tokens
every relative offset stays in DeBERTa's linear bucket band, so a short-only corpus
would leave `make_log_bucket_position` untested).

Each layer stores an evenly spaced `[16 rows x 64 dims]` grid plus `sum`/`abs_sum`/
`min`/`max` over the **full** tensor. Full tensors would be 278 MB of JSON, which is
not a fixture; the grid localizes and the reductions make the localization complete.

```bash
uvx --from huggingface_hub hf download microsoft/mdeberta-v3-base \
    --local-dir testdata/mdeberta-v3-base \
    --include config.json spm.model tokenizer_config.json pytorch_model.bin
# the upstream ships pytorch_model.bin and no safetensors; convert once
uv run --with torch --with safetensors python -c "import torch; \
    from safetensors.torch import save_file; \
    sd = torch.load('testdata/mdeberta-v3-base/pytorch_model.bin', map_location='cpu', weights_only=True); \
    save_file({k: v.contiguous().to(torch.float32) for k, v in sd.items() if isinstance(v, torch.Tensor)}, \
    'testdata/mdeberta-v3-base/model.safetensors')"
.venv/bin/python scripts/oracle/pin_deberta.py
go test ./encoder -run TestDeBERTa
```

## `gliner2_golden.json`

Produced by `scripts/oracle/pin_gliner2.py` from `gliner-multi-v2.5/`
(fastino/gliner2.5-multi-v1). Forward + decode oracle for the GLiNER2 boundary path
(`ner.LoadGLiNER2`): the word split, per-word tokenization against HF's fast
tokenizer, the boundary/classification scores, and the decoded entity and
classification sets — each stage separate so a red gate localizes.

```bash
uvx --from huggingface_hub hf download fastino/gliner2.5-multi-v1 \
    --local-dir testdata/gliner-multi-v2.5
.venv/bin/python scripts/oracle/pin_gliner2.py
go test ./ner
```

## `tokenclassification_golden.json`

Produced by `scripts/oracle/pin_tokenclassification.py` from
`AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs`. The forward + decode oracle
for the `ForTokenClassification` path (`ner.TokenClassifier`, the DistilBERT
trunk from `encoder`): per case, the argmax span set, the tau=0.99 thresholded
set, the masked text, and the invalid-BIO-transition count; per piece
(`start`/`end`/`label`/`p` — character offsets in the PYTHON convention, the Go
tests convert) for the first three cases, to separate a forward bug from a
decode bug. Case 0 also carries a `trunk` block — specials-wrapped ids and
`last_hidden_state` stats — which is `encoder.LoadDistilBERT`'s own oracle
(`TestDistilBERT_trunkGolden`), so a red `ner` gate can be bisected.

The reference logic is the model repo's `span_infer.py`: manual overflow windows
(body = max_length−2, stride 128, first-window-wins dedup), lenient BIO decode,
entity score = mean P(B)+P(I). One fixture span sits at 0.986 against the 0.99
threshold on purpose — tau-set membership is where float32 drift would show.

```bash
uvx --from huggingface_hub hf download AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs \
    --local-dir testdata/distilbert-secret-masker-v3.3a-rs
uv run python scripts/oracle/pin_tokenclassification.py
go test ./ner ./encoder ./examples/secretmasker
```

## `mdeberta-v3-base/`, `gliner-multi-v2.5/`, `distilbert-secret-masker-v3.3a-rs/` (gitignored, per-machine)

Checkpoints the `ner` / `encoder` tests load directly; every such test `t.Skip()`s
when its directory is absent.

- `microsoft/mdeberta-v3-base` — upstream ships `pytorch_model.bin` and no
  safetensors; aikit's loaders are mmap-over-safetensors only, deliberately, so
  convert once after download (command in the deberta section above).
- `fastino/gliner2.5-multi-v1` — GLiNER2 boundary checkpoint (self-contained).
- `AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs` — DistilBERT
  token-classification head (see the tokenclassification section above).

## `parity.jsonl` (gitignored)

Produced by `scripts/oracle/parity_dump.py`. The 100k-input corpus-scale
tokenizer parity fixture. Run the `parity`-tagged Go test against it:

```bash
.venv/bin/python scripts/oracle/parity_dump.py
go test -tags=parity ./embed/ -run TestParity -v
```

## `model/` (gitignored, per-machine)

A local snapshot of `minishlab/potion-code-16M` for tests that exercise
the full inference pipeline (the golden cosine assertion, and the parity
harness). Tests read `testdata/model/` directly and `t.Skip()` when it's
absent — CI without HF access stays green.

```bash
uvx --from huggingface_hub hf download minishlab/potion-code-16M \
    tokenizer.json config.json model.safetensors \
    --local-dir testdata/model
```

aikit itself has no model-fetching CLI of its own; a symlink into an
existing local cache (e.g. `~/.cache/huggingface/...`) works equally well
as long as `testdata/model/` resolves to the three files above.

## `repo/`

The polyglot smoke fixture (tiny files in Python/Go/TypeScript/Java/Rust
plus a markdown stub). Used by chunker tests and the search/MCP
integration tests.

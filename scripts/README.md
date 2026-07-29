# scripts/ — the parity oracle

The `pin_*.py` generators are aikit's parity oracle. Each loads a real model (via
PyTorch / sentence-transformers / transformers / model2vec) and dumps a golden
fixture into `testdata/`, which the Go tests then assert against bit-for-bit-ish.
This is design rule 3 (numerics are parity-pinned): every model-touching Go path —
`embed`, `encoder`, the `linalg` quant kernels — is checked against this reference,
so a port bug surfaces as a failing test, not a silent accuracy regression.

## Toolchain setup (roadmap §2.1)

CPU-only, runnable on a Mac. The venv is gitignored; `requirements.txt` pins the
versions that produced the committed goldens.

```sh
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
```

Verify it works (loads + embeds, ~90 MB download on first run):

```sh
.venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; \
  print(S('sentence-transformers/all-MiniLM-L6-v2').encode('hi').shape)"   # (384,)
```

## Regenerating a golden

Each script writes a fixed `testdata/*.json` path; run from the repo root, e.g.:

```sh
.venv/bin/python scripts/pin_encoder.py     # → testdata/encoder_golden.json (CodeRankEmbed)
.venv/bin/python scripts/pin_inference.py   # → the Model2Vec embed golden
```

Models are fetched from the Hugging Face Hub on first run. GGUF dequant scripts
(`pin_iq_dequant.py`) additionally need `pip install gguf`.

## Fetching `testdata/splade-model`

The SPLADE tests and benchmarks (`encoder/splade_test.go`,
`encoder/splade_bench_test.go`) skip without this checkpoint. It needs one extra
step the other fixtures don't: `naver/splade-cocondenser-ensembledistil`
publishes **only `pytorch_model.bin`**, while `LoadBERT` reads
`model.safetensors`. Run from the repo root:

```sh
.venv/bin/python - <<'PY'
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

d = snapshot_download('naver/splade-cocondenser-ensembledistil',
                      allow_patterns=['config.json', 'tokenizer.json',
                                      'tokenizer_config.json',
                                      'special_tokens_map.json', 'vocab.txt',
                                      'pytorch_model.bin'],
                      local_dir='testdata/splade-model')
sd = torch.load(d + '/pytorch_model.bin', map_location='cpu', weights_only=True)
# clone() breaks the decoder.weight <-> word_embeddings.weight tying that
# safetensors refuses to store; position_ids is a buffer, not a weight.
save_file({k: v.contiguous().clone() for k, v in sd.items()
           if v.dtype.is_floating_point and not k.endswith('position_ids')},
          'testdata/splade-model/model.safetensors', metadata={'format': 'pt'})
PY
rm testdata/splade-model/pytorch_model.bin   # ~438 MB, no longer needed
```

Untying the decoder makes `model.safetensors` (~532 MB) larger than the `.bin`;
`testdata/splade-model` is git-ignored. Verify with
`go test ./encoder/ -run SPLADE_parity -v` — it should report cosine 1.000000
against `testdata/splade_golden.json`.

## Fetching `testdata/arctic2-m`

The GTE tests/benchmarks (`encoder/gte_test.go`, `encoder/gte_bench_test.go`,
`encoder/rope_cache_test.go`) need `Snowflake/snowflake-arctic-embed-m-v2.0`.
This one ships safetensors, so it is a plain download (~1.2 GB):

```sh
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Snowflake/snowflake-arctic-embed-m-v2.0',
    allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'special_tokens_map.json', 'model.safetensors',
                    'modules.json', '1_Pooling/config.json',
                    'config_sentence_transformers.json'],
    local_dir='testdata/arctic2-m')"
```

Verify with `go test ./encoder/ -run TestGTE_parity -v` — worst emb cosine
1.000000 against `testdata/gte_golden.json`.

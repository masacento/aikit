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
huggingface-cli download minishlab/potion-code-16M \
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

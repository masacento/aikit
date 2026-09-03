#!/usr/bin/env python3
"""pin_deberta.py — PER-LAYER forward-parity golden for the DeBERTa-v2/v3 encoder
(microsoft/mdeberta-v3-base), the oracle for aikit's encoder/deberta.go.

Per-layer, not last-layer-only, and that is the whole point. This architecture has
four independent ways to be silently wrong — the log-bucket arithmetic, the p2c
gather index, the 1/sqrt(3d) scale, and which tensor encoder.LayerNorm normalizes —
and all four present identically at the output: no crash, no NaN, just numbers that
are a bit off. A whole-model fixture says THAT you diverged; this one says WHERE,
which is the difference between a one-day bug and a three-day one.

Dumps, per case, for hidden_states[0..12] (the embedding output followed by each of
the 12 layer outputs — HF's output_hidden_states=True):

  - sample : an evenly spaced [rows x dims] grid of the hidden state. The full
             tensors are 278 MB of JSON across this corpus, which is not a fixture;
             a grid is. Every failure mode this gate targets perturbs the attention
             SCORES, so it moves essentially every element — a grid catches it.
  - sum / abs_sum / min / max : reductions over the FULL tensor, so an error confined
             to dimensions the grid skips still shows up. The grid localizes; these
             make the localization complete.

Run from repo root (after scripts/fetch_mdeberta.sh):
    .venv/bin/python scripts/pin_deberta.py
"""
import json
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "mdeberta-v3-base"
OUT = REPO_ROOT / "testdata" / "deberta_golden.json"

CASES = [
    # Short sequences keep every offset inside the LINEAR bucket band (|i-j| <= 128),
    # so they exercise the gather plumbing but not the log branch.
    "Hello world.",
    "a",
    "The quick brown fox jumps over the lazy dog.",
    "日本語の文をエンコードします。",
    "Barack Obama was born in Honolulu, Hawaii.",
    "Angela Merkel besuchte Paris im Juni 2019.",
    # These are the ones that matter: past 128 tokens the log-bucket branch and its
    # ceil/sign engage, and past 256 the clamp does. A corpus of short strings would
    # leave make_log_bucket_position almost entirely untested.
    " ".join(f"token{i}" for i in range(160)),
    " ".join(f"word{i}" for i in range(300)),
    "The history of natural language processing began in the 1950s. " * 20,
]


MAX_ROWS = 16
MAX_DIMS = 64


def sample_indices(n: int, k: int) -> list[int]:
    """Evenly spaced indices over [0, n), always including the first and last.

    The endpoints are not cosmetic: row 0 is [CLS] and row L-1 is [SEP], and the
    largest relative offsets in the sequence are exactly between them — which is the
    region the log-bucket branch and the clamp govern.
    """
    if n <= k:
        return list(range(n))
    step = (n - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def main() -> None:
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit(f"missing {MODEL_DIR} — run scripts/fetch_mdeberta.sh")

    # Tokenize with sentencepiece directly rather than AutoTokenizer. With
    # split_by_punct=False, DebertaV2Tokenizer._tokenize IS spm.encode, and the fast
    # conversion would pull in a protobuf dependency for no gain. [CLS]=1, [SEP]=2.
    sp = spm.SentencePieceProcessor()
    sp.Load(str(MODEL_DIR / "spm.model"))

    model = AutoModel.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.eval()

    cases = []
    with torch.no_grad():
        for text in CASES:
            body = sp.EncodeAsIds(text)[: 512 - 2]
            ids = torch.tensor([[1] + body + [2]], dtype=torch.long)
            mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = out.hidden_states  # tuple: embeddings output + one per layer
            L = int(ids.shape[1])
            rows = sample_indices(L, MAX_ROWS)
            dims = sample_indices(model.config.hidden_size, MAX_DIMS)
            layers = []
            for h in hs:
                t = h[0].to(torch.float32)
                layers.append({
                    "sample": t[rows][:, dims].flatten().tolist(),
                    "sum": float(t.sum()),
                    "abs_sum": float(t.abs().sum()),
                    "min": float(t.min()),
                    "max": float(t.max()),
                })
            cases.append({
                "text": text if len(text) < 120 else text[:117] + "...",
                "input_ids": ids[0].tolist(),
                "L": L,
                "rows": rows,
                "dims": dims,
                "layers": layers,
            })
            print(f"  L={L:4d}  layers={len(layers)}  {text[:48]!r}")

    OUT.write_text(json.dumps({
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "cases": cases,
    }))
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(cases)} cases, {mb:.1f} MB")


if __name__ == "__main__":
    main()

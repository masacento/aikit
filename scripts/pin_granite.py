#!/usr/bin/env python3
"""pin_granite.py — golden fixture for ibm-granite/granite-embedding-125m-english,
the parity oracle for aikit's RoBERTa forward + byte-level BPE tokenizer path.
Granite-english is a RoBERTa (learned-absolute positions with the pad+1 offset,
GELU FFN, CLS pooling) whose tokenizer is GPT-2/RoBERTa byte-level BPE — the new
tokenizer backend this certifies.

Dumps, per curated case, from the real reference (loaded fp32):
  - input_ids : the byte-level BPE ids the model ate (<s> … </s>).
  - hidden    : last_hidden_state [L, 768] flattened row-major, with L.
  - embedding : the CLS-pooled, L2-normalized sentence embedding [768].

Run from repo root:
    .venv/bin/python scripts/pin_granite.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "granite-en"
OUT = REPO_ROOT / "testdata" / "granite_golden.json"

# ASCII + light punctuation/whitespace cases — byte-level BPE's tricky spots are
# leading/interior/trailing spaces, newlines, and tabs (the pre-tokenizer's
# whitespace give-back rule), so exercise them explicitly.
CASES = [
    "how do i parse json",
    "def add(a, b):\n    return a + b",
    "",
    "hello world",
    "x",
    "compute the sha256 hash of a file",
    "a  double  space",
    "trailing space ",
    " leading space",
    "tab\tseparated\tvalues",
]


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    sys.stderr.write(f"[pin_granite] loading {MODEL_DIR} (CPU, fp32) ...\n")
    m = SentenceTransformer(
        str(MODEL_DIR), device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    cfg = m[0].auto_model.config
    out = {
        "model": "ibm-granite/granite-embedding-125m-english",
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "pooling": "cls",
        "cases": [],
    }
    for text in CASES:
        ids = m.tokenize([text])["input_ids"][0].tolist()
        hid = m.encode(text, output_value="token_embeddings", convert_to_numpy=True)
        emb = m.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        out["cases"].append({
            "text": text,
            "input_ids": [int(i) for i in ids],
            "L": int(hid.shape[0]),
            "hidden": [round(float(x), 6) for x in hid.reshape(-1).tolist()],
            "embedding": [round(float(x), 6) for x in emb.tolist()],
        })

    OUT.write_text(json.dumps(out))
    sys.stderr.write(
        f"[pin_granite] wrote {OUT} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"pooling=cls, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

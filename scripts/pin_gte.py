#!/usr/bin/env python3
"""pin_gte.py — golden fixture for the GTE encoder (Snowflake/snowflake-arctic-embed-m-v2.0,
based on Alibaba gte-multilingual-base), the parity oracle for aikit's GTE forward
(encoder/gte.go). GTE is a post-norm BERT-family encoder with RoPE positions, a
packed qkv projection, and a gated-GELU (GeGLU) MLP — none of which the plain-BERT
or nomic-SwiGLU paths exercise.

Dumps, per curated case, from the real reference (loaded as fp32):
  - input_ids : the SentencePiece ids the model ate (<s> … </s>).
  - hidden    : last_hidden_state [L, 768] flattened row-major, with L.
  - embedding : the CLS-pooled, L2-normalized sentence embedding [768].

Run from repo root:
    .venv/bin/python scripts/pin_gte.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "arctic2-m"
OUT = REPO_ROOT / "testdata" / "gte_golden.json"

CASES = [
    "how do i parse json",
    "def add(a, b):\n    return a + b",
    "",
    "hello world",
    "x",
    "compute the sha256 hash of a file",
    "Bonjour le monde, ça va?",
    "Здравствуй, мир",
    "識別子を検索する",
    "quantization reduces model memory",
]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    sys.stderr.write(f"[pin_gte] loading {MODEL_DIR} (CPU, fp32) ...\n")
    # Drive the model directly rather than via SentenceTransformer: the reference
    # defaults to xformers memory-efficient / unpadded attention (extra dependency,
    # and its auto-computed position_ids buffer is uninitialized on the padded
    # path). Disable both fancy paths and pass position_ids explicitly — this is a
    # plain dense bidirectional softmax forward, numerically identical to the
    # xformers path and exactly what aikit's gte.go computes.
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    # low_cpu_mem_usage=False is REQUIRED: the RotaryEmbedding's cos/sin caches and
    # the position_ids are non-persistent buffers (not in the state dict). The
    # default low_cpu_mem_usage=True leaves them on the meta device (zeros), which
    # silently produces an all-NaN forward. Loading eagerly runs __init__ so the
    # rope cache is real.
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, dtype=torch.float32,
        use_memory_efficient_attention=False, unpad_inputs=False,
        low_cpu_mem_usage=False,
    ).eval()
    cfg = model.config
    # Reconstruct the rotary embedding cache. The RotaryEmbedding's inv_freq and
    # cos/sin caches are NON-PERSISTENT buffers (not in the state dict), and this
    # custom-code model leaves them uninitialized — inv_freq comes back all-zeros,
    # so the model's own RoPE is a silent no-op (cos≡1/garbage), which would make
    # the golden a wrong, position-blind forward. Recompute inv_freq from the
    # config (1/base^(arange(0,dim,2)/dim)) and rebuild the cache, giving the RoPE
    # the model is documented to use (rope_theta = 160000).
    re = model.embeddings.rotary_emb
    re.inv_freq = 1.0 / (re.base ** (torch.arange(0, re.dim, 2, dtype=torch.float32) / re.dim))
    re._set_cos_sin_cache(seq_len=cfg.max_position_embeddings, device="cpu", dtype=torch.float32)
    assert not bool((re.cos_cached == 0).all()), "rope cache still dead"
    out = {
        "model": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "intermediate": cfg.intermediate_size,
        "rope_theta": cfg.rope_theta,
        "pooling": "cls",
        "cases": [],
    }
    for text in CASES:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.max_position_embeddings)
        L = enc["input_ids"].shape[1]
        pos = torch.arange(L).unsqueeze(0)
        with torch.no_grad():
            hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       position_ids=pos).last_hidden_state[0]  # [L, hidden]
        emb = F.normalize(hs[0:1], p=2, dim=1)[0]  # CLS + L2
        out["cases"].append({
            "text": text,
            "input_ids": [int(i) for i in enc["input_ids"][0].tolist()],
            "L": int(L),
            "hidden": [round(float(x), 6) for x in hs.reshape(-1).tolist()],
            "embedding": [round(float(x), 6) for x in emb.tolist()],
        })

    OUT.write_text(json.dumps(out))
    sys.stderr.write(
        f"[pin_gte] wrote {OUT} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"pooling=cls, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

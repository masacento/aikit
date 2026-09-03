#!/usr/bin/env python3
"""pin_bekko.py — golden fixture for the ModernBERT encoder
(hotchpotch/bekko-embedding-v1-a8m, a ModernBERT-base fine-tune), the parity
oracle for aikit's ModernBERT forward (encoder/modernbert.go). ModernBERT is a
pre-norm, bias-free BERT-family encoder with sans_pos RoPE, alternating
local (bidirectional sliding window) / global attention, and a GeGLU MLP —
none of which the plain-BERT, GTE or nomic paths exercise together.

Dumps, per curated case, from the real reference (loaded as fp32, eager attn):
  - input_ids : the ids the model ate (<bos> … <eos>).
  - hidden    : last_hidden_state [L, 384] flattened row-major (AFTER final_norm),
                with L — pins every layer, so a local-window mask bug shows up.
  - embedding : the mean-pooled (over ALL tokens incl. specials), L2-normalized
                sentence embedding [384] — what sentence-transformers returns.

Run from repo root:
    .venv/bin/python scripts/pin_bekko.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "bekko-embedding-v1-a8m"
OUT = REPO_ROOT / "testdata" / "bekko_golden.json"

# Long inputs so the local layers' sliding window actually masks positions — short
# cases never trip |i-j| > S and would pass with a global-only forward. LONG_EN
# clears the 64-token half-window (~180 tokens) but stays under the hidden-dump
# ceiling so its full last_hidden_state is pinned; LONG_JA is longer and pins only
# the end-to-end embedding (keeping the committed golden compact).
LONG_EN = "The quick brown fox jumps over the lazy dog. " * 18
LONG_JA = "識別子を検索して埋め込みを生成する処理は、量子化によってメモリを削減できる。" * 20

# Full last_hidden_state is dumped only up to this length, bounding the committed
# golden's size (the two long cases dominate it); every case still dumps its
# embedding, so long-input parity is checked end-to-end regardless.
HIDDEN_DUMP_MAX_L = 256

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
    "量子化はモデルのメモリを削減する",
    LONG_EN,
    LONG_JA,
]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    sys.stderr.write(f"[pin_bekko] loading {MODEL_DIR} (CPU, fp32, eager) ...\n")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    # eager attention materializes the additive bidirectional / sliding-window mask
    # and runs scores+mask+softmax explicitly — exactly what modernbert.go computes.
    # fp32 widens the on-disk BF16 weights (bf16 ⊂ f32, exact), the same starting
    # point as aikit's TensorF32. low_cpu_mem_usage=False runs __init__ so the RoPE
    # buffers are real, not meta-device zeros.
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=False,
    ).eval()
    cfg = model.config
    # transformers 5.x folds the rope thetas into a rope_parameters dict rather
    # than exposing global_rope_theta / local_rope_theta as attributes, so read
    # the architecture metadata from the raw config.json (the source of truth).
    raw_cfg = json.loads((MODEL_DIR / "config.json").read_text())
    out = {
        "model": "hotchpotch/bekko-embedding-v1-a8m",
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "intermediate": cfg.intermediate_size,
        "global_rope_theta": raw_cfg.get("global_rope_theta"),
        "local_rope_theta": raw_cfg.get("local_rope_theta"),
        "global_attn_every_n_layers": raw_cfg.get("global_attn_every_n_layers"),
        "local_attention": raw_cfg.get("local_attention"),
        "sliding_window": getattr(cfg, "sliding_window", raw_cfg.get("local_attention", 0) // 2),
        "layer_types": list(getattr(cfg, "layer_types", [])),
        "pooling": "mean",
        "cases": [],
    }
    for text in CASES:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.max_position_embeddings)
        L = enc["input_ids"].shape[1]
        with torch.no_grad():
            hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[0]  # [L, hidden]
        # sentence-transformers mean pooling: average over all attended tokens
        # (incl. <bos>/<eos>), then L2-normalize.
        emb = F.normalize(hs.mean(dim=0, keepdim=True), p=2, dim=1)[0]
        case = {
            "text": text,
            "input_ids": [int(i) for i in enc["input_ids"][0].tolist()],
            "L": int(L),
            "embedding": [round(float(x), 6) for x in emb.tolist()],
        }
        if L <= HIDDEN_DUMP_MAX_L:
            case["hidden"] = [round(float(x), 6) for x in hs.reshape(-1).tolist()]
        out["cases"].append(case)
        sys.stderr.write(
            f"[pin_bekko]   case L={L}{' (+hidden)' if L <= HIDDEN_DUMP_MAX_L else ''}: {text[:40]!r}\n"
        )

    OUT.write_text(json.dumps(out))
    sys.stderr.write(
        f"[pin_bekko] wrote {OUT} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"pooling=mean, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

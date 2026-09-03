#!/usr/bin/env python3
"""pin_ettin.py — ModernBERT trunk parity golden for the Ettin reranker's backbone
(cross-encoder/ettin-reranker-17m-v1), the oracle for encoder/modernbert.go's forward
on an OLMo-vocab, transformers-5.x-exported checkpoint.

Deliberately trunk-only: the classification head is pinned separately by
pin_ettin_reranker.py. Splitting them isolates the RoPE-theta fix (the flat
global_rope_theta / local_rope_theta keys are gone from this config, replaced by a
nested rope_parameters block that a 4.x loader ignores) from head-loading bugs.

The config is built EXPLICITLY here rather than via from_pretrained, for two reasons:
  1. the installed transformers is 4.x and cannot read rope_parameters at all, and
  2. it makes rope_theta=160000 an independent statement of what the artifact says,
     rather than a second copy of the parsing this golden is meant to check.

Cases include one pair longer than 384 tokens, which is the threshold above which
aikit takes its banded sliding-window attention path — the short rerank pairs the
model is actually for never reach it, so it would otherwise go uncovered.

Run from repo root:
    .venv/bin/python scripts/pin_ettin.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = REPO_ROOT / "testdata" / "ettin-reranker-17m"
OUT = REPO_ROOT / "testdata" / "ettin_golden.json"

PAIRS = [
    ("what is the capital of France?", "Paris is the capital and largest city of France."),
    ("quantization", "int8 quantization reduces model memory footprint at a small accuracy cost."),
    ("日本語", "これは日本語のテキストです。"),
    ("", "empty query"),
    # > 384 tokens after framing, so the Go side takes the banded local-attention path
    ("long document retrieval", "The quick brown fox jumps over the lazy dog. " * 40),
]


def main() -> int:
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
    from transformers.models.modernbert.modeling_modernbert import ModernBertModel

    raw = json.loads((MODEL / "config.json").read_text())
    rope = raw["rope_parameters"]
    cfg = ModernBertConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        intermediate_size=raw["intermediate_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        norm_eps=raw["norm_eps"],
        norm_bias=raw["norm_bias"],
        attention_bias=raw["attention_bias"],
        mlp_bias=raw["mlp_bias"],
        hidden_activation=raw["hidden_activation"],
        local_attention=raw["local_attention"],
        global_attn_every_n_layers=raw["global_attn_every_n_layers"],
        global_rope_theta=rope["full_attention"]["rope_theta"],
        local_rope_theta=rope["sliding_attention"]["rope_theta"],
        cls_token_id=raw["cls_token_id"],
        sep_token_id=raw["sep_token_id"],
        pad_token_id=raw["pad_token_id"],
        # eager, not sdpa/flash: the reference must be the plain math, not a fused
        # kernel with its own accumulation order.
        _attn_implementation="eager",
    )
    model = ModernBertModel(cfg)
    state = load_file(MODEL / "model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # ModernBertModel wraps the encoder; tolerate only the head-side absences.
    if unexpected:
        sys.stderr.write(f"[pin_ettin] unexpected tensors: {unexpected}\n")
        return 1
    if missing:
        sys.stderr.write(f"[pin_ettin] MISSING tensors: {missing}\n")
        return 1
    model.eval()

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    tok.no_truncation()

    out = {
        "model": "cross-encoder/ettin-reranker-17m-v1",
        "hidden": raw["hidden_size"],
        "rope_theta": rope["full_attention"]["rope_theta"],
        "cases": [],
    }
    for query, doc in PAIRS:
        ids = tok.encode(query, doc).ids  # [CLS] q [SEP] d [SEP]
        t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            h = model(input_ids=t, attention_mask=torch.ones_like(t)).last_hidden_state[0]
        out["cases"].append(
            {
                "query": query,
                "doc": doc,
                "input_ids": ids,
                # The full [L, D] state, not just row 0: a RoPE error at one position
                # class can leave [CLS] almost right while the rest drifts.
                "hidden_state": [round(float(x), 6) for x in h.flatten().tolist()],
            }
        )

    OUT.write_text(json.dumps(out, ensure_ascii=False))
    sys.stderr.write(
        f"[pin_ettin] wrote {OUT} — {len(PAIRS)} cases, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""pin_ettin_reranker.py — end-to-end reranker parity golden for
cross-encoder/ettin-reranker-17m-v1, the oracle for
encoder/modernbert_crossencoder.go.

Reimplements the sentence-transformers module chain directly (four modules, three
tensor files) rather than importing sentence_transformers, because the installed
Python is 3.9 and st 5.4.1 / transformers 5.7 need 3.10+. That is a feature here:
the chain below is transcribed from modules.json and the four config.json files, so
it is an independent reading of the artifact rather than a second copy of the Go
loader's assumptions.

    h     = ModernBertModel(input_ids).last_hidden_state
    cls   = h[0]                       # 1_Pooling, pooling_mode "cls"
    x     = gelu(cls @ W2.T)           # 2_Dense, 256->256, bias=false
    x     = LayerNorm(x, g3, b3)       # 3_LayerNorm
    score = x @ W4.T + b4              # 4_Dense, 256->1

Pairs are framed by the real HF tokenizer (post_processor pair template), so the
input_ids in the golden also pin aikit's own pair assembly.

Run from repo root:
    .venv/bin/python scripts/pin_ettin_reranker.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = REPO_ROOT / "testdata" / "ettin-reranker-17m"
OUT = REPO_ROOT / "testdata" / "ettin_reranker_golden.json"

QUERY = "What is the capital of France?"

CASES = [
    # (query, doc). The first three share a query and are a known ranking:
    # relevant > related-but-wrong > unrelated. A sign error or a cls/mean pooling
    # mistake cannot reproduce that order and the magnitudes at the same time.
    (QUERY, "Paris is the capital and most populous city of France."),
    (QUERY, "Berlin is the capital of Germany."),
    (QUERY, "Sourdough needs a starter, flour, water and salt."),
    ("int8 quantization", "Quantizing weights to int8 cuts memory roughly 4x versus float32."),
    ("日本語の検索", "これは日本語で書かれた文書です。検索の対象になります。"),
    ("", "an empty query still has to score"),
    ("a query with no matching document at all", ""),
    # Longer than max_seq_length (7999) once framed, so the longest_first trim path
    # is exercised end to end rather than only in a unit test.
    ("truncation", "The quick brown fox jumps over the lazy dog. " * 900),
    # A query long enough that longest_first has to trim the QUERY, not the doc.
    ("quantization " * 5000, "int8 quantization reduces memory."),
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
        _attn_implementation="eager",
    )
    model = ModernBertModel(cfg)
    missing, unexpected = model.load_state_dict(
        load_file(MODEL / "model.safetensors"), strict=False
    )
    if missing or unexpected:
        sys.stderr.write(f"[pin_ettin_reranker] state_dict mismatch: {missing} / {unexpected}\n")
        return 1
    model.eval()

    w2 = load_file(MODEL / "2_Dense" / "model.safetensors")["linear.weight"]
    ln = load_file(MODEL / "3_LayerNorm" / "model.safetensors")
    d4 = load_file(MODEL / "4_Dense" / "model.safetensors")
    w4, b4 = d4["linear.weight"], d4["linear.bias"]
    eps = raw["norm_eps"]
    D = raw["hidden_size"]

    # max_seq_length: sentence_bert_config.json declares none for this checkpoint,
    # so the ceiling is max_position_embeddings. The Go side derives the same value;
    # pinning it here makes a divergence a golden mismatch rather than a silent
    # difference in how much of a long document each side reads.
    max_seq = raw["max_position_embeddings"]

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    tok.no_truncation()

    def frame(query: str, doc: str):
        """[CLS] q [SEP] d [SEP] with longest_first truncation to max_seq."""
        q = tok.encode(query, add_special_tokens=False).ids
        d = tok.encode(doc, add_special_tokens=False).ids
        avail = max(max_seq - 3, 0)
        while len(q) + len(d) > avail:
            if len(q) > len(d):
                q = q[:-1]
            else:
                d = d[:-1]
        cls, sep = raw["cls_token_id"], raw["sep_token_id"]
        return [cls] + q + [sep] + d + [sep]

    out = {
        "model": "cross-encoder/ettin-reranker-17m-v1",
        "labels": int(w4.shape[0]),
        "max_seq": max_seq,
        "cases": [],
    }
    for query, doc in CASES:
        ids = frame(query, doc)
        t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            h = model(input_ids=t, attention_mask=torch.ones_like(t)).last_hidden_state[0]
            cls = h[0]
            x = torch.nn.functional.gelu(cls @ w2.T)          # exact erf GELU
            x = torch.nn.functional.layer_norm(x, (D,), ln["norm.weight"], ln["norm.bias"], eps)
            score = x @ w4.T + b4
        out["cases"].append(
            {
                "query": query,
                "doc": doc,
                "input_ids": ids,
                "score": [float(v) for v in score.tolist()],
            }
        )

    OUT.write_text(json.dumps(out, ensure_ascii=False))
    sys.stderr.write(
        f"[pin_ettin_reranker] wrote {OUT} — {len(CASES)} cases, "
        f"{OUT.stat().st_size // 1024} KB\n"
    )
    for c in out["cases"][:3]:
        sys.stderr.write(f"    {c['score'][0]:+.4f}  {c['doc'][:56]!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

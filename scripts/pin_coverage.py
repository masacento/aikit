#!/usr/bin/env python3
"""pin_coverage.py — golden fixtures for the BERT-family coverage-breadth models
(docs/task-embedding-coverage.md, Bucket A). One driver for several models that
share the already-certified architectures (learned-absolute BERT + WordPiece, or
BERT-shaped + Unigram), so each new row is config/tokenizer wiring plus a gate,
not new architecture.

For each model it (optionally) downloads the checkpoint into testdata/<dir>, then
dumps, per curated case, from the real sentence-transformers reference:
  - input_ids : the ids the model ate ([CLS] … [SEP]).
  - hidden    : last_hidden_state [L, D] flattened row-major, with L.
  - embedding : the pooled (CLS or mean), L2-normalized sentence embedding [D].

Usage (from repo root):
    .venv/bin/python scripts/pin_coverage.py bge-large      # one model
    .venv/bin/python scripts/pin_coverage.py --all          # every model
    .venv/bin/python scripts/pin_coverage.py bge-large --download   # fetch first
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# key -> (HF repo id, testdata dir, golden filename, pooling)
MODELS = {
    "bge-large":               ("BAAI/bge-large-en-v1.5",                               "bge-large",   "bge_large_golden.json",   "cls"),
    "mxbai-large":             ("mixedbread-ai/mxbai-embed-large-v1",                   "mxbai-large", "mxbai_golden.json",       "cls"),
    "arctic-embed-m":          ("Snowflake/snowflake-arctic-embed-m",                   "arctic-embed-m", "arctic_golden.json",   "cls"),
    "paraphrase-multilingual": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-ml", "paraphrase_ml_golden.json", "mean"),
}

# Curated cases: short, varied, and (for the multilingual model) cross-script, so
# the tokenizer + forward are exercised on more than ASCII. Kept small — the
# per-token hidden dump stays compact.
CASES = [
    "how do i parse json",
    "def add(a, b):\n    return a + b",
    "",  # degenerate: [CLS][SEP]
    "hello world",
    "x",  # single char
    "compute the sha256 hash of a file",
    "Represent this sentence for searching relevant passages",
    "Bonjour le monde, ça va?",
    "Здравствуй, мир",
    "識別子を検索する",
]


def pin_one(key: str, download: bool) -> int:
    from sentence_transformers import SentenceTransformer

    repo, dirname, goldenname, pooling = MODELS[key]
    model_dir = REPO_ROOT / "testdata" / dirname
    out = REPO_ROOT / "testdata" / goldenname

    if download or not (model_dir / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        sys.stderr.write(f"[pin_coverage] downloading {repo} -> {model_dir} ...\n")
        snapshot_download(
            repo_id=repo,
            local_dir=str(model_dir),
            allow_patterns=[
                "config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json",
                "sentence_bert_config.json", "modules.json",
                "1_Pooling/config.json", "sentencepiece.bpe.model", "vocab.txt",
            ],
        )

    sys.stderr.write(f"[pin_coverage] loading {model_dir} (CPU, fp32) ...\n")
    # Force float32: some checkpoints ship fp16 weights (e.g. mxbai), and a fp16
    # reference forward would diverge from aikit's fp32 kernel by ~1e-2 in the
    # hidden states (cosine stays ~1, but the hidden-state gate would trip on
    # reference rounding, not a real bug). aikit widens fp16→fp32 and computes in
    # fp32; the reference must do the same for an apples-to-apples parity gate.
    import torch
    m = SentenceTransformer(
        str(model_dir), device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    cfg = m[0].auto_model.config

    out_obj = {
        "model": repo,
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "intermediate": cfg.intermediate_size,
        "max_pos": cfg.max_position_embeddings,
        "type_vocab": cfg.type_vocab_size,
        "ln_eps": cfg.layer_norm_eps,
        "act": cfg.hidden_act,
        "pos": cfg.position_embedding_type,
        "pooling": pooling,
        "cases": [],
    }
    for text in CASES:
        ids = m.tokenize([text])["input_ids"][0].tolist()
        hid = m.encode(text, output_value="token_embeddings", convert_to_numpy=True)
        emb = m.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        out_obj["cases"].append({
            "text": text,
            "input_ids": [int(i) for i in ids],
            "L": int(hid.shape[0]),
            "hidden": [round(float(x), 6) for x in hid.reshape(-1).tolist()],
            "embedding": [round(float(x), 6) for x in emb.tolist()],
        })

    out.write_text(json.dumps(out_obj))
    sys.stderr.write(
        f"[pin_coverage] wrote {out} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"pooling={pooling}, {out.stat().st_size // 1024} KB\n"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", choices=sorted(MODELS), help="model key")
    ap.add_argument("--all", action="store_true", help="pin every model")
    ap.add_argument("--download", action="store_true", help="force re-download")
    args = ap.parse_args()

    keys = sorted(MODELS) if args.all else ([args.model] if args.model else [])
    if not keys:
        ap.error("give a model key or --all")
    rc = 0
    for k in keys:
        rc |= pin_one(k, args.download)
    return rc


if __name__ == "__main__":
    sys.exit(main())

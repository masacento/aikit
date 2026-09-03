#!/usr/bin/env python3
"""pin_tokenclassification.py — span parity golden for ner's TokenClassifier,
the oracle being the model card's own span_infer.py (AndrewAndrewsen/
distilbert-secret-masker-v3.3a-rs). The checkpoint is secret-flavored, but
what it pins is the generic ForTokenClassification path; the secret-specific
consumer (masking, CLI) lives in examples/secretmasker.

The reference logic is reproduced inline rather than imported (the checkpoint
ships span_infer.py, but the point of the golden is to pin THAT contract
explicitly): raw-text tokenization with offset_mapping, manual overflow
windows (body = max_length-2, first-window-wins dedup by (start, end)),
lenient BIO decode, entity score = mean P(B)+P(I), threshold mode on top.
Label ids 1/2 are the B-/I- roles of this checkpoint's config (0=O).

Dumps, per case:
  - spans      : argmax mode (start/end are PYTHON CHARACTER offsets; the Go
                 side reports byte offsets and the test converts)
  - spans_tau  : threshold mode, tau = 0.99 (the reference CLI default)
  - masked     : mask_text over the argmax spans, mask = "***MASKED***"
  - invalid_bio_transitions
  - pieces     : per-piece (start, end, label, p_secret) — only for the first
                 SHORT cases, to let a red gate say "forward" vs "decode"
Plus one trunk block (case 0): specials-wrapped ids and last_hidden_state
stats, the oracle for encoder.LoadDistilBERT in isolation.

Run from repo root:
    uv pip install -q torch transformers safetensors
    uv run python scripts/pin_tokenclassification.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "distilbert-secret-masker-v3.3a-rs"
OUT = REPO_ROOT / "testdata" / "tokenclassification_golden.json"

MAX_LENGTH = 512
STRIDE = 128
TAU = 0.99
MASK = "***MASKED***"
N_PIECE_CASES = 3  # cases [0, N_PIECE_CASES) also store their piece tensors

LONG_LINES = []
for i in range(60):
    LONG_LINES.append(f"service_{i}: api_key = AKIA{i:016X}")
    LONG_LINES.append(f"postgresql://root:pw-{i}-hunter2@db-{i}.internal:5432/app{i}")
LONG_TEXT = "\n".join(LONG_LINES)

CASES = [
    # the canonical pair
    "aws_access_key_id = AKIAIOSFODNN7EXAMPLE\naws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    # single-line tokens of different shapes
    "token: ghp_16CharsFFFFFFFFFFFFFFFF1234  # github PAT",
    "password = hunter2!",
    # multiline secret block
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7X\nabc123def456\n-----END RSA PRIVATE KEY-----",
    # JSON with mixed secrets
    '{"api_key": "sk-live-9hK2mL", "Authorization": "Bearer eyJhbGciOi.eyJzdWIi.9f8e7d", "debug": true}',
    # connection string with credentials
    "DATABASE_URL=postgres://admin:s3cr3t!pass@prod-db.example.com:5432/main",
    # no secrets at all
    "def add(a, b):\n    return a + b\n\nprint(add(1, 2))",
    "The quick brown fox jumps over the lazy dog.",
    # degenerate inputs
    "",
    " ",
    # unicode: CJK (per-char spaces from handle_chinese_chars), accents that
    # normalization strips (offsets must survive), full-width punctuation
    "パスワード: P@sswörd-2026! éè",
    "café token = ghp_ naïve ünïcode password1234",
    # an added-token literal in the raw text (matched pre-normalization)
    "[MASK] = supersecret123",
    "[CLS] suspicious [SEP] = hunter2",
    # long synthetic document: many windows, overlap dedup, secrets landing
    # inside overlap regions
    LONG_TEXT,
]


def infer(text, tok, model, id2label, mode, tau):
    """span_infer.py's predict_pieces + decode_entities + infer_spans."""
    full = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = full["input_ids"], full["offset_mapping"]
    n = len(ids)
    body = MAX_LENGTH - 2
    step = max(1, body - STRIDE)
    windows = []
    for w0 in range(0, max(n, 1), step):
        w1 = min(w0 + body, n)
        windows.append((w0, w1))
        if w1 >= n:
            break

    pieces, seen = [], set()
    for b0 in range(0, len(windows), 8):
        batch = windows[b0:b0 + 8]
        maxlen = max(w1 - w0 for w0, w1 in batch) + 2
        input_ids, attn = [], []
        for w0, w1 in batch:
            row = [tok.cls_token_id] + ids[w0:w1] + [tok.sep_token_id]
            pad = maxlen - len(row)
            input_ids.append(row + [tok.pad_token_id] * pad)
            attn.append([1] * len(row) + [0] * pad)
        assert maxlen <= MAX_LENGTH
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor(input_ids),
                attention_mask=torch.tensor(attn),
            ).logits
        probs = torch.softmax(logits, dim=-1)
        for wi, (w0, w1) in enumerate(batch):
            for k in range(w1 - w0):
                s, e = offs[w0 + k]
                if s == e or (s, e) in seen:
                    continue
                seen.add((s, e))
                p = probs[wi, k + 1]  # +1 skips [CLS]
                pieces.append((s, e, int(p.argmax()), float(p[1] + p[2])))
    pieces.sort(key=lambda x: (x[0], x[1]))

    # lenient BIO decode
    entities, invalid, cur, prev_label = [], 0, None, "O"
    for (s, e, lid, p_secret) in pieces:
        label = id2label[lid]
        if label.startswith("B-"):
            if cur is not None:
                entities.append(cur)
            cur = [s, e, [p_secret]]
        elif label.startswith("I-"):
            if cur is None:
                invalid += 1
                cur = [s, e, [p_secret]]
            else:
                cur[1] = e
                cur[2].append(p_secret)
        else:
            if cur is not None:
                entities.append(cur)
                cur = None
        prev_label = label
    if cur is not None:
        entities.append(cur)

    spans = []
    for s, e, ps in entities:
        score = sum(ps) / len(ps)
        if mode == "threshold" and score < tau:
            continue
        spans.append({
            "start": s,
            "end": e,
            "line": text[:s].count("\n") + 1,
            "value": text[s:e],
            "score": round(score, 6),
        })
    return spans, pieces, invalid


def mask_text(text, spans):
    out = text
    for sp in sorted(spans, key=lambda x: x["start"], reverse=True):
        out = out[:sp["start"]] + MASK + out[sp["end"]:]
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    assert tok.is_fast
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    assert id2label == {0: "O", 1: "B-SECRET", 2: "I-SECRET"}, id2label

    cases = []
    for ci, text in enumerate(CASES):
        spans, pieces, invalid = infer(text, tok, model, id2label, "argmax", None)
        spans_tau, _, _ = infer(text, tok, model, id2label, "threshold", TAU)
        case = {
            "text": text,
            "spans": spans,
            "spans_tau": spans_tau,
            "masked": mask_text(text, spans),
            "invalid_bio_transitions": invalid,
        }
        if ci < N_PIECE_CASES:
            case["pieces"] = [
                {"start": s, "end": e, "label": l, "p": round(p, 6)}
                for (s, e, l, p) in pieces
            ]
        cases.append(case)

    # trunk block: encoder.LoadDistilBERT's oracle, isolated from head+decode.
    ref = CASES[0]
    enc = tok(ref, return_tensors="pt")
    with torch.no_grad():
        hidden = model.distilbert(**enc).last_hidden_state[0]
    flat = hidden.flatten()
    cases[0]["trunk"] = {
        "ids": enc["input_ids"][0].tolist(),
        "mean": float(flat.mean()),
        "abs_mean": float(flat.abs().mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "row0": [float(v) for v in hidden[0, :16]],
        "row1": [float(v) for v in hidden[1, :16]],
    }

    golden = {
        "model": "AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs",
        "reference": "span_infer.py (model repo), max_length=512 stride=128",
        "tau": TAU,
        "mask": MASK,
        "cases": cases,
    }
    OUT.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n")
    n_spans = sum(len(c["spans"]) for c in cases)
    print(f"wrote {OUT} — {len(cases)} cases, {n_spans} argmax spans, "
          f"{sum(len(c['spans_tau']) for c in cases)} tau spans")


if __name__ == "__main__":
    main()

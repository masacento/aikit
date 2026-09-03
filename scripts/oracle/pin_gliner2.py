#!/usr/bin/env python
# pin_gliner2.py — golden generator for the GLiNER2 boundary port.
#
# Phase 1 (this much is standalone): word-splitter and tokenizer goldens from
# gliner2's own WhitespaceTokenSplitter and the HF fast tokenizer loaded from
# tokenizer.json, plus the schema-prompt token layout. Phase 2 extends this with
# backbone/head logits via BoundaryExtractorModel.score_candidates.
#
# Usage:  uv run python scripts/pin_gliner2.py
# Writes: testdata/gliner2_golden.json

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO_ROOT, "testdata", "gliner2_golden.json")
CKPT = os.path.join(REPO_ROOT, "testdata", "gliner-multi-v2.5")

SPLIT_CASES = [
    "Apple hired John Smith in Cupertino.",
    "権藤三峰 は 武将 で ある",
    "Visit https://Example.COM/Path?q=1 or www.fastino.ai/docs today",
    "Mail John.Doe+tag@Example.Co.UK about it",
    "not.an.email@broken and a@b.c stay split",
    "ping @gliner2_dev and @user-1 now",
    "state-of-the-art snake_case_name a- b- x_y-z",
    "Hawaii. 「日本語、句読点。」 3.14159 *+-/ =",
    "UPPER lower MiXeD İstanbul",
]

# (text, labels) pairs for the forward/parity goldens.
EXTRACT_CASES = [
    ("Apple hired John Smith in Cupertino.", ["person", "organization"]),
    ("権藤三峰 は 武将 で ある", ["person", "weapon"]),
    ("Angela Merkel visited Paris on 12 March 2019 and met Emmanuel Macron at the "
     "Élysée Palace to discuss climate policy with Ursula von der Leyen.",
     ["person", "location", "date"]),
]


def split_golden():
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    splitter = WhitespaceTokenSplitter()
    cases = []
    for text in SPLIT_CASES:
        toks = [
            {"text": t, "start": s, "end": e}  # CHARACTER offsets (Python str)
            for t, s, e in splitter(text, lower=True)
        ]
        cases.append({"text": text, "tokens": toks})
    return cases


def tokenize_golden(tok):
    groups = [
        ["Apple", "hired", "John", "Smith"],
        ["権藤三峰", "は", "武将", "で", "ある"],
        ["state-of-the-art", "snake_case_name", "İstanbul"],
        ["https://Example.COM/Path?q=1", "John.Doe+tag@Example.Co.UK"],
        ["", " "],
    ]
    cases = []
    for words in groups:
        ids, first = [], []
        for w in words:
            sub = tok.tokenize(w)
            first.append(len(ids))
            ids.extend(tok.convert_tokens_to_ids(sub))
        cases.append({"words": words, "ids": ids, "first": first})
    return cases


def prompt_golden(tok):
    """Replicate gliner2's _transform_schema word sequence for entity requests:

        ( [P] entities ( [E] label₁ [E] label₂ … ) ) [SEP_TEXT]

    each label appended as ONE word; then tokenize word-by-word, recording each
    [E] marker's position (a single token) and, per prompt word, the first sub-id
    offset the way the Go buildPrompt assembles ids.
    """
    groups = [
        ["person"],
        ["person", "location", "date of birth"],
    ]
    special = tok.convert_tokens_to_ids
    cases = []
    for labels in groups:
        words = ["(", "[P]", "entities", "("]
        markers = []  # positions into the ID sequence
        for l in labels:
            words.append("[E]")
            words.append(l)
        words += [")", ")", "[SEP_TEXT]"]
        ids = []
        for w in words:
            if w in ("[P]", "[E]", "[SEP_TEXT]"):
                if w == "[E]":
                    markers.append(len(ids))
                ids.append(special(w))
            else:
                sub = tok.tokenize(w)
                ids.extend(tok.convert_tokens_to_ids(sub))
        cases.append({"labels": labels, "ids": ids, "markers": markers})
    return cases


def extract_golden():
    """Forward-pass + decoded goldens from the BoundaryExtractor internals.

    Mirrors BoundaryExtractor._extract_from_batch (engine.py) step by step for
    B=1: collate → _encode_core → boundary_head → candidates + aux logits, and
    also runs the public batch_extract for the decoded entities.
    """
    import torch
    from gliner2.inference.runtime import ExtractorCollator
    from gliner2.models.boundary.engine import BoundaryExtractor

    model = BoundaryExtractor.from_pretrained(CKPT)
    model.eval()
    collator = ExtractorCollator(model.processor, is_training=False, architecture="boundary")

    cases = []
    with torch.no_grad():
        for text, labels in EXTRACT_CASES:
            schema = {"entities": {l: {} for l in labels}}
            schema_dicts, metadata_list = model._build_schema_dicts_and_metadata([schema])
            batch = collator([(text, schema_dicts[0])])
            core = model._encode_core(batch)
            out = model.boundary_head(
                core["text_states"], core["text_mask"],
                core["query_states"], core["query_mask"],
            )
            cand = out.candidates
            L = int(core["text_mask"][0].sum())
            Q = int(core["query_mask"][0].sum())
            decoded = model.batch_extract(
                [text], [schema], include_confidence=True, include_spans=True)
            cases.append({
                "text": text,
                "labels": labels,
                "input_ids": batch.input_ids[0][: int(batch.attention_mask[0].sum())].tolist(),
                "word_positions": batch.text_word_indices[0][:L].tolist(),
                "marker_positions": batch.query_marker_indices[0][:Q].tolist(),
                "text_states": core["text_states"][0].tolist(),
                "query_states": core["query_states"][0].tolist(),
                "start_logits": out.start_logits[0].tolist(),
                "end_logits": out.end_logits[0].tolist(),
                "inside_logits": out.inside_logits[0].tolist(),
                "null_logits": (out.null_logits[0].tolist()
                                if out.null_logits is not None else None),
                "candidate_indices": cand.indices[0].tolist(),
                "candidate_valid": cand.valid_mask[0].tolist(),
                "pair_logits": cand.pair_logits[0].tolist(),
                "decoded": decoded[0],
            })
    return cases


def classification_golden():
    """One classification case through the shared classifier head."""
    import torch
    from gliner2.inference.runtime import ExtractorCollator
    from gliner2.models.boundary.engine import BoundaryExtractor

    model = BoundaryExtractor.from_pretrained(CKPT)
    model.eval()
    collator = ExtractorCollator(model.processor, is_training=False, architecture="boundary")

    text = "The board approved the merger after weeks of negotiation."
    labels = ["positive", "negative", "neutral"]
    schema = {"classifications": [{"task": "sentiment", "labels": labels, "true_label": []}]}
    with torch.no_grad():
        schema_dicts, _ = model._build_schema_dicts_and_metadata([schema])
        batch = collator([(text, schema_dicts[0])])
        decoded = model.batch_extract([text], [schema])
        cases = [{
            "text": text,
            "task": "sentiment",
            "labels": labels,
            "input_ids": batch.input_ids[0][: int(batch.attention_mask[0].sum())].tolist(),
            "marker_positions": batch.cls_marker_indices[0].tolist(),
            "decoded": decoded[0],
        }]
    return cases


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    golden = {
        "checkpoint": "fastino/gliner2.5-multi-v1",
        "split_cases": split_golden(),
        "tokenize_cases": tokenize_golden(tok),
        "prompt_cases": prompt_golden(tok),
        "extract_cases": extract_golden(),
        "classification_cases": classification_golden(),
    }
    with open(DEST, "w") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)
    print(f"wrote {DEST}: {len(golden['split_cases'])} split, "
          f"{len(golden['tokenize_cases'])} tokenize, "
          f"{len(golden['prompt_cases'])} prompt, "
          f"{len(golden['extract_cases'])} extract cases")


if __name__ == "__main__":
    sys.exit(main())

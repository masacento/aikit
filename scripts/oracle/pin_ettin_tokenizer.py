#!/usr/bin/env python3
"""pin_ettin_tokenizer.py — tokenizer parity golden for the OLMo/ModernBERT
byte-level BPE tokenizer (cross-encoder/ettin-reranker-17m-v1), the oracle for
aikit's embed bpeBackend (embed/tokenize_bpe.go).

This is the SAME backend granite-embedding-*-english uses, in its 2026 spelling:
`merges` as [a,b] pairs, a TemplateProcessing post-processor ([CLS] $A [SEP])
rather than RobertaProcessing, and an explicit NFC normalizer. The corpus below
targets those three plus the two things that make this vocab unusual:

  - NFC: pairs written both composed and decomposed (café / cafe + U+0301). The
    two MUST tokenize identically; without the normalizer they do not.
  - the whitespace added-tokens (ids 50254-50276, runs of 2..24 spaces), which are
    carved out of the RAW text before the ByteLevel pre-tokenizer sees it and so
    interact with preTokenize's whitespace give-back pass.

Dumps, per case:
  - ids  : the wrapped sequence ([CLS] … [SEP]) — what EncodeWithSpecials produces.
  - bare : the bare id sequence — what Encode produces.

Run from repo root:
    .venv/bin/python scripts/pin_ettin_tokenizer.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOK_JSON = REPO_ROOT / "testdata" / "ettin-reranker-17m" / "tokenizer.json"
OUT = REPO_ROOT / "testdata" / "ettin_tokenizer_golden.json"

CASES = [
    "hello world",
    " hello",
    "hello ",
    "",
    "x",
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b",
    # whitespace added-tokens: 2, 3, 8 and 24 spaces are single ids in this vocab
    # (50276, 50275, 50270, 50254); 25 is not, and must fall back to the merges.
    "a  b",
    "a   b",
    "a        b",
    "a" + " " * 24 + "b",
    "a" + " " * 25 + "b",
    " " * 4,
    "  leading and trailing  ",
    # NFC pairs — each composed form is followed by its decomposed twin.
    "café",                    # café, composed
    "café",                   # café, decomposed → must match the line above
    "Ångström",           # Ångström, composed
    "Ångström",         # Ångström, decomposed
    "ẛ̣",                 # ẛ̣ — NFC-reorders rather than composing
    "ß",                       # ß (no decomposition)
    # CJK / emoji / control — byte-level BPE has no <unk>, so these exercise the
    # byte→rune table rather than a fallback path.
    "識別子を検索する",   # 識別子を検索する
    "中文",                                       # 中文
    "\U0001F680",                                         # 🚀
    "mix ジョギング\U0001F680end",    # mix ジョギング🚀end
    "\x00",
    "a\x01b",
    "\U00029E3D",                                         # 𩸽
    "Ω",                                             # Ω
    "‽",                                             # ‽
    "123 456",
    "price: $100 (USD) [ok] {yes}",
    "email@example.com",
    "https://example.com/path?q=1&r=2",
    "UPPER lower MiXeD",
    "tab\there",
    "line1\nline2\r\nline3",
    "unicode: café ß Ω 中文 \U0001F680",
    # the literal special tokens, which the added-token carve-out must match
    # against raw text rather than tokenizing as text
    "[CLS] and [SEP] and [MASK]",
    "The quick brown fox jumps over the lazy dog. " * 12,
    # longer than max_position_embeddings (7999): the Go side truncates, so this
    # only pins that the untruncated prefix agrees.
    "quantization reduces model memory. " * 2500,
]


def main() -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_JSON))
    tok.no_truncation()
    out = {"model": "cross-encoder/ettin-reranker-17m-v1", "cases": []}
    for text in CASES:
        ids = tok.encode(text).ids                              # with specials
        bare = tok.encode(text, add_special_tokens=False).ids   # bare
        out["cases"].append({"text": text, "ids": ids, "bare": bare})

    OUT.write_text(json.dumps(out, ensure_ascii=False))
    sys.stderr.write(
        f"[pin_ettin_tokenizer] wrote {OUT} — {len(CASES)} cases, "
        f"{OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

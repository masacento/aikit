#!/usr/bin/env python3
"""pin_ruri_tokenizer.py — tokenizer parity golden for the cl-nagoya/ruri-v3-30m
SentencePiece Unigram tokenizer, the oracle for aikit's embed Unigram backend
(embed/tokenize_unigram.go) on the three axes that distinguish ruri from the
XLM-R / bge-m3 Unigram variants:

  - normalizer:null → the identity normalizer (no Precompiled charsmap): case and
    fullwidth are preserved, nothing is NFKC-folded.
  - Metaspace(prepend_scheme="never", split=false) → spaces map to ▁ with NO
    leading ▁ and the text kept as one chunk (vs bge-m3's prepend, XLM-R's
    whitespace-split).
  - byte_fallback:true → a character absent from the vocab decomposes into its
    UTF-8 bytes as "<0xNN>" tokens (vs XLM-R's single fused <unk>).

Dumps, per curated case, from the real HF `tokenizers` pipeline:
  - ids  : the wrapped sequence (<s> … </s>) — what EncodeWithSpecials produces.
  - bare : the bare id sequence (no specials) — what Encode produces.

Run from repo root:
    .venv/bin/python scripts/pin_ruri_tokenizer.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOK_JSON = REPO_ROOT / "testdata" / "ruri-v3-30m" / "tokenizer.json"
OUT = REPO_ROOT / "testdata" / "ruri_tokenizer_golden.json"

CASES = [
    # Metaspace(prepend_scheme="never"): leading/trailing/multiple spaces.
    "hello world",
    " hello",
    "hello ",
    "a  b",
    " ",
    "  ",
    "",
    "x",
    "  leading and trailing  ",
    # Tabs / other whitespace are NOT the Metaspace " " pattern — they pass through.
    "\t",
    "hello\tworld",
    "tab\there",
    "line1\nline2\r\nline3",
    # byte_fallback: chars absent from the vocab → <0xNN> byte tokens.
    "\u0000",                                                            # NUL → 1 byte
    "a\u0001b",                                                          # control char
    "\U0001F680",                                                        # 🚀 → 4 bytes
    "\U00029E3D",                                                        # 𩸽 → 4 bytes
    "\uE000",                                                            # private-use → 3 bytes
    "\u203D",                                                            # ‽ → 3 bytes
    "\U0001F1EF\U0001F1F5",                                              # 🇯🇵 → 8 bytes
    "mix \u30B8\u30E7\u30AE\u30F3\u30B0\U0001F680end",                    # mix ジョギング🚀end
    # identity normalizer: case + fullwidth preserved (no lowercase / NFKC).
    "ABC",
    "UPPER lower MiXeD",
    "\uFF23\uFF21\uFF26\uFF25",                                          # ＣＡＦＥ fullwidth
    "caf\u00E9",                                                         # café (é in vocab)
    "\u00E9",                                                            # é
    "\u00DF",                                                            # ß
    "\u03A9",                                                            # Ω
    # Japanese (ruri's core domain).
    "\u5BFF\u53F8\u306E\u7279\u5FB4\u306F\u4F55\u3067\u3059\u304B\uFF1F",  # 寿司の特徴は何ですか？
    "\u8B58\u5225\u5B50\u3092\u691C\u7D22\u3059\u308B",                    # 識別子を検索する
    "\u91CF\u5B50\u5316\u306F\u30E2\u30C7\u30EB\u306E\u30E1\u30E2\u30EA\u3092\u524A\u6E1B\u3059\u308B",  # 量子化は…削減する
    "\u4E2D\u6587",                                                      # 中文
    # Punctuation / structure.
    "price: $100 (USD) [ok] {yes}",
    "email@example.com",
    "https://example.com/path?q=1&r=2",
    "123 456",
    "def add(a, b):\n    return a + b",
    "The quick brown fox jumps over the lazy dog.",
    "unicode: caf\u00E9 \u00DF \u03A9 \u4E2D\u6587 \U0001F680",
    "The quick brown fox jumps over the lazy dog. " * 12,
]


def main() -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_JSON))
    out = {"model": "cl-nagoya/ruri-v3-30m", "cases": []}
    for text in CASES:
        ids = tok.encode(text).ids                              # with specials (<s> … </s>)
        bare = tok.encode(text, add_special_tokens=False).ids   # bare
        out["cases"].append({"text": text, "ids": ids, "bare": bare})

    OUT.write_text(json.dumps(out, ensure_ascii=False))
    sys.stderr.write(
        f"[pin_ruri_tokenizer] wrote {OUT} — {len(CASES)} cases, "
        f"{OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

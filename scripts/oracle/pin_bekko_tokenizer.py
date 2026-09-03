#!/usr/bin/env python3
"""pin_bekko_tokenizer.py — tokenizer parity golden for the ModernBERT /
bekko SentencePiece-style BPE tokenizer (Metaspace + byte-fallback), the oracle
for aikit's embed spBPEBackend (embed/tokenize_sp_bpe.go).

Dumps, per curated case, from the real HF `tokenizers` pipeline:
  - ids  : the wrapped sequence (<bos> … <eos>) — what EncodeWithSpecials produces.
  - bare : the bare id sequence (no specials) — what Encode produces.

The corpus deliberately stresses the axes that differ from the GPT-2 byte-level
BPE path: the Metaspace pre-tokenizer (leading/trailing/multiple spaces), the
byte-fallback decomposition (chars absent from the vocab → <0xNN>), CJK/emoji,
control characters, and the TemplateProcessing specials.

Run from repo root:
    .venv/bin/python scripts/pin_bekko_tokenizer.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOK_JSON = REPO_ROOT / "testdata" / "bekko-embedding-v1-a8m" / "tokenizer.json"
OUT = REPO_ROOT / "testdata" / "bekko_tokenizer_golden.json"

CASES = [
    "hello world",
    " hello",
    "hello ",
    "a  b",
    "",
    "x",
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b",
    "\u8B58\u5225\u5B50\u3092\u691C\u7D22\u3059\u308B",                       # 識別子を検索する
    "\u91CF\u5B50\u5316\u306F\u30E2\u30C7\u30EB\u306E\u30E1\u30E2\u30EA\u3092\u524A\u6E1B\u3059\u308B",  # 量子化は…削減する
    "caf\u00E9",                                                            # café
    "\u00DF",                                                               # ß
    "\U0001F680",                                                           # 🚀
    "mix \u30B8\u30E7\u30AE\u30F3\u30B0\U0001F680end",                       # mix ジョギング🚀end
    "\x00",                                                                 # NUL → byte fallback
    "a\x01b",                                                               # control char (in vocab)
    "\U00029E3D",                                                           # 𩸽 → byte fallback (4 bytes)
    "\uE000",                                                               # private-use (in vocab)
    "\u03A9",                                                               # Ω
    "\u203D",                                                               # ‽
    "123 456",
    "\u00E9",                                                               # é
    "\u4E2D\u6587",                                                         # 中文
    " ",
    "  ",
    "\t",
    "price: $100 (USD) [ok] {yes}",
    "email@example.com",
    "https://example.com/path?q=1&r=2",
    "UPPER lower MiXeD",
    "  leading and trailing  ",
    "tab\there",
    "line1\nline2\r\nline3",
    "unicode: caf\u00E9 \u00DF \u03A9 \u4E2D\u6587 \U0001F680",
    "The quick brown fox jumps over the lazy dog. " * 12,
]


def main() -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_JSON))
    out = {"model": "hotchpotch/bekko-embedding-v1-a8m", "cases": []}
    for text in CASES:
        ids = tok.encode(text).ids                              # with specials
        bare = tok.encode(text, add_special_tokens=False).ids   # bare
        out["cases"].append({"text": text, "ids": ids, "bare": bare})

    OUT.write_text(json.dumps(out, ensure_ascii=False))
    sys.stderr.write(
        f"[pin_bekko_tokenizer] wrote {OUT} — {len(CASES)} cases, "
        f"{OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

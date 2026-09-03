#!/usr/bin/env python3
"""pin_gliner_tokenizer.py — tokenizer parity golden for the GLiNER / mDeBERTa-v3
SentencePiece tokenizer, the oracle for aikit's raw-spm.model reader
(embed/tokenize_spm.go).

This path differs from every Unigram tokenizer aikit already covers, because the
checkpoint ships NO tokenizer.json — only a raw `spm.model` — so the pipeline is
configured from the ModelProto itself rather than from HF's converted JSON:

  - normalizer_spec: nmt_nfkc charsmap, add_dummy_prefix=1, remove_extra_whitespaces=1
    → a leading ▁ is inserted, space runs collapse, and the ends are trimmed.
  - ONE Viterbi chunk over the whole normalized string (not per whitespace-split
    word), so a vocab piece may span what used to be a space.
  - byte_fallback=1 → characters with no piece decompose to "<0xNN>" BYTE pieces.
  - piece TYPES matter: sentencepiece searches only NORMAL/USER_DEFINED/UNUSED
    pieces, so the literal texts "[CLS]" and "<0x41>" must NOT collapse to the
    control / byte ids that carry those exact spellings.

The reference is sentencepiece itself. transformers' DebertaV2Tokenizer is a thin
wrapper: with `split_by_punct=False` (which is what tokenizer_config.json declares)
its `_tokenize` is exactly `spm.encode(text)`, and its added-token handling is the
split-then-encode-each-segment below. Using sentencepiece directly keeps the oracle
free of a torch/transformers install.

Run from repo root:
    .venv/bin/python scripts/pin_gliner_tokenizer.py
"""
import json
from pathlib import Path

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "gliner-multi-v2.1"
SPM_MODEL = MODEL_DIR / "spm.model"
ADDED = MODEL_DIR / "added_tokens.json"
OUT = REPO_ROOT / "testdata" / "gliner_tokenizer_golden.json"

CLS_ID, SEP_ID = 1, 2

CASES = [
    # ── add_dummy_prefix / remove_extra_whitespaces ───────────────────────────
    "hello world",
    " hello",
    "hello ",
    "a  b",
    "a   very    spaced   line",
    "  leading and trailing  ",
    " ",
    "   ",
    "",
    "x",
    # Tabs and newlines are not U+0020; the charsmap decides their fate. Tab, LF and
    # CR all fold to a space, so they are indistinguishable from " " downstream —
    # but U+000B (VT) and U+0085 (NEL) do NOT fold: the charsmap deletes VT outright
    # and keeps NEL as a piece of its own. Both are `unicode.IsSpace` in Go, so they
    # are exactly where the single-chunk spm pre-tokenizer separates from a
    # whitespace-splitting one, which would drop them. Keep them in the corpus.
    "\t",
    "hello\tworld",
    "line1\nline2\r\nline3",
    "a\u000bb",
    "hello\u000bworld",
    "a\u000b\u000bb",
    "a\u0085b",
    "hello\u0085world",
    "\u0085x",
    "x\u0085",
    # ── multilingual ──────────────────────────────────────────────────────────
    "The quick brown fox jumps over the lazy dog.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "El veloz zorro marrón salta sobre el perro perezoso.",
    "Le rapide renard brun saute par-dessus le chien paresseux.",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "Η γρήγορη καφέ αλεπού πηδάει πάνω από το τεμπέλικο σκυλί.",
    "日本語のトークナイザをテストします。",
    "東京都渋谷区で機械学習の研究をしています",
    "中文分词测试：北京大学的研究人员",
    "한국어 토크나이저 테스트입니다",
    "ภาษาไทยทดสอบ",
    "नमस्ते दुनिया, यह एक परीक्षण है",
    "مرحبا بالعالم، هذا اختبار",
    "שלום עולם, זהו מבחן",
    "Xin chào thế giới",
    # ── nmt_nfkc folding: fullwidth, ligatures, compatibility forms ───────────
    "ＡＢＣ１２３",
    "ｱｲｳｴｵ",
    "ﬁre ﬂower",
    "①②③",
    "㈱㍿",
    "ＨＥＬＬＯ　ＷＯＲＬＤ",
    # Combining marks: the charsmap keys the base+mark cluster.
    "école",
    "école",
    "Å",
    # ── byte_fallback ─────────────────────────────────────────────────────────
    "\U0001F680 rocket",
    "🎉🎊🎈",
    "𩸽の刺身",
    "\u0000",
    "a\u0001b",
    # ── split_digits ──────────────────────────────────────────────────────────
    "1234567890",
    "version 2.1 released 2024-06-30",
    "3.14159",
    # ── code / punctuation (split_by_punct is FALSE for this tokenizer) ───────
    "func main() { fmt.Println(\"hi\") }",
    "a,b;c:d!e?f",
    "e-mail: user@example.com",
    "https://huggingface.co/urchade/gliner_multi-v2.1",
    # ── piece-type discipline: these must NOT hit control / byte / unk ids ────
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[UNK]",
    "<0x41>",
    "the [CLS] token",
    # ── added tokens (carved out before the Viterbi) ──────────────────────────
    "<<ENT>>",
    "<<SEP>>",
    "[MASK]",
    "[FLERT]",
    "<<ENT>>person<<ENT>>organization<<SEP>>Barack Obama was born in Hawaii",
    "text with [MASK] inside",
    "<<ENT>>a<<SEP>>b",
    # ── long-ish realistic NER inputs ─────────────────────────────────────────
    "Barack Obama was born in Honolulu, Hawaii on August 4, 1961.",
    "アップルは2007年にiPhoneを発表した。",
    "Angela Merkel besuchte Paris im Juni 2019.",
]

# Word-level cases for EncodeWords (HF `is_split_into_words=True`): each word is
# encoded independently, so each gets its own dummy prefix. The empty word is the
# interesting one — it produces no sub-tokens at all, which the Go side must report
# as -1 rather than silently aliasing the next word.
WORD_CASES = [
    ["Barack", "Obama", "was", "born", "in", "Hawaii"],
    ["東京", "都", "に", "住ん", "で", "います"],
    ["a", "", "b"],
    ["", ""],
    ["multi-word", "hyphen-ated", "tokens"],
    ["1961", ".", "August"],
    ["🚀", "rocket"],
    [],
]


def main() -> None:
    if not SPM_MODEL.exists():
        raise SystemExit(f"missing {SPM_MODEL} — fetch spm.model from the GLiNER mirror")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(SPM_MODEL))

    added = json.loads(ADDED.read_text()) if ADDED.exists() else {}
    # Longest-first, so a literal that prefixes another cannot shadow it. This is
    # HF's added-token trie split, flattened.
    added_keys = sorted(added, key=lambda k: (-len(k), k))

    def encode_bare(text: str) -> list[int]:
        """Carve out added tokens, spm-encode each remaining segment."""
        out: list[int] = []
        buf = ""
        i = 0
        while i < len(text):
            for k in added_keys:
                if text.startswith(k, i):
                    if buf:
                        out.extend(sp.EncodeAsIds(buf))
                        buf = ""
                    out.append(added[k])
                    i += len(k)
                    break
            else:
                buf += text[i]
                i += 1
        if buf:
            out.extend(sp.EncodeAsIds(buf))
        return out

    cases = []
    for text in CASES:
        bare = encode_bare(text)
        cases.append({
            "text": text,
            "bare": bare,
            "wrapped": [CLS_ID] + bare + [SEP_ID],
            "pieces": [sp.IdToPiece(i) if i < sp.GetPieceSize() else "<added>" for i in bare],
        })

    words = []
    for ws in WORD_CASES:
        ids: list[int] = []
        first: list[int] = []
        for w in ws:
            sub = encode_bare(w)
            if not sub:
                first.append(-1)
                continue
            first.append(len(ids))
            ids.extend(sub)
        words.append({"words": ws, "ids": ids, "first_subtok": first})

    OUT.write_text(json.dumps({
        "vocab_size": sp.GetPieceSize(),
        "added_tokens": added,
        "cases": cases,
        "word_cases": words,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(cases)} cases, "
          f"{len(words)} word cases, spm vocab {sp.GetPieceSize()}")


if __name__ == "__main__":
    main()

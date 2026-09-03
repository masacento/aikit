package embed

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"unicode/utf8"
)

const offsetsTokJSON = `{
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "special": true},
    {"id": 100, "content": "[UNK]", "special": true},
    {"id": 101, "content": "[CLS]", "special": true},
    {"id": 102, "content": "[SEP]", "special": true},
    {"id": 103, "content": "[MASK]", "special": true}
  ],
  "normalizer": {"type": "BertNormalizer", "clean_text": true,
                 "handle_chinese_chars": true, "strip_accents": null,
                 "lowercase": true},
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102,
              "[MASK]": 103, "cafe": 1, "##ine": 2, "x": 3, "中": 4,
              "文": 5, "abc": 6, "pwd": 7, "tab": 8, "=": 9}
  }
}`

func loadOffsetsTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(offsetsTokJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestEncodeOffsets_spans pins the offset contract on hand-computable cases:
// every span must re-slice the ORIGINAL text, survive normalization that
// changes byte length (accent strip, lowercase, CJK padding), and cover whole
// words for [UNK] — the exact values the WordPiece-level BIO decode consumes
// upstream.
func TestEncodeOffsets_spans(t *testing.T) {
	tok := loadOffsetsTokenizer(t)
	cases := []struct {
		text string
		want [][2]int
	}{
		// "Café" is 4 runes / 5 bytes; the é that NFD strips must not shrink
		// the piece's span.
		{"Café x", [][2]int{{0, 5}, {6, 7}}},
		// lowercase maps runes in place; spans stay byte-exact.
		{"ABC = cafeine", [][2]int{{0, 3}, {4, 5}, {6, 10}, {10, 13}}},
		// handle_chinese_chars pads each ideograph with spaces in the
		// NORMALIZED text; spans point back into the unpadded original.
		{"中文x", [][2]int{{0, 3}, {3, 6}, {6, 7}}},
		// an added-token literal carries its own raw span, pre-normalization.
		{"[MASK] pwd", [][2]int{{0, 6}, {7, 10}}},
		// [UNK] covers its whole word — HF's offsets for it too.
		{"qqq x", [][2]int{{0, 3}, {4, 5}}},
		// control chars (\t) normalize to spaces and vanish at pre-tokenize.
		{"\tTab\tx", [][2]int{{1, 4}, {5, 6}}},
		{"", nil},
	}
	for _, c := range cases {
		ids, offs, err := tok.EncodeOffsets(c.text)
		if err != nil {
			t.Fatalf("EncodeOffsets(%q): %v", c.text, err)
		}
		if len(ids) != len(c.want) {
			t.Errorf("EncodeOffsets(%q): %d pieces, want %d (ids %v offs %v)",
				c.text, len(ids), len(c.want), ids, offs)
			continue
		}
		for k, want := range c.want {
			s, e := offs[k][0], offs[k][1]
			if s != want[0] || e != want[1] {
				t.Errorf("EncodeOffsets(%q) piece %d span [%d,%d), want [%d,%d)",
					c.text, k, s, e, want[0], want[1])
			}
			if s > e || e > len(c.text) {
				t.Errorf("EncodeOffsets(%q) piece %d span [%d,%d) out of bounds", c.text, k, s, e)
			}
		}
	}
}

// TestEncodeOffsets_idsMatchEncode: the offsets path re-walks the pipeline
// with tracking, so its IDS must equal the ids-only Encode byte-for-byte —
// the offsets feature may never fork tokenization.
func TestEncodeOffsets_idsMatchEncode(t *testing.T) {
	tok := loadOffsetsTokenizer(t)
	texts := []string{
		"Café Résumé 中文 x=cafeine",
		"[MASK] = [CLS] literal specials [SEP]",
		"QQQ abcDEF ##weird   multiple   spaces",
		"tab\tseparated\nlines\r\nmixed  ",
		"x\xff bad byte",
		"!!!???...",
	}
	for _, text := range texts {
		want := tok.Encode(text)
		ids, offs, err := tok.EncodeOffsets(text)
		if err != nil {
			t.Fatalf("EncodeOffsets(%q): %v", text, err)
		}
		if !reflect.DeepEqual(ids, want) {
			t.Errorf("EncodeOffsets(%q) ids = %v, Encode = %v", text, ids, want)
		}
		if len(offs) != len(ids) {
			t.Errorf("EncodeOffsets(%q): %d offsets for %d ids", text, len(offs), len(ids))
			continue
		}
		for k := range offs {
			s, e := offs[k][0], offs[k][1]
			if s < 0 || e > len(text) || s >= e {
				t.Errorf("EncodeOffsets(%q) piece %d span [%d,%d) invalid", text, k, s, e)
				continue
			}
			if !utf8.ValidString(text[s:e]) {
				t.Errorf("EncodeOffsets(%q) piece %d span [%d,%d) splits a rune", text, k, s, e)
			}
		}
	}
}

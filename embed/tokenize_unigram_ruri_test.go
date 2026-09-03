package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// TestUnigram_ruriEncode certifies the cl-nagoya/ruri-v3-30m tokenizer variant
// against the raw HF `tokenizers` contract (scripts/pin_ruri_tokenizer.py). ruri
// differs from the XLM-R / bge-m3 Unigram variants on three config-driven axes
// this exercises: a null normalizer (identity — no Precompiled charsmap, so case
// and fullwidth survive), a Metaspace(prepend_scheme="never", split=false)
// pre-tokenizer (spaces → ▁ with NO leading ▁, one chunk), and byte_fallback
// (an out-of-vocab char decomposes into "<0xNN>" byte tokens rather than one
// fused <unk>). Cases include leading/trailing/multiple spaces, byte-fallback
// chars (NUL, emoji, rare CJK, private-use), case/fullwidth preservation, and
// Japanese — so this is the gate that the ruri path is exact, not the XLM-R or
// bge-m3 path reused.
func TestUnigram_ruriEncode(t *testing.T) {
	const path = "../testdata/ruri-v3-30m/tokenizer.json"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no ruri-v3-30m tokenizer at %s", path)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	if tok.uni == nil {
		t.Fatal("expected Unigram backend for ruri-v3-30m")
	}
	if tok.uni.metaspace != metaspaceKindNever {
		t.Fatalf("ruri pre_tokenizer is Metaspace(prepend_scheme=never), got kind %d", tok.uni.metaspace)
	}
	if tok.uni.norm != nil {
		t.Fatal("ruri normalizer is null — expected a nil (identity) Precompiled normalizer")
	}
	if !tok.uni.model.byteFallback {
		t.Fatal("ruri model has byte_fallback=true")
	}

	raw, err := os.ReadFile("../testdata/ruri_tokenizer_golden.json")
	if err != nil {
		t.Skip("no encode golden — run scripts/pin_ruri_tokenizer.py")
	}
	var g struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`  // wrapped (<s> … </s>)
			Bare []int32 `json:"bare"` // no specials
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	const maxLen = 8192
	mism := 0
	for _, c := range g.Cases {
		wrapped, err := tok.EncodeWithSpecials(c.Text, maxLen)
		if err != nil {
			t.Fatalf("%q: %v", c.Text, err)
		}
		if !equalIDs(wrapped, c.IDs) {
			mism++
			t.Errorf("wrapped %q:\n got  %v\n want %v", c.Text, wrapped, c.IDs)
		}
		if bare := tok.Encode(c.Text); !equalIDs(bare, c.Bare) {
			mism++
			t.Errorf("bare %q:\n got  %v\n want %v", c.Text, bare, c.Bare)
		}
	}
	if mism == 0 {
		t.Logf("ruri Unigram: %d/%d cases id-exact (wrapped + bare) vs raw HF tokenizer", len(g.Cases), len(g.Cases))
	}
}

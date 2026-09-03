package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// TestBPE_ettinParity certifies the byte-level BPE tokenizer in its OLMo/ModernBERT
// spelling (cross-encoder/ettin-reranker-17m-v1) against the real HF `tokenizers`
// pipeline (scripts/oracle/pin_ettin_tokenizer.py). Same bpeBackend granite uses; what is
// new is the [a,b] merge format, the TemplateProcessing post-processor and the NFC
// normalizer. Each of those fails differently — the first at load, the other two
// silently — so the assertion is exact id equality over the whole corpus, wrapped
// and bare.
//
// Truncation is deliberately NOT exercised here: the golden is dumped with
// no_truncation() so the long case pins the full sequence, and maxLen is set above
// every case's length. Truncation is the caller's (encoder's) contract, not this
// backend's.
func TestBPE_ettinParity(t *testing.T) {
	const tokJSON = "../testdata/ettin-reranker-17m/tokenizer.json"
	if _, err := os.Stat(tokJSON); err != nil {
		t.Skipf("no ettin tokenizer at %s — run scripts/fetch_ettin.sh", tokJSON)
	}
	tok, err := LoadTokenizer(tokJSON)
	if err != nil {
		t.Fatal(err)
	}
	// The Metaspace probe must NOT claim this one: a ByteLevel pre-tokenizer routed
	// to spBPEBackend mis-tokenizes silently.
	if tok.bpe == nil {
		t.Fatalf("tokenizer.json dispatched elsewhere, want *bpeBackend (byte-level BPE)")
	}
	if !tok.bpe.normNFC {
		t.Error("normalizer NFC not picked up; non-ASCII cases will diverge")
	}
	if p, s := tok.TemplateSpecials(); len(p) != 1 || p[0] != 50281 || len(s) != 1 || s[0] != 50282 {
		t.Fatalf("TemplateProcessing specials = %v/%v, want [50281]/[50282] ([CLS]/[SEP])", p, s)
	}

	raw, err := os.ReadFile("../testdata/ettin_tokenizer_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_ettin_tokenizer.py")
	}
	var gld struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
			Bare []int32 `json:"bare"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &gld); err != nil {
		t.Fatal(err)
	}

	eq := func(a, b []int32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}

	// maxLen above the longest case (the 2500× repeat is ~15k tokens), so
	// EncodeWithSpecials never trims and the comparison stays exact.
	const maxLen = 1 << 20
	var bad int
	for _, c := range gld.Cases {
		gotSpecial, err := tok.EncodeWithSpecials(c.Text, maxLen)
		if err != nil {
			t.Fatalf("%q: EncodeWithSpecials: %v", c.Text, err)
		}
		if !eq(gotSpecial, c.IDs) {
			bad++
			t.Errorf("wrapped mismatch %q:\n  got  %v\n  want %v", shortCase(c.Text), gotSpecial, c.IDs)
		}
		if gotBare := tok.Encode(c.Text); !eq(gotBare, c.Bare) {
			bad++
			t.Errorf("bare mismatch %q:\n  got  %v\n  want %v", shortCase(c.Text), gotBare, c.Bare)
		}
	}
	if bad == 0 {
		t.Logf("byte-level BPE id-exact vs HF tokenizers over %d cases (wrapped + bare)", len(gld.Cases))
	}
}

// shortCase keeps a failure message readable when the case is the 100 KB repeat.
func shortCase(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
}

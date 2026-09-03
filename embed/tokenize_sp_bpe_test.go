package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// TestSPBPE_parity certifies the SentencePiece-style BPE tokenizer
// (hotchpotch/bekko-embedding-v1-a8m: Metaspace pre-tokenizer + byte-fallback +
// TemplateProcessing specials) against the real HF `tokenizers` pipeline
// (scripts/oracle/pin_bekko_tokenizer.py). It is the gate that proves aikit's
// spBPEBackend reproduces HF id-for-id — including the byte-fallback <0xNN>
// decomposition and the Metaspace handling of leading/trailing spaces — and that
// the loader dispatches this tokenizer.json to spBPEBackend rather than the
// GPT-2 byte-level bpeBackend (which would silently mis-tokenize it).
func TestSPBPE_parity(t *testing.T) {
	const tokJSON = "../testdata/bekko-embedding-v1-a8m/tokenizer.json"
	if _, err := os.Stat(tokJSON); err != nil {
		t.Skipf("no bekko tokenizer at %s — fetch the model first", tokJSON)
	}
	tok, err := LoadTokenizer(tokJSON)
	if err != nil {
		t.Fatal(err)
	}
	if tok.spbpe == nil {
		t.Fatalf("tokenizer.json dispatched to %T, want *spBPEBackend (Metaspace BPE)", tok)
	}

	raw, err := os.ReadFile("../testdata/bekko_tokenizer_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_bekko_tokenizer.py")
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

	var mismatchSpecial, mismatchBare int
	for _, c := range gld.Cases {
		gotSpecial, err := tok.EncodeWithSpecials(c.Text, 8192)
		if err != nil {
			t.Fatalf("%q: EncodeWithSpecials: %v", c.Text, err)
		}
		if !eq(gotSpecial, c.IDs) {
			mismatchSpecial++
			t.Errorf("wrapped mismatch %q:\n  got  %v\n  want %v", c.Text, gotSpecial, c.IDs)
		}
		gotBare := tok.Encode(c.Text)
		if !eq(gotBare, c.Bare) {
			mismatchBare++
			t.Errorf("bare mismatch %q:\n  got  %v\n  want %v", c.Text, gotBare, c.Bare)
		}
	}
	if mismatchSpecial == 0 && mismatchBare == 0 {
		t.Logf("SP-BPE id-exact vs HF tokenizers over %d cases (wrapped + bare)", len(gld.Cases))
	}
}

// TestSPBPE_rejectsUnsupportedPreTokenizer prevents a tokenizer from loading
// successfully when its pre-tokenizer would be executed differently from the
// bare Metaspace(always, split=true) pipeline implemented by encodeSegment.
// Silently accepting either case produces plausible but incorrect token IDs.
func TestSPBPE_rejectsUnsupportedPreTokenizer(t *testing.T) {
	tests := []struct {
		name string
		json string
	}{
		{
			name: "sequence with ignored component",
			json: `{"type":"Sequence","pretokenizers":[` +
				`{"type":"WhitespaceSplit"},` +
				`{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}`,
		},
		{
			name: "unsupported prepend scheme",
			json: `{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true}`,
		},
		{
			name: "unsupported split mode",
			json: `{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":false}`,
		},
		{
			name: "unsupported replacement",
			json: `{"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := spBpePreTok(json.RawMessage(tt.json)); err == nil {
				t.Fatal("spBpePreTok accepted a pipeline that encodeSegment does not implement")
			}
		})
	}

	// This is the one configuration metaspaceBare implements and must remain
	// accepted for bekko-embedding-v1-a8m.
	supported := json.RawMessage(`{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}`)
	if err := spBpePreTok(supported); err != nil {
		t.Fatalf("supported bare Metaspace rejected: %v", err)
	}
}

func TestSPBPE_addedTokenStrip(t *testing.T) {
	tok, err := parseSPBPETokenizer([]byte(`{
		"added_tokens":[
			{"id":4,"content":"<left>","lstrip":true},
			{"id":5,"content":"<right>","rstrip":true}
		],
		"normalizer":null,
		"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},
		"model":{"type":"BPE","unk_token":"<unk>","vocab":{"<unk>":3,"▁":10},"merges":[],"byte_fallback":false,"fuse_unk":true},
		"post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{}}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		text string
		want int32
	}{
		{" <left>", 4},
		{"<right> ", 5},
	} {
		got := tok.encode(tc.text)
		if len(got) != 1 || got[0] != tc.want {
			t.Errorf("encode(%q) = %v, want [%d]; added-token strip flag was ignored", tc.text, got, tc.want)
		}
	}
}

func TestSPBPE_rejectsUnsupportedPostProcessor(t *testing.T) {
	_, err := parseSPBPETokenizer([]byte(`{
		"normalizer":null,
		"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},
		"model":{"type":"BPE","unk_token":"<unk>","vocab":{"<unk>":3,"▁":10},"merges":[]},
		"post_processor":{"type":"ByteLevel"}
	}`))
	if err == nil {
		t.Fatal("SP-BPE accepted a post-processor it does not implement")
	}
}

package embed

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const (
	spmModelDir = "../testdata/gliner-multi-v2.1"
	spmGolden   = "../testdata/gliner_tokenizer_golden.json"
)

type spmGoldenFile struct {
	VocabSize   int              `json:"vocab_size"`
	AddedTokens map[string]int32 `json:"added_tokens"`
	Cases       []struct {
		Text    string   `json:"text"`
		Bare    []int32  `json:"bare"`
		Wrapped []int32  `json:"wrapped"`
		Pieces  []string `json:"pieces"`
	} `json:"cases"`
	WordCases []struct {
		Words       []string `json:"words"`
		IDs         []int32  `json:"ids"`
		FirstSubtok []int32  `json:"first_subtok"`
	} `json:"word_cases"`
}

// loadSPMFixture loads the GLiNER tokenizer and its oracle, skipping when either
// is absent (the checkpoint files are gitignored; CI without them stays green).
func loadSPMFixture(t *testing.T) (*Tokenizer, *spmGoldenFile) {
	t.Helper()
	spmPath := filepath.Join(spmModelDir, "spm.model")
	if _, err := os.Stat(spmPath); err != nil {
		t.Skipf("no spm.model at %s — fetch it from the GLiNER mirror", spmModelDir)
	}
	raw, err := os.ReadFile(spmGolden)
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_gliner_tokenizer.py")
	}
	var g spmGoldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizerSPM(spmPath, filepath.Join(spmModelDir, "added_tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	return tok, &g
}

// TestSPMTokenizer_parity certifies the raw-spm.model reader (tokenize_spm.go)
// against sentencepiece itself (scripts/oracle/pin_gliner_tokenizer.py) for the GLiNER /
// mDeBERTa-v3 tokenizer. Id-exact — the bar the other tokenizer backends hold.
//
// This path is configured entirely from the ModelProto (there is no tokenizer.json
// in the checkpoint), so the gate is really over four decisions the reader makes:
// the dummy prefix + whitespace collapsing, the single-chunk Viterbi, byte_fallback,
// and the piece-TYPE filter that keeps control/byte pieces out of the search.
func TestSPMTokenizer_parity(t *testing.T) {
	tok, g := loadSPMFixture(t)

	// 250101 spm pieces + the 4 added tokens = the 250105 rows of the GLiNER
	// embedding table. If this drifts, every id above the spm vocab is suspect.
	if got, want := tok.VocabSize(), g.VocabSize+len(g.AddedTokens); got != want {
		t.Errorf("VocabSize = %d, want %d (%d spm pieces + %d added)",
			got, want, g.VocabSize, len(g.AddedTokens))
	}

	for _, c := range g.Cases {
		if got := tok.Encode(c.Text); !slices.Equal(got, c.Bare) {
			t.Errorf("Encode(%q)\n got %v\nwant %v\npieces %v", c.Text, got, c.Bare, c.Pieces)
		}
		got, err := tok.EncodeWithSpecials(c.Text, 512)
		if err != nil {
			t.Fatalf("EncodeWithSpecials(%q): %v", c.Text, err)
		}
		if !slices.Equal(got, c.Wrapped) {
			t.Errorf("EncodeWithSpecials(%q)\n got %v\nwant %v", c.Text, got, c.Wrapped)
		}
	}
}

// TestSPMTokenizer_encodeWords gates the word-level path GLiNER's "first"
// sub-token pooling needs. The alignment contract is the point: a word that
// produces no sub-tokens must report -1, not the next word's start.
func TestSPMTokenizer_encodeWords(t *testing.T) {
	tok, g := loadSPMFixture(t)

	for _, wc := range g.WordCases {
		ids, first := tok.EncodeWords(wc.Words)
		if !slices.Equal(ids, wc.IDs) {
			t.Errorf("EncodeWords(%q) ids\n got %v\nwant %v", wc.Words, ids, wc.IDs)
		}
		if !slices.Equal(first, wc.FirstSubtok) {
			t.Errorf("EncodeWords(%q) firstSubtok\n got %v\nwant %v", wc.Words, first, wc.FirstSubtok)
		}
		// Every non-empty word must index a real token.
		for i, f := range first {
			if f >= 0 && int(f) >= len(ids) {
				t.Errorf("EncodeWords(%q) firstSubtok[%d] = %d out of range (%d ids)",
					wc.Words, i, f, len(ids))
			}
		}
	}
}

// TestSPMTokenizer_pieceTypes pins the piece-TYPE filter directly rather than
// leaving it implicit in the parity corpus. sentencepiece searches only NORMAL,
// USER_DEFINED and UNUSED pieces; CONTROL ([CLS]/[SEP]/[PAD]), UNKNOWN ([UNK]) and
// BYTE (<0xNN>) pieces carry real spellings that text can contain, and matching
// them would emit ids sentencepiece never emits from text.
func TestSPMTokenizer_pieceTypes(t *testing.T) {
	tok, _ := loadSPMFixture(t)

	for _, literal := range []string{"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<0x41>"} {
		ids := tok.Encode(literal)
		if len(ids) == 1 {
			t.Errorf("Encode(%q) = %v — collapsed to a single reserved id; the "+
				"piece-type filter is not excluding CONTROL/UNKNOWN/BYTE pieces", literal, ids)
		}
		for _, id := range ids {
			if id <= 3 {
				t.Errorf("Encode(%q) = %v contains reserved id %d", literal, ids, id)
			}
		}
	}
}

// TestSPMTokenizer_breakItFirst proves the gate is load-bearing: perturb each of
// the three ModelProto-derived normalization decisions and parity must go red. A
// gate that stays green under these is testing nothing.
func TestSPMTokenizer_breakItFirst(t *testing.T) {
	_, g := loadSPMFixture(t)
	spmPath := filepath.Join(spmModelDir, "spm.model")
	addedPath := filepath.Join(spmModelDir, "added_tokens.json")

	perturb := map[string]func(*unigramBackend){
		"no dummy prefix":            func(u *unigramBackend) { u.spmAddDummyPrefix = false },
		"keep extra whitespace":      func(u *unigramBackend) { u.spmRemoveExtraWS = false },
		"whitespace-split metaspace": func(u *unigramBackend) { u.metaspace = metaspaceKindWhitespaceSplit },
	}
	for name, broke := range perturb {
		t.Run(name, func(t *testing.T) {
			tok, err := LoadTokenizerSPM(spmPath, addedPath)
			if err != nil {
				t.Fatal(err)
			}
			broke(tok.uni)
			for _, c := range g.Cases {
				if !slices.Equal(tok.Encode(c.Text), c.Bare) {
					return // diverged, as it must
				}
			}
			t.Errorf("%s: still id-exact over %d cases — the gate does not constrain this", name, len(g.Cases))
		})
	}
}

package main

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/ner"
)

// The model + golden are shared with the ner package's parity test; this
// example owns the masked-text half of the contract, since masking lives here
// (the secret-specific presentation), not in ner.
const (
	modelDir   = "../../testdata/distilbert-secret-masker-v3.3a-rs"
	goldenPath = "../../testdata/tokenclassification_golden.json"
)

type goldenFile struct {
	Mask  string `json:"mask"`
	Cases []struct {
		Text   string `json:"text"`
		Masked string `json:"masked"`
	} `json:"cases"`
}

// TestSecretMasker_maskedParity: for every golden case, running the model and
// replacing the reported spans must reproduce the reference's mask_text
// output byte-for-byte.
func TestSecretMasker_maskedParity(t *testing.T) {
	if _, err := os.Stat(modelDir + "/model.safetensors"); err != nil {
		t.Skipf("no distilbert-secret-masker at %s — uvx --from huggingface_hub hf download "+
			"AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs", modelDir)
	}
	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_tokenclassification.py")
	}
	var g goldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}

	m, err := ner.LoadTokenClassifier(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })

	for ci, c := range g.Cases {
		spans, err := m.Predict(c.Text, ner.TokenOpts{})
		if err != nil {
			t.Fatalf("case %d: Predict: %v", ci, err)
		}
		if got := maskSpans(c.Text, spans, g.Mask); got != c.Masked {
			t.Errorf("case %d masked mismatch:\n got %q\nwant %q", ci, got, c.Masked)
		}
	}
}

// TestMaskSpans_splice pins the replacement on an input where the spans are
// neither adjacent nor left-to-right in application order: the right-to-left
// splice is what keeps the not-yet-applied byte offsets valid.
func TestMaskSpans_splice(t *testing.T) {
	text := "key=AAA bb key=CCC"
	spans := []ner.TokenEntity{
		{Text: "AAA", Start: 4, End: 7},
		{Text: "CCC", Start: 15, End: 18},
	}
	want := "key=***MASKED*** bb key=***MASKED***"
	if got := maskSpans(text, spans, defaultMask); got != want {
		t.Errorf("masked = %q, want %q", got, want)
	}
	// Empty mask deletes the spans outright.
	if got := maskSpans(text, spans, ""); got != "key= bb key=" {
		t.Errorf("delete mask = %q", got)
	}
	if got := maskSpans(text, nil, defaultMask); got != text {
		t.Errorf("no spans: %q", got)
	}
}

// TestLineOf pins the reference's line convention on multi-line input.
func TestLineOf(t *testing.T) {
	text := "first\nsecond\nthird=SECRET"
	if got := lineOf(text, 0); got != 1 {
		t.Errorf("line at 0 = %d, want 1", got)
	}
	if got := lineOf(text, len(text)-6); got != 3 {
		t.Errorf("line at SECRET = %d, want 3", got)
	}
}

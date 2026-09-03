package main

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
)

// These parity tests are the black-box ports of the encoder-package ModernBERT
// gates (encoder/modernbert_ruri_test.go, modernbert_ettin_test.go): everything
// expressible through the public API — embedding cosine vs the HF golden, the
// full text→embedding pipeline, and the reranker's live Score leg. The
// white-box break-it-first gates (RoPE theta, sliding window, pooling override)
// need unexported state and stay in encoder.

func cos32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// repoPath resolves one of the ./testdata/… constants for the context the test
// runs in: `go run .` uses them as-is from the repo root, but `go test` executes
// from this package directory, where the same layout is ../../testdata.
func repoPath(p string) string {
	if strings.HasPrefix(p, "./testdata/") {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		return "../.." + p[1:]
	}
	return p
}

func golden(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(repoPath(path))
	if err != nil {
		t.Skipf("no golden %s — run scripts/oracle/pin_*.py", path)
	}
	return raw
}

func checkpoint(t *testing.T, dir string) {
	t.Helper()
	if _, err := os.Stat(repoPath(dir) + "/model.safetensors"); err != nil {
		t.Skipf("no checkpoint at %s — fetch it first", dir)
	}
}

// TestRuriParity pins ruri-v3-30m against the HF golden through the public
// surface: Embed on the golden's exact input_ids isolates the forward, and
// Encode re-runs aikit's own tokenizer, tying the two into one end-to-end
// check against sentence-transformers.
func TestRuriParity(t *testing.T) {
	checkpoint(t, ruriDir)
	m, err := encoder.LoadModernBERT(repoPath(ruriDir))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	var gld struct {
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			L        int       `json:"L"`
			Hidden   []float32 `json:"hidden"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(golden(t, "./testdata/ruri_golden.json"), &gld); err != nil {
		t.Fatal(err)
	}

	tok, err := embed.LoadTokenizer(repoPath(ruriDir) + "/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range gld.Cases {
		if cos := cos32(m.Embed(c.InputIDs), c.Emb); cos < 0.9999 {
			t.Errorf("%q: Embed cosine %.6f < 0.9999", c.Text, cos)
		}
		ids, err := tok.EncodeWithSpecials(c.Text, 1024)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != len(c.InputIDs) {
			t.Errorf("%q: tokenizer produced %d ids, golden has %d", c.Text, len(ids), len(c.InputIDs))
		} else {
			for i := range ids {
				if ids[i] != c.InputIDs[i] {
					t.Errorf("%q: tokenizer mismatch at %d: got %d want %d", c.Text, i, ids[i], c.InputIDs[i])
					break
				}
			}
		}
		emb, err := m.Encode(c.Text)
		if err != nil {
			t.Fatal(err)
		}
		if cos := cos32(emb, c.Emb); cos < 0.9999 {
			t.Errorf("%q: Encode cosine %.6f < 0.9999", c.Text, cos)
		}
	}
}

// TestBekkoParity pins hotchpotch/bekko-embedding-v1-a8m — the second ModernBERT
// spelling (sans_pos, equal local/global RoPE thetas) — against its HF golden
// through the same public surface as TestRuriParity.
func TestBekkoParity(t *testing.T) {
	checkpoint(t, bekkoDir)
	m, err := encoder.LoadModernBERT(repoPath(bekkoDir))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	var gld struct {
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			L        int       `json:"L"`
			Hidden   []float32 `json:"hidden"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(golden(t, "./testdata/bekko_golden.json"), &gld); err != nil {
		t.Fatal(err)
	}

	tok, err := embed.LoadTokenizer(repoPath(bekkoDir) + "/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range gld.Cases {
		if cos := cos32(m.Embed(c.InputIDs), c.Emb); cos < 0.9999 {
			t.Errorf("%q: Embed cosine %.6f < 0.9999", c.Text, cos)
		}
		ids, err := tok.EncodeWithSpecials(c.Text, 1024)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != len(c.InputIDs) {
			t.Errorf("%q: tokenizer produced %d ids, golden has %d", c.Text, len(ids), len(c.InputIDs))
		} else {
			for i := range ids {
				if ids[i] != c.InputIDs[i] {
					t.Errorf("%q: tokenizer mismatch at %d: got %d want %d", c.Text, i, ids[i], c.InputIDs[i])
					break
				}
			}
		}
		emb, err := m.Encode(c.Text)
		if err != nil {
			t.Fatal(err)
		}
		if cos := cos32(emb, c.Emb); cos < 0.9999 {
			t.Errorf("%q: Encode cosine %.6f < 0.9999", c.Text, cos)
		}
	}
}

// TestEttinRerankerParity pins ettin-reranker-17m's live Score leg against the
// Python golden: aikit's own pair framing (byte-level BPE tokenizer included)
// must reproduce the reference logits.
func TestEttinRerankerParity(t *testing.T) {
	checkpoint(t, ettinDir)
	rr, err := encoder.LoadModernBERTCrossEncoder(repoPath(ettinDir))
	if err != nil {
		t.Fatal(err)
	}
	defer rr.Close()

	var gld struct {
		Cases []struct {
			Query string    `json:"query"`
			Doc   string    `json:"doc"`
			Score []float32 `json:"score"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(golden(t, "./testdata/ettin_reranker_golden.json"), &gld); err != nil {
		t.Fatal(err)
	}
	for _, c := range gld.Cases {
		got, err := rr.Score(c.Query, c.Doc)
		if err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(float64(got) - float64(c.Score[0])); d > 1e-4 {
			t.Errorf("score(%q | %q) = %v, golden %v (max|Δ| %.2e)", c.Query, c.Doc, got, c.Score[0], d)
		}
	}
}

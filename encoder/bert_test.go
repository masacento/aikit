package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestBERT_parity pins the MiniLM forward (§2.2) against the sentence-transformers
// golden (scripts/oracle/pin_minilm.py): feeding the golden's input_ids, the Go forward's
// last_hidden_state must match per-element and its mean-pooled L2-normalized
// embedding must match by cosine. Model-gated (skips without testdata/minilm-model).
func TestBERT_parity(t *testing.T) {
	const dir = "../testdata/minilm-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no MiniLM model at %s — fetch via scripts/README.md", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/minilm_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Hidden int `json:"hidden"`
		Cases  []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			L        int       `json:"L"`
			HiddenSt []float32 `json:"hidden"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}

	var worstHidden, worstCos float64 = 0, 1
	for _, c := range g.Cases {
		h := b.hiddenStates(c.InputIDs, nil)
		if len(h) != len(c.HiddenSt) {
			t.Fatalf("%q: hidden len %d != golden %d", c.Text, len(h), len(c.HiddenSt))
		}
		var maxd float64
		for i := range h {
			if d := math.Abs(float64(h[i]) - float64(c.HiddenSt[i])); d > maxd {
				maxd = d
			}
		}
		cos := cos32(b.Embed(c.InputIDs), c.Emb)
		if maxd > worstHidden {
			worstHidden = maxd
		}
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-34q L=%2d  hidden maxΔ %.2e  emb cosine %.6f", c.Text, c.L, maxd, cos)

		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (forward bug?)", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: embedding cosine %.6f < 0.9999", c.Text, cos)
		}
	}
	t.Logf("MiniLM parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f", len(g.Cases), worstHidden, worstCos)
}

// TestBERT_encodeEndToEnd pins the full text→embedding pipeline: aikit's WordPiece
// tokenizer must produce the same input_ids as HF, and BERT.Encode(text) must match
// the golden sentence embedding.
func TestBERT_encodeEndToEnd(t *testing.T) {
	const dir = "../testdata/minilm-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no MiniLM model at %s", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/minilm_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	tokMismatch := 0
	for _, c := range g.Cases {
		ids, err := b.tok.EncodeWithSpecials(c.Text, b.maxSeq)
		if err != nil {
			t.Fatal(err)
		}
		same := len(ids) == len(c.InputIDs)
		for i := range ids {
			if same && ids[i] != c.InputIDs[i] {
				same = false
			}
		}
		if !same {
			tokMismatch++
			t.Logf("tokenizer mismatch %q: got %v want %v", c.Text, ids, c.InputIDs)
		}
		emb, err := b.Encode(c.Text)
		if err != nil {
			t.Fatal(err)
		}
		if cos := cos32(emb, c.Emb); cos < 0.9999 {
			t.Errorf("%q: Encode cosine %.6f < 0.9999", c.Text, cos)
		}
	}
	if tokMismatch > 0 {
		t.Errorf("%d/%d cases: aikit WordPiece ids differ from the HF golden", tokMismatch, len(g.Cases))
	}
}

// TestCLSHiddenState_matchesFullRow0 is §4.1's gate: trimming the last layer to
// row 0 must give bit-for-bit what the full forward's row 0 gives.
//
// Exact equality, not a tolerance. The claim rests on M-invariance — the blocked
// matmul's dst[i,j] reduces over k-tiles fixed by kBlock and K, with M choosing
// only the m-block boundary — and a tolerance would accept a reassociated sum,
// which is precisely the failure this has to exclude. Everything else the layer
// does (addBias, gelu, residual, layerNorm, softmaxRows) is per-row already.
//
// Both segment shapes are covered because the cross-encoder passes two and the
// embedder passes none, and the segment embedding is added before the trunk.
func TestCLSHiddenState_matchesFullRow0(t *testing.T) {
	b := loadTestBERT(t)
	D := b.cfg.Hidden

	for _, n := range []int{1, 2, 7, 64, 200} {
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32((i*7919 + 13) % 1000)
		}
		for _, withSegs := range []bool{false, true} {
			var segs []int32
			if withSegs {
				segs = make([]int32, n)
				for i := range segs {
					if i > n/2 {
						segs[i] = 1
					}
				}
			}
			full := b.hiddenStates(ids, segs)
			trimmed := b.clsHiddenState(ids, segs)
			if len(trimmed) != D {
				t.Fatalf("n=%d segs=%v: trimmed length %d, want %d", n, withSegs, len(trimmed), D)
			}
			for j := range D {
				if trimmed[j] != full[j] {
					t.Fatalf("n=%d segs=%v component %d: trimmed %v, full row 0 %v",
						n, withSegs, j, trimmed[j], full[j])
				}
			}
		}
	}
}

// loadTestBERT opens whichever BERT-family checkpoint this machine has.
//
// It tries several rather than pinning one because the fixtures differ per box —
// the amd64 machine has the cross-encoder and SPLADE checkpoints but not MiniLM —
// and a structural test that skips everywhere is worth nothing. Any BERT trunk
// exercises the same forward.
func loadTestBERT(tb testing.TB) *BERT {
	tb.Helper()
	var tried []string
	for _, dir := range []string{
		"../testdata/minilm-model",
		"../testdata/crossencoder-model",
		"../testdata/splade-model",
	} {
		if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
			tried = append(tried, dir)
			continue
		}
		b, err := LoadBERT(dir)
		if err != nil {
			tb.Fatal(err)
		}
		return b
	}
	tb.Skipf("no BERT checkpoint in any of %v — fetch via scripts/README.md", tried)
	return nil
}

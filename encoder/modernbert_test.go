package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestModernBERT_parity certifies the ModernBERT forward
// (hotchpotch/bekko-embedding-v1-a8m, a ModernBERT-base fine-tune) against its HF
// reference (scripts/oracle/pin_bekko.py). ModernBERT is a pre-norm, bias-free
// BERT-family encoder with four axes the other paths don't exercise together:
// sans_pos RoPE, alternating local (bidirectional sliding window) / global
// attention, a final LayerNorm, and a GeGLU MLP that activates the first Wi
// chunk. Two break-it-first gates prove the load-bearing bits: CLS pooling must
// diverge from the declared MEAN golden, and disabling the sliding window (forcing
// local layers global) must diverge on a long-enough input.
func TestModernBERT_parity(t *testing.T) {
	const dir = "../testdata/bekko-embedding-v1-a8m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no bekko ModernBERT at %s — fetch + run scripts/oracle/pin_bekko.py", dir)
	}
	m, err := LoadModernBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.pool != poolMean {
		t.Fatalf("ModernBERT pool = %q, want mean (from 1_Pooling/config.json)", m.pool)
	}

	raw, err := os.ReadFile("../testdata/bekko_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_bekko.py")
	}
	var gld struct {
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			L        int       `json:"L"`
			HiddenSt []float32 `json:"hidden"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &gld); err != nil {
		t.Fatal(err)
	}
	D := m.cfg.Hidden

	var worstHidden, worstCos float64 = 0, 1
	var brokePooling, brokeWindow bool
	for _, c := range gld.Cases {
		h := m.forward(c.InputIDs)
		if len(c.HiddenSt) > 0 {
			if len(h) != len(c.HiddenSt) {
				t.Fatalf("%q: hidden len %d != golden %d", c.Text, len(h), len(c.HiddenSt))
			}
			var maxd float64
			for i := range h {
				if d := math.Abs(float64(h[i]) - float64(c.HiddenSt[i])); d > maxd {
					maxd = d
				}
			}
			if maxd > worstHidden {
				worstHidden = maxd
			}
			if maxd > 5e-3 {
				t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (RoPE/GeGLU/local-window/final-norm bug?)", c.Text, maxd)
			}
			t.Logf("%-44q L=%3d  hidden maxΔ %.2e", c.Text, c.L, maxd)
		}

		cos := cos32(m.Embed(c.InputIDs), c.Emb)
		if cos < worstCos {
			worstCos = cos
		}
		if cos < 0.9999 {
			t.Errorf("%q: mean embedding cosine %.6f < 0.9999", c.Text, cos)
		}

		// Break-it-first #1: CLS pooling instead of mean must diverge on a
		// non-degenerate case (proves pooling is load-bearing).
		if c.L > 2 {
			clsCos := cos32(l2norm(poolOne(h, len(h)/D, D, poolCLS)), c.Emb)
			if clsCos < 0.99 {
				brokePooling = true
			}
			if clsCos >= cos {
				t.Errorf("%q: CLS cosine %.6f >= mean %.6f — pooling gate can't tell them apart", c.Text, clsCos, cos)
			}
		}

		// Break-it-first #2: forcing local layers to attend globally (window wider
		// than the sequence) must diverge on an input longer than the window — the
		// only case where |i-j| > S actually masks something.
		if c.L > 2*m.slidingW+1 && len(c.HiddenSt) > 0 {
			saved := m.slidingW
			m.slidingW = c.L + 1 // |i-j| ≤ L always true ⇒ no masking ⇒ global
			globalCos := cos32(m.Embed(c.InputIDs), c.Emb)
			m.slidingW = saved
			if globalCos < 0.99 {
				brokeWindow = true
			}
			if globalCos >= cos {
				t.Errorf("%q: global-attn cosine %.6f >= windowed %.6f — sliding-window gate can't tell them apart", c.Text, globalCos, cos)
			}
		}
	}
	if !brokePooling {
		t.Error("break-it-first vacuous: CLS pooling never diverged from the mean golden")
	}
	if !brokeWindow {
		t.Error("break-it-first vacuous: disabling the sliding window never diverged on a long input")
	}
	t.Logf("ModernBERT parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f",
		len(gld.Cases), worstHidden, worstCos)
}

// TestModernBERT_encodeEndToEnd certifies the full text→embedding pipeline: aikit's
// SentencePiece-BPE tokenizer must produce the same input_ids as HF, and
// Encode(text) must match the mean-pooled golden embedding. It reuses the forward
// golden (whose input_ids are the HF tokenization), so it ties the tokenizer parity
// (embed/tokenize_sp_bpe_test.go) and the forward parity above into one end-to-end
// check against sentence-transformers.
func TestModernBERT_encodeEndToEnd(t *testing.T) {
	const dir = "../testdata/bekko-embedding-v1-a8m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no bekko ModernBERT at %s", dir)
	}
	m, err := LoadModernBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.tok == nil {
		t.Fatal("ModernBERT tokenizer not loaded")
	}
	raw, err := os.ReadFile("../testdata/bekko_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_bekko.py")
	}
	var gld struct {
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			Emb      []float32 `json:"embedding"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &gld); err != nil {
		t.Fatal(err)
	}
	tokMismatch := 0
	for _, c := range gld.Cases {
		ids, err := m.tok.EncodeWithSpecials(c.Text, m.maxSeq)
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
			t.Errorf("tokenizer mismatch %q: got %v want %v", c.Text, ids, c.InputIDs)
		}
		emb, err := m.Encode(c.Text)
		if err != nil {
			t.Fatal(err)
		}
		if cos := cos32(emb, c.Emb); cos < 0.9999 {
			t.Errorf("%q: Encode cosine %.6f < 0.9999", c.Text, cos)
		}
	}
	if tokMismatch > 0 {
		t.Errorf("%d/%d cases had tokenizer id mismatches", tokMismatch, len(gld.Cases))
	}
}

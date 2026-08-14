package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestGTE_parity certifies the GTE forward (Snowflake/snowflake-arctic-embed-m-v2.0,
// Alibaba gte-multilingual-base) against its HF reference (scripts/oracle/pin_gte.py). GTE
// is a post-norm BERT-family encoder with three axes the other paths don't exercise
// together: RoPE positions, a PACKED qkv projection, and a gated-GELU (GeGLU) MLP.
// Break-it-first pools MEAN instead of the declared CLS — it must diverge, proving
// the forward + pooling are load-bearing.
func TestGTE_parity(t *testing.T) {
	const dir = "../testdata/arctic2-m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no arctic-embed-m-v2.0 at %s — fetch + run scripts/oracle/pin_gte.py", dir)
	}
	g, err := LoadGTE(dir)
	if err != nil {
		t.Fatal(err)
	}
	if g.pool != poolCLS {
		t.Fatalf("GTE pool = %q, want cls (from 1_Pooling/config.json)", g.pool)
	}

	raw, err := os.ReadFile("../testdata/gte_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_gte.py")
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
	D := g.cfg.Hidden

	var worstHidden, worstCos float64 = 0, 1
	var brokeAtLeastOnce bool
	for _, c := range gld.Cases {
		h := g.hiddenStates(c.InputIDs)
		if len(h) != len(c.HiddenSt) {
			t.Fatalf("%q: hidden len %d != golden %d", c.Text, len(h), len(c.HiddenSt))
		}
		var maxd float64
		for i := range h {
			if d := math.Abs(float64(h[i]) - float64(c.HiddenSt[i])); d > maxd {
				maxd = d
			}
		}
		cos := cos32(g.Embed(c.InputIDs), c.Emb)
		if maxd > worstHidden {
			worstHidden = maxd
		}
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-34q L=%2d  hidden maxΔ %.2e  emb(cls) cosine %.6f", c.Text, c.L, maxd, cos)
		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (RoPE/GeGLU/attention bug?)", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: CLS embedding cosine %.6f < 0.9999", c.Text, cos)
		}

		// Break-it-first: mean pooling instead of CLS must diverge on a
		// non-degenerate case.
		if c.L > 2 {
			meanCos := cos32(l2norm(poolOne(h, len(h)/D, D, poolMean)), c.Emb)
			if meanCos < 0.99 {
				brokeAtLeastOnce = true
			}
			if meanCos >= cos {
				t.Errorf("%q: mean cosine %.6f >= CLS %.6f — gate can't tell them apart", c.Text, meanCos, cos)
			}
		}
	}
	if !brokeAtLeastOnce {
		t.Error("break-it-first vacuous: mean pooling never diverged from the CLS golden")
	}
	t.Logf("GTE parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f",
		len(gld.Cases), worstHidden, worstCos)
}

// TestGTE_encodeEndToEnd certifies the full text→embedding pipeline: aikit's
// SentencePiece/Unigram tokenizer must produce the same input_ids as HF, and
// Encode(text) must match the CLS-pooled golden.
func TestGTE_encodeEndToEnd(t *testing.T) {
	const dir = "../testdata/arctic2-m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no arctic-embed-m-v2.0 at %s", dir)
	}
	g, err := LoadGTE(dir)
	if err != nil {
		t.Fatal(err)
	}
	if g.tok == nil {
		t.Fatal("GTE tokenizer not loaded")
	}
	raw, err := os.ReadFile("../testdata/gte_golden.json")
	if err != nil {
		t.Skip("no golden")
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
		ids, err := g.tok.EncodeWithSpecials(c.Text, g.maxSeq)
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
		emb, err := g.Encode(c.Text)
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

// TestGTE_clsHiddenStateMatchesFullRow0 is the GTE half of lens §4.1's gate.
//
// It has one thing the BERT half does not: the trimmed path replaces the single
// packed [3D] QKV matmul with three [D] ones, so it asserts not only
// M-invariance but that splitting an output by columns leaves each element's
// reduction unchanged. If that failed, K and V would differ and row 0's
// attention with them — which is exactly what this compares.
func TestGTE_clsHiddenStateMatchesFullRow0(t *testing.T) {
	g := loadTestGTE(t)
	D := g.cfg.Hidden
	for _, n := range []int{1, 2, 9, 64, 200} {
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32((i*7919 + 13) % 1000)
		}
		full := g.hiddenStates(ids)
		trimmed := g.clsHiddenState(ids)
		if len(trimmed) != D {
			t.Fatalf("n=%d: trimmed length %d, want %d", n, len(trimmed), D)
		}
		for j := range D {
			if trimmed[j] != full[j] {
				t.Fatalf("n=%d component %d: trimmed %v, full row 0 %v", n, j, trimmed[j], full[j])
			}
		}
	}
}

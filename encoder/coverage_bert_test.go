package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// coverage_bert_test.go certifies the Bucket-A coverage-breadth embedders that
// reuse the already-built architectures (learned-absolute BERT + WordPiece, or a
// BERT-shaped model + Unigram). Each is config/tokenizer wiring plus a gate, not
// new architecture — so they share one parity harness. Goldens come from
// scripts/pin_coverage.py; the checkpoints are gitignored, so each gate skips
// unless the model is present locally.

// assertBERTParity is the shared oracle: load the model, verify its declared
// pooling + position offset are what the table claims, then match hidden states
// and the pooled embedding against the sentence-transformers golden. Break-it-first
// re-pools the OTHER way (mean↔CLS) and requires it to diverge, so the gate proves
// the declared pooling is load-bearing rather than incidental.
func assertBERTParity(t *testing.T, dir, goldenPath string, wantPool pooling, wantPosOff int) {
	t.Helper()
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no model at %s — fetch + run scripts/pin_coverage.py", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.pool != wantPool {
		t.Fatalf("%s pool = %q, want %q (from 1_Pooling/config.json)", dir, b.pool, wantPool)
	}
	if b.posOff != wantPosOff {
		t.Errorf("%s posOff = %d, want %d", dir, b.posOff, wantPosOff)
	}

	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skip("no golden — run scripts/pin_coverage.py")
	}
	var g struct {
		Cases []struct {
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
	D := b.cfg.Hidden

	// The pooling to use for break-it-first: the wrong one.
	wrongPool := poolMean
	if wantPool == poolMean {
		wrongPool = poolCLS
	}

	var worstHidden, worstCos float64 = 0, 1
	var brokeAtLeastOnce bool
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
		t.Logf("%-52q L=%2d  hidden maxΔ %.2e  emb cosine %.6f", c.Text, c.L, maxd, cos)
		if maxd > 5e-3 {
			t.Errorf("%q: hidden-state max |Δ| %.2e vs golden (forward bug?)", c.Text, maxd)
		}
		if cos < 0.9999 {
			t.Errorf("%q: embedding cosine %.6f < 0.9999", c.Text, cos)
		}

		// Break-it-first: the wrong pooling must materially diverge on a
		// non-degenerate case.
		if c.L > 2 {
			wrongCos := cos32(l2norm(poolOne(h, len(h)/D, D, wrongPool)), c.Emb)
			if wrongCos < 0.99 {
				brokeAtLeastOnce = true
			}
			if wrongCos >= cos {
				t.Errorf("%q: wrong-pooling cosine %.6f >= right %.6f — gate can't tell them apart", c.Text, wrongCos, cos)
			}
		}
	}
	if !brokeAtLeastOnce {
		t.Error("break-it-first vacuous: wrong pooling never diverged materially from the golden")
	}
	t.Logf("%s parity over %d cases: worst hidden maxΔ %.2e, worst emb cosine %.6f",
		dir, len(g.Cases), worstHidden, worstCos)
}

// TestBGELarge_parity — BAAI/bge-large-en-v1.5: the 1024-dim, 24-layer CLS-BERT.
// Same architecture as bge-small, at the large width, so it guards the loader on a
// bigger checkpoint.
func TestBGELarge_parity(t *testing.T) {
	assertBERTParity(t, "../testdata/bge-large", "../testdata/bge_large_golden.json", poolCLS, 0)
}

// TestMxbaiLarge_parity — mixedbread-ai/mxbai-embed-large-v1: a 1024-dim CLS-BERT,
// a different training lineage than BGE on the same architecture.
func TestMxbaiLarge_parity(t *testing.T) {
	assertBERTParity(t, "../testdata/mxbai-large", "../testdata/mxbai_golden.json", poolCLS, 0)
}

// TestArcticEmbedM_parity — Snowflake/snowflake-arctic-embed-m: a 768-dim CLS-BERT
// (arctic-embed 1.0; the 2.0 line is a different GTE/RoPE architecture, not this).
func TestArcticEmbedM_parity(t *testing.T) {
	assertBERTParity(t, "../testdata/arctic-embed-m", "../testdata/arctic_golden.json", poolCLS, 0)
}

// TestParaphraseMultilingual_parity — sentence-transformers/paraphrase-multilingual-
// MiniLM-L12-v2: a BERT-shaped model with the XLM-R Unigram vocab and MEAN pooling,
// position offset 0 (model_type=bert). Certifies the Unigram tokenizer on the
// BERT (not XLM-R) loader path, cross-script.
func TestParaphraseMultilingual_parity(t *testing.T) {
	assertBERTParity(t, "../testdata/paraphrase-ml", "../testdata/paraphrase_ml_golden.json", poolMean, 0)
}

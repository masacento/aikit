package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// distilbertGoldenDir / tokenclassificationGolden are the pinned
// AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs checkpoint and the golden
// scripts/oracle/pin_tokenclassification.py produces. The trunk block (case 0) is THIS
// package's oracle — hidden states for a known id sequence — while the ner
// package's parity test owns spans/decode. Keeping both readers on one file
// means a red ner gate can be bisected: trunk green + spans red points at
// head/decode/tokenization, not at the forward.
const (
	distilbertGoldenDir   = "../testdata/distilbert-secret-masker-v3.3a-rs"
	tokenclassGoldenP     = "../testdata/tokenclassification_golden.json"
	distilbertHiddenTol   = 7.5e-4 // the backbone float32 noise floor (TestDeBERTa_parity)
	distilbertExtremesTol = 2e-3   // min/max are single extreme values, where drift peaks
)

func loadDistilBERTTrunkGolden(t *testing.T) (*DistilBERT, struct {
	IDs     []int32   `json:"ids"`
	Mean    float64   `json:"mean"`
	AbsMean float64   `json:"abs_mean"`
	Min     float64   `json:"min"`
	Max     float64   `json:"max"`
	Row0    []float64 `json:"row0"`
	Row1    []float64 `json:"row1"`
}) {
	t.Helper()
	if _, err := os.Stat(distilbertGoldenDir + "/model.safetensors"); err != nil {
		t.Skipf("no distilbert-secret-masker at %s — uvx --from huggingface_hub hf download "+
			"AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs", distilbertGoldenDir)
	}
	raw, err := os.ReadFile(tokenclassGoldenP)
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_tokenclassification.py")
	}
	var g struct {
		Cases []struct {
			Trunk *struct {
				IDs     []int32   `json:"ids"`
				Mean    float64   `json:"mean"`
				AbsMean float64   `json:"abs_mean"`
				Min     float64   `json:"min"`
				Max     float64   `json:"max"`
				Row0    []float64 `json:"row0"`
				Row1    []float64 `json:"row1"`
			} `json:"trunk"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Cases) == 0 || g.Cases[0].Trunk == nil {
		t.Skip("golden has no trunk block")
	}
	d, err := LoadDistilBERT(distilbertGoldenDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d, *g.Cases[0].Trunk
}

// TestDistilBERT_trunkGolden pins LoadDistilBERT + the shared bert.go forward
// against torch's DistilBertModel on the same ids: elementwise on sampled
// dims, plus reductions over the FULL [L, 768] tensor. The reductions make the
// sampling non-evasive — a drift hiding outside the 32 sampled dims still
// moves mean/abs_mean.
func TestDistilBERT_trunkGolden(t *testing.T) {
	d, want := loadDistilBERTTrunkGolden(t)

	h := d.HiddenStates(want.IDs)
	D := d.HiddenDim()
	if len(h) == 0 || len(h)%D != 0 {
		t.Fatalf("hidden states len %d not a multiple of hidden %d", len(h), D)
	}
	L := len(h) / D
	if L < 2 {
		t.Fatalf("trunk golden ran on %d tokens, need ≥2 for the row samples", L)

	}
	if got := d.MaxSeqLength(); got != 512 {
		t.Errorf("MaxSeqLength = %d, want 512", got)
	}

	var sum, absSum float64
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range h {
		f := float64(v)
		sum += f
		absSum += math.Abs(f)
		mn = math.Min(mn, f)
		mx = math.Max(mx, f)
	}
	n := float64(len(h))
	for name, pair := range map[string][2]float64{
		"mean":     {sum / n, want.Mean},
		"abs_mean": {absSum / n, want.AbsMean},
		"min":      {mn, want.Min},
		"max":      {mx, want.Max},
	} {
		tol := distilbertHiddenTol
		if name == "min" || name == "max" {
			tol = distilbertExtremesTol
		}
		if d := math.Abs(pair[0] - pair[1]); d > tol {
			t.Errorf("%s = %.6g, want %.6g (diff %.3g > %.g)", name, pair[0], pair[1], d, tol)
		}
	}

	for row, sample := range map[int][]float64{0: want.Row0, 1: want.Row1} {
		if len(sample) == 0 {
			continue
		}
		base := row * D
		if len(sample) > D {
			t.Fatalf("row %d sample longer than hidden", row)
		}
		for j, w := range sample {
			if d := math.Abs(float64(h[base+j]) - w); d > distilbertHiddenTol {
				t.Errorf("hidden[%d,%d] = %.6g, want %.6g (diff %.3g)", row, j, h[base+j], w, d)
			}
		}
	}
}

// TestDistilBERT_smoke checks the config mapping on the real checkpoint:
// DistilBERT's own key names (dim/n_layers/n_heads/hidden_dim) must land in
// the mapped config, and a wrong activation must be rejected — the forward
// silently assumes gelu.
func TestDistilBERT_smoke(t *testing.T) {
	if _, err := os.Stat(distilbertGoldenDir + "/model.safetensors"); err != nil {
		t.Skipf("no distilbert-secret-masker at %s", distilbertGoldenDir)
	}
	d, err := LoadDistilBERT(distilbertGoldenDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = d.Close() })
	if got := d.HiddenDim(); got != 768 {
		t.Errorf("HiddenDim = %d, want 768", got)
	}
	if got := d.trunk.cfg.Layers; got != 6 {
		t.Errorf("layers = %d, want 6", got)
	}
	if got := d.trunk.cfg.Heads; got != 12 {
		t.Errorf("heads = %d, want 12", got)
	}
	if got := d.trunk.cfg.Intermediate; got != 3072 {
		t.Errorf("intermediate = %d, want 3072", got)
	}
	// The forward runs and produces finite output on a wrapped sequence.
	h := d.HiddenStates([]int32{101, 22091, 2015, 3229, 102}) // [CLS] aws access key [SEP]
	if len(h) != 5*768 {
		t.Fatalf("hidden len = %d, want %d", len(h), 5*768)
	}
	for i, v := range h {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("hidden[%d] non-finite: %v", i, v)
		}
	}
}

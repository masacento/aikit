package encoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

const (
	debertaDir    = "../testdata/mdeberta-v3-base"
	debertaGolden = "../testdata/deberta_golden.json"
)

type debertaGoldenFile struct {
	HiddenSize int `json:"hidden_size"`
	NumLayers  int `json:"num_layers"`
	Cases      []struct {
		Text     string  `json:"text"`
		InputIDs []int32 `json:"input_ids"`
		L        int     `json:"L"`
		Rows     []int   `json:"rows"`
		Dims     []int   `json:"dims"`
		Layers   []struct {
			Sample []float32 `json:"sample"`
			Sum    float64   `json:"sum"`
			AbsSum float64   `json:"abs_sum"`
			Min    float64   `json:"min"`
			Max    float64   `json:"max"`
		} `json:"layers"`
	} `json:"cases"`
}

func TestDeBERTaDoesNotTruncatePastConfiguredPositionLength(t *testing.T) {
	// A zero-layer model is enough to exercise input embedding and sequence
	// sizing. DeBERTa uses relative positions, so MaxPos calibrates buckets but
	// must not silently cap the number of returned token states.
	d := &DeBERTa{
		cfg: debertaConfig{
			Hidden: 2, Heads: 1, Intermediate: 2, MaxPos: 2, LNEps: 1e-7,
		},
		wordEmb:   linalg.WrapF32([]float32{1, 0, 0, 1}, 2, 2),
		embLNW:    []float32{1, 1},
		embLNB:    []float32{0, 0},
		maxSeq:    2,
		attSpan:   4,
		maxRel:    8,
		relScale:  1,
		relBucket: buildRelBucketTable(2, 4, 8),
	}
	ids := []int32{0, 1, 0, 1, 0}
	if got := d.HiddenStates(ids); len(got) != len(ids)*d.HiddenDim() {
		t.Fatalf("HiddenStates returned %d floats for %d tokens, want %d", len(got), len(ids), len(ids)*d.HiddenDim())
	}
	buckets, center := d.relativeBuckets(len(ids))
	if len(buckets) != 2*len(ids)-1 || center != len(ids)-1 {
		t.Fatalf("extended bucket table = (len %d, center %d), want (%d, %d)", len(buckets), center, 2*len(ids)-1, len(ids)-1)
	}
}

func loadDeBERTaFixture(t *testing.T) (*DeBERTa, *debertaGoldenFile) {
	t.Helper()
	if _, err := os.Stat(debertaDir + "/model.safetensors"); err != nil {
		t.Skipf("no mdeberta-v3-base at %s — download microsoft/mdeberta-v3-base (see testdata/README.md)", debertaDir)
	}
	raw, err := os.ReadFile(debertaGolden)
	if err != nil {
		t.Skip("no golden — run scripts/pin_deberta.py")
	}
	var g debertaGoldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	d, err := LoadDeBERTa(debertaDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d, &g
}

// debertaCompare returns the worst per-layer sample deviation and the worst
// full-tensor reduction deviation across every case, reporting through report.
func debertaCompare(t *testing.T, d *DeBERTa, g *debertaGoldenFile,
	report func(layer int, kind string, got, want float64)) (worstSample, worstReduce float64) {
	t.Helper()
	D := d.HiddenDim()

	for _, c := range g.Cases {
		all := d.AllHiddenStates(c.InputIDs)
		if len(all) != len(c.Layers) {
			t.Fatalf("%q: got %d hidden states, want %d", c.Text, len(all), len(c.Layers))
		}
		for li, want := range c.Layers {
			h := all[li]
			if len(h) != c.L*D {
				t.Fatalf("%q layer %d: hidden len %d, want %d", c.Text, li, len(h), c.L*D)
			}
			k := 0
			for _, r := range c.Rows {
				for _, dim := range c.Dims {
					diff := math.Abs(float64(h[r*D+dim]) - float64(want.Sample[k]))
					if diff > worstSample {
						worstSample = diff
						report(li, "sample", float64(h[r*D+dim]), float64(want.Sample[k]))
					}
					k++
				}
			}
			var sum, absSum float64
			mn, mx := math.Inf(1), math.Inf(-1)
			for _, v := range h {
				f := float64(v)
				sum += f
				absSum += math.Abs(f)
				mn, mx = math.Min(mn, f), math.Max(mx, f)
			}
			// The reductions are over L*D terms, so their absolute error grows with
			// the tensor; compare them relative to the tensor's own L1 mass.
			scale := math.Max(want.AbsSum, 1)
			for _, p := range []struct {
				kind      string
				got, want float64
			}{
				{"sum", sum, want.Sum},
				{"abs_sum", absSum, want.AbsSum},
				{"min", mn, want.Min},
				{"max", mx, want.Max},
			} {
				rel := math.Abs(p.got-p.want) / scale
				if rel > worstReduce {
					worstReduce = rel
					report(li, p.kind, p.got, p.want)
				}
			}
		}
	}
	return worstSample, worstReduce
}

// TestDeBERTa_parity certifies the DeBERTa-v2 forward (microsoft/mdeberta-v3-base)
// against its HF reference PER LAYER (scripts/pin_deberta.py).
//
// Per layer, because this architecture's failure modes are silent and
// indistinguishable at the output: the log-bucket arithmetic, the p2c gather index,
// the 1/sqrt(3d) scale, and which tensor encoder.LayerNorm normalizes. A last-layer
// gate tells you the forward is wrong; this one tells you where it went wrong.
//
// The corpus deliberately includes sequences of 262, 482 and 512 tokens: below ~128
// every relative offset stays in the LINEAR bucket band, so a short-only corpus
// leaves make_log_bucket_position and the clamp untested.
//
// ON THE TOLERANCE. sampleTol is the 5e-3 hidden-state bar TestGTE_parity,
// TestNomicEmbed_parity and TestModernBERT_parity already hold. The measured worst
// here is 7.5e-4, at the last layer, and it is float32 rounding rather than
// headroom being spent: running the SAME reference in float64 shows torch's own
// fp32 diverging from its fp64 by 6.6e-4 at layer 11 and 1.0e-3 at layer 12 — more
// than this forward diverges from torch fp32. There is no structural error left to
// find below that floor. (The jump at layer 11 is the model, not the arithmetic:
// its activation RMS is 2.89 against ~0.4 everywhere else, so equal relative error
// shows up as a larger absolute one. TestDeBERTa_layerDrift prints the profile.)
func TestDeBERTa_parity(t *testing.T) {
	d, g := loadDeBERTaFixture(t)

	var wl int
	var wk string
	var wg, ww float64
	sample, reduce := debertaCompare(t, d, g, func(layer int, kind string, got, want float64) {
		wl, wk, wg, ww = layer, kind, got, want
	})

	const sampleTol, reduceTol = 5e-3, 5e-5
	if sample > sampleTol || reduce > reduceTol {
		t.Errorf("DeBERTa parity: worst sample diff %.3g (tol %.g), worst relative "+
			"reduction diff %.3g (tol %.g); worst was layer %d %s got %g want %g",
			sample, sampleTol, reduce, reduceTol, wl, wk, wg, ww)
	}
	t.Logf("worst sample diff %.3g, worst relative reduction diff %.3g", sample, reduce)
}

// TestDeBERTa_breakItFirst proves the parity gate constrains the four decisions that
// fail silently. Each perturbation must push the forward outside the tolerance; one
// that does not means the gate is not testing that decision at all.
func TestDeBERTa_breakItFirst(t *testing.T) {
	d, g := loadDeBERTaFixture(t)

	// A no-op report; break-it-first only cares whether the numbers move.
	nop := func(int, string, float64, float64) {}
	base, _ := debertaCompare(t, d, g, nop)
	if base > 5e-3 {
		t.Fatalf("baseline is already failing (%.3g) — fix parity before break-it-first", base)
	}

	perturb := map[string]func(*DeBERTa){
		// Drop the p2c term: the c2p half alone must not reproduce the reference.
		"no p2c": func(d *DeBERTa) { d.posAtt.p2c = false },
		// Drop the c2p term.
		"no c2p": func(d *DeBERTa) { d.posAtt.c2p = false },
		// The classic port bug: scale by 1/sqrt(d) instead of 1/sqrt(3d).
		"scale without scale_factor": func(d *DeBERTa) {
			d.relScale = math.Sqrt(float64(d.cfg.Hidden / d.cfg.Heads))
		},
		// Linear buckets everywhere — i.e. forget make_log_bucket_position. Only
		// observable past |i-j| > 128, which is why the corpus has long cases.
		"linear buckets": func(d *DeBERTa) {
			for i := range d.relBucket {
				d.relBucket[i] = int32(i - (d.maxSeq - 1))
			}
		},
	}
	for name, broke := range perturb {
		t.Run(name, func(t *testing.T) {
			victim, _ := loadDeBERTaFixture(t)
			broke(victim)
			got, _ := debertaCompare(t, victim, g, nop)
			if got <= 5e-3 {
				t.Errorf("%s: worst sample diff still %.3g — the gate does not "+
					"constrain this decision", name, got)
			}
		})
	}
}

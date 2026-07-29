package encoder

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/sparse"
)

// TestSPLADE_endToEnd shows the closed loop: SPLADE expansions feed the sparse
// inverted index directly, and a learned-sparse query ranks the relevant doc first
// — in-process, no Python. Model-gated.
func TestSPLADE_endToEnd(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatal(err)
	}
	docs := []string{
		"def parse_json(s):\n    return json.loads(s)",
		"train a deep neural network with gradient descent",
		"the cat sat quietly on the warm mat",
	}
	dvecs := make([]sparse.SparseVec, len(docs))
	for i, d := range docs {
		v, err := s.Expand(d)
		if err != nil {
			t.Fatal(err)
		}
		dvecs[i] = v
	}
	ix := sparse.New(dvecs)
	q, err := s.Expand("how to read and parse a json file in python")
	if err != nil {
		t.Fatal(err)
	}
	hits := ix.Query(q, len(docs))
	if len(hits) == 0 {
		t.Fatal("no hits")
	}
	t.Logf("SPLADE→sparse end-to-end: query ranks doc %d first (score %.3f)", hits[0].Index, hits[0].Score)
	if hits[0].Index != 0 {
		t.Errorf("expected the json doc (0) to rank first for a json query, got doc %d", hits[0].Index)
	}
}

// TestSPLADE_parity pins the SPLADE expansion (§2.3) against the Python reference
// (scripts/pin_splade.py): feeding the golden's input_ids, expandIDs must produce
// the same sparse term-weight vector (compared by cosine; term-set agreement logged).
// Model-gated.
func TestSPLADE_parity(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/splade_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		Vocab int `json:"vocab"`
		Cases []struct {
			Text     string    `json:"text"`
			InputIDs []int32   `json:"input_ids"`
			Terms    []uint32  `json:"terms"`
			Weights  []float32 `json:"weights"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	worstCos, worstJac := 1.0, 1.0
	for _, c := range g.Cases {
		v := s.expandIDs(c.InputIDs)
		got := map[uint32]float32{}
		for i, tm := range v.Terms {
			got[tm] = v.Weights[i]
		}
		want := map[uint32]float32{}
		for i, tm := range c.Terms {
			want[tm] = c.Weights[i]
		}
		union := map[uint32]bool{}
		for tm := range got {
			union[tm] = true
		}
		for tm := range want {
			union[tm] = true
		}
		var dot, ng, nw float64
		inter := 0
		for tm := range union {
			a, b := float64(got[tm]), float64(want[tm])
			dot += a * b
			ng += a * a
			nw += b * b
			if got[tm] > 0 && want[tm] > 0 {
				inter++
			}
		}
		cos := dot / (math.Sqrt(ng) * math.Sqrt(nw))
		jac := float64(inter) / float64(len(union))
		if cos < worstCos {
			worstCos = cos
		}
		if jac < worstJac {
			worstJac = jac
		}
		t.Logf("%-40q nnz go=%d py=%d  cosine %.6f  term-jaccard %.4f", c.Text, len(v.Terms), len(c.Terms), cos, jac)
		if cos < 0.999 {
			t.Errorf("%q: SPLADE cosine %.6f < 0.999", c.Text, cos)
		}
	}
	t.Logf("SPLADE parity: worst cosine %.6f, worst term-jaccard %.4f", worstCos, worstJac)
}

// TestSpladePooling_hoistedLog1pIsExact gates perf item 2's load-bearing claim:
// float32∘Log1p∘relu is monotone non-decreasing and maps 0→0, so taking the max of
// the RAW logits and applying log1p once per vocab entry is BIT-IDENTICAL to applying
// it per positive element and maxing the results.
//
// This is the whole justification for the rewrite, so it is asserted directly rather
// than inferred from an end-to-end cosine — which would hide a 1-ULP drift.
func TestSpladePooling_hoistedLog1pIsExact(t *testing.T) {
	const L, V = 64, 3000
	rng := rand.New(rand.NewSource(3))
	logits := make([]float32, L*V)
	for i := range L {
		for v := range V {
			switch {
			case v%5 == 0:
				// All-negative column: every row negative, so the max stays 0 and
				// log1p is never applied. This is the boundary the hoist relies on
				// (f(0)=0) and it must be present, or the equality is cheap.
				logits[i*V+v] = float32(-rng.ExpFloat64())
			case v%5 == 1 && i == 0:
				logits[i*V+v] = 0 // exactly zero
			default:
				logits[i*V+v] = float32(rng.NormFloat64() * 3)
			}
		}
	}

	// Old form: log1p per positive element, then max.
	want := make([]float32, V)
	for i := range L {
		for v, x := range logits[i*V : (i+1)*V] {
			if x > 0 {
				if w := float32(math.Log1p(float64(x))); w > want[v] {
					want[v] = w
				}
			}
		}
	}
	// New form: max the raw logits, then log1p once per vocab entry.
	got := make([]float32, V)
	for i := range L {
		for v, x := range logits[i*V : (i+1)*V] {
			if x > got[v] {
				got[v] = x
			}
		}
	}
	for v, x := range got {
		if x > 0 {
			got[v] = float32(math.Log1p(float64(x)))
		}
	}

	for v := range want {
		if got[v] != want[v] {
			t.Fatalf("vocab %d: hoisted %v != per-element %v (must be bit-identical)", v, got[v], want[v])
		}
	}
	// Vacuity: the fixture must contain all three regimes, or the equality is cheap.
	var pos, zero int
	for _, x := range want {
		if x > 0 {
			pos++
		} else {
			zero++
		}
	}
	if pos == 0 || zero == 0 {
		t.Errorf("fixture degenerate: %d positive, %d zero columns — need both", pos, zero)
	}
	t.Logf("bit-identical over %d vocab entries (%d positive, %d zero/negative-only)", V, pos, zero)
}

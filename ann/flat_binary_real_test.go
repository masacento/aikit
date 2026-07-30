package ann

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// realVecs embeds a clustered code-ish corpus with the Model2Vec checkpoint —
// the same generator recall_real_test.go uses, duplicated here rather than
// shared because that file is in package ann_test and this one needs the
// unexported builder.
//
// Skips without the per-machine model, so CI stays green on the synthetic gates.
func realVecs(t *testing.T) (vecs, queries [][]float32) {
	t.Helper()
	const modelDir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		t.Skipf("no model at %s — see testdata/README.md", modelDir)
	}
	m, err := embed.LoadFromFS(os.DirFS(modelDir), ".")
	if err != nil {
		t.Fatal(err)
	}
	verbs := []string{"get", "set", "make", "parse", "read", "write", "open", "close", "find", "build",
		"encode", "decode", "merge", "split", "load", "store", "scan", "flush", "seek", "trim"}
	nouns := []string{"User", "Index", "Buffer", "Token", "Vector", "Config", "Result", "Node", "Query", "Cache",
		"Chunk", "Model", "Weight", "Header", "Segment", "Stream", "Record", "Shard", "Batch", "Session"}
	types := []string{"int", "string", "[]byte", "error", "bool", "float64", "[]float32", "uint64"}
	for _, v := range verbs {
		for _, n := range nouns {
			for _, ty := range types {
				vecs = append(vecs, m.Encode(fmt.Sprintf("func %s%s(in %s) (%s, error)", v, n, ty, ty)))
			}
		}
	}
	for _, q := range []string{
		"function that parses a user token",
		"open and read a config buffer",
		"build an index over result vectors",
		"close the query cache node",
		"write bytes to an output buffer",
		"decode a batch of weight records",
		"flush the segment stream shard",
		"seek to a header in the model file",
	} {
		queries = append(queries, m.Encode(q))
	}
	return vecs, queries
}

// TestFlatBinary_recallReal_Model2Vec calibrates DefaultOverquery on real
// embeddings rather than on the synthetic clusters, and logs the whole curve so
// the constant can be re-justified when the default model changes.
//
// The synthetic gate cannot do this job. It is generated from isotropic
// Gaussian clusters, which is the geometry binary quantization likes best; real
// embedding sets are anisotropic and have a large common component, and that is
// precisely what decides whether sign bits carry information.
func TestFlatBinary_recallReal_Model2Vec(t *testing.T) {
	vecs, queries := realVecs(t)
	flat := New(vecs)
	t.Logf("real corpus: %d Model2Vec embeddings, dim %d, %d queries", len(vecs), len(vecs[0]), len(queries))

	for _, k := range []int{10, 50} {
		var atDefault float64
		for _, over := range []int{1, 2, 4, 8, 16, 32} {
			r := binRecallAt(flat, NewFlatBinaryOverquery(vecs, over), queries, k)
			t.Logf("k=%2d overquery %2d: recall %.4f", k, over, r)
			if over == DefaultOverquery {
				atDefault = r
			}
		}
		if atDefault < 0.90 {
			t.Errorf("k=%d: recall at DefaultOverquery (%d) = %.4f, want >= 0.90", k, DefaultOverquery, atDefault)
		}
	}
}

// TestFlatBinary_centeringIsWhatMakesRealCorporaWork measures the claim in the
// center field's doc comment instead of asserting it from theory.
//
// Real embedding sets have a large common component: every vector agrees in
// sign on a majority of dimensions, and those dimensions then contribute a
// constant to every Hamming distance — carrying no information about which
// document is which. Subtracting the corpus mean moves each dimension's split
// point to where the corpus actually divides.
//
// The synthetic clusters would not show this at all: their centers are drawn
// from a zero-mean Gaussian, so the corpus mean is already ~0 and centering is
// a no-op. This is the test that justifies paying for it.
func TestFlatBinary_centeringIsWhatMakesRealCorporaWork(t *testing.T) {
	vecs, queries := realVecs(t)
	flat := New(vecs)

	const k = 10
	centered := binRecallAt(flat, newFlatBinary(vecs, DefaultOverquery, true), queries, k)
	raw := binRecallAt(flat, newFlatBinary(vecs, DefaultOverquery, false), queries, k)

	// How lopsided is the corpus? Report the fraction of dimensions on which
	// 90%+ of the corpus agrees in sign — the dimensions centering rescues.
	dim := len(vecs[0])
	pos := make([]int, dim)
	for _, v := range vecs {
		for j, x := range v {
			if x >= 0 {
				pos[j]++
			}
		}
	}
	lopsided := 0
	for _, p := range pos {
		frac := float64(p) / float64(len(vecs))
		if frac > 0.9 || frac < 0.1 {
			lopsided++
		}
	}
	t.Logf("recall@%d centered %.4f, uncentered %.4f — %d/%d dimensions carry no sign information uncentered",
		k, centered, raw, lopsided, dim)

	if centered < raw {
		t.Errorf("centering made recall worse: %.4f vs %.4f", centered, raw)
	}
}

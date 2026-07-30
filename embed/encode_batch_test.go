package embed

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

// TestEncodeBatch_matchesSerial is A1's gate, and it is EXACT over the WHOLE
// corpus rather than a sample: the failure mode of a fan-out is a rare input or
// a rare interleaving, not a common one.
//
// Bit-identity here is structural — StaticModel is immutable after load and
// Encode touches no shared mutable state — so what this really guards is that
// the scatter puts each vector back at its own index. An off-by-one in the work
// counter produces perfectly valid vectors in the wrong order, which no
// numerical tolerance would catch and which every downstream cosine would
// quietly accept.
func TestEncodeBatch_matchesSerial(t *testing.T) {
	m := loadTestStaticModel(t)
	texts := goSourceChunks(t)
	t.Logf("corpus: %d chunks", len(texts))

	want := make([][]float32, len(texts))
	for i, s := range texts {
		want[i] = m.Encode(s)
	}

	for _, conc := range []int{0, 1, 2, 3, 8, 16, 64, len(texts) + 10} {
		got := m.EncodeBatch(texts, conc)
		if len(got) != len(want) {
			t.Fatalf("concurrency %d: got %d vectors, want %d", conc, len(got), len(want))
		}
		for i := range want {
			if len(got[i]) != len(want[i]) {
				t.Fatalf("concurrency %d, text %d: dim %d, want %d", conc, i, len(got[i]), len(want[i]))
			}
			for j := range want[i] {
				if got[i][j] != want[i][j] {
					t.Fatalf("concurrency %d, text %d, component %d: %v != serial %v",
						conc, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
}

// TestEncodeBatch_orderIsInputOrder is the cheap, targeted version of the above:
// distinct texts must come back distinguishable and in position. It uses inputs
// whose vectors are far apart, so a misordered scatter is unmistakable rather
// than a near-tie.
func TestEncodeBatch_orderIsInputOrder(t *testing.T) {
	m := loadTestStaticModel(t)
	texts := []string{
		"func encode(text string) []float32",
		"the quick brown fox jumps over the lazy dog",
		"SELECT * FROM users WHERE id = ?",
		"import numpy as np",
		"",
		"package main",
	}
	for _, conc := range []int{1, 2, 6, 16} {
		got := m.EncodeBatch(texts, conc)
		for i, s := range texts {
			want := m.Encode(s)
			for j := range want {
				if got[i][j] != want[j] {
					t.Fatalf("concurrency %d, text %d (%q), component %d: %v != %v",
						conc, i, s, j, got[i][j], want[j])
				}
			}
		}
	}
}

// TestEncodeBatch_degenerate covers the shapes a bulk API meets in production
// and never in a benchmark.
func TestEncodeBatch_degenerate(t *testing.T) {
	m := loadTestStaticModel(t)
	if got := m.EncodeBatch(nil, 0); len(got) != 0 {
		t.Errorf("nil input returned %d vectors", len(got))
	}
	if got := m.EncodeBatch([]string{}, 8); len(got) != 0 {
		t.Errorf("empty input returned %d vectors", len(got))
	}
	// One text, more workers than work.
	got := m.EncodeBatch([]string{"hello"}, 16)
	if len(got) != 1 {
		t.Fatalf("single text returned %d vectors", len(got))
	}
	want := m.Encode("hello")
	for j := range want {
		if got[0][j] != want[j] {
			t.Fatalf("single text component %d: %v != %v", j, got[0][j], want[j])
		}
	}
	// All-empty strings: every vector is the documented zero vector, and none
	// is nil (a caller indexing the result must not have to check).
	got = m.EncodeBatch([]string{"", "", ""}, 4)
	for i, v := range got {
		if len(v) != m.Dim() {
			t.Fatalf("empty text %d: dim %d, want %d", i, len(v), m.Dim())
		}
		for j, x := range v {
			if x != 0 {
				t.Fatalf("empty text %d component %d = %v, want 0", i, j, x)
			}
		}
	}
}

// TestEncodeBatch_concurrentCallers checks the other half of the contract the
// type doc claims: EncodeBatch is itself safe to call from several goroutines,
// which a caller sharding a corpus across services will do. Run under -race.
func TestEncodeBatch_concurrentCallers(t *testing.T) {
	m := loadTestStaticModel(t)
	texts := goSourceChunks(t)
	if len(texts) > 200 {
		texts = texts[:200]
	}
	want := m.EncodeBatch(texts, 1)

	var wg sync.WaitGroup
	errs := make([]string, 4)
	for w := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := m.EncodeBatch(texts, runtime.NumCPU())
			for i := range want {
				for j := range want[i] {
					if got[i][j] != want[i][j] {
						errs[w] = "vector mismatch under concurrent EncodeBatch calls"
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	for _, e := range errs {
		if e != "" {
			t.Fatal(e)
		}
	}
}

// loadTestStaticModel opens the Model2Vec checkpoint, skipping without it.
func loadTestStaticModel(tb testing.TB) *StaticModel {
	tb.Helper()
	const dir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		tb.Skipf("no static model at %s — see testdata/README.md", dir)
	}
	m, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

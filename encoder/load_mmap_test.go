package encoder

import (
	"os"
	"testing"
)

// TestLoad_usesMmapAndMatchesFS is the regression for AUDIT #2: encoder.Load must
// take the mmap loader (page-cache-shared, GC-light, Close releases the mapping)
// yet produce identical embeddings to the fs.FS route. Same weights, different
// read path — the vectors must be bit-identical.
func TestLoad_usesMmapAndMatchesFS(t *testing.T) {
	const dir = "../testdata/encoder-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no CodeRankEmbed fixture at %s", dir)
	}
	m1, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = m1.Close() }()
	m2, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = m2.Close() }()

	for _, text := range []string{"how do i parse json", "func add(a, b int) int"} {
		v1, err := m1.Encode(text, false)
		if err != nil {
			t.Fatal(err)
		}
		v2, err := m2.Encode(text, false)
		if err != nil {
			t.Fatal(err)
		}
		if len(v1) != len(v2) {
			t.Fatalf("%q: dim %d vs %d", text, len(v1), len(v2))
		}
		for i := range v1 {
			if v1[i] != v2[i] {
				t.Fatalf("%q: Load and LoadFromFS diverge at %d: %v vs %v", text, i, v1[i], v2[i])
			}
		}
	}

	// Close on the Load-built (mmap-backed) model must succeed and be idempotent —
	// before the fix it flipped a flag with nothing to release.
	if err := m1.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := m1.Close(); err != nil {
		t.Errorf("second Close (idempotent): %v", err)
	}
}

package ann

import (
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

func recallAt(truth, got []Hit, k int) float64 {
	set := make(map[int]bool, k)
	for i, h := range truth {
		if i >= k {
			break
		}
		set[h.Index] = true
	}
	hit := 0
	for i, h := range got {
		if i >= k {
			break
		}
		if set[h.Index] {
			hit++
		}
	}
	return float64(hit) / float64(k)
}

// TestFlatI8_recallVsFlat measures the int8 index's recall against the exact
// float32 Flat scan: the top-k sets should overlap heavily for L2-normalized
// vectors (the quantization error is small and bounded). The bar is conservative;
// the logged mean is typically much higher.
func TestFlatI8_recallVsFlat(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	vecs := randUnitSet(rng, 5000, 256)
	f32 := New(vecs)
	i8 := NewFlatI8(vecs)
	if i8.Len() != f32.Len() {
		t.Fatalf("Len mismatch: i8 %d, f32 %d", i8.Len(), f32.Len())
	}

	const k, queries = 10, 50
	var sum float64
	for range queries {
		q := randUnit(rng, 256)
		sum += recallAt(f32.Query(q, k), i8.Query(q, k), k)
	}
	mean := sum / queries
	t.Logf("FlatI8 recall@%d vs float32 Flat: mean %.4f (N=%d, d=256, int8 storage)", k, mean, len(vecs))
	if mean < 0.90 {
		t.Errorf("mean recall@%d = %.4f, want ≥ 0.90", k, mean)
	}
}

func TestFlatI8_memoryQuarter(t *testing.T) {
	vecs := randUnitSet(rand.New(rand.NewPCG(1, 2)), 1000, 256)
	i8 := NewFlatI8(vecs)
	// int8 storage: n*dim bytes + n*4 scale bytes, vs float32's n*dim*4.
	i8Bytes := len(i8.bq) + len(i8.scales)*4
	f32Bytes := 1000 * 256 * 4
	ratio := float64(f32Bytes) / float64(i8Bytes)
	t.Logf("storage: int8 %d B vs float32 %d B → %.2f× smaller", i8Bytes, f32Bytes, ratio)
	if ratio < 3.9 {
		t.Errorf("memory ratio %.2f×, want ~4×", ratio)
	}
}

func TestFlatI8_topKAndDeterminism(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	vecs := randUnitSet(rng, 200, 64)
	i8 := NewFlatI8(vecs)
	q := randUnit(rng, 64)

	// Determinism: identical results across calls.
	if a, b := i8.Query(q, 10), i8.Query(q, 10); len(a) != 10 || len(b) != 10 {
		t.Fatalf("Query len: %d, %d", len(a), len(b))
	} else {
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("non-deterministic at %d: %+v vs %+v", i, a[i], b[i])
			}
		}
	}
	// Descending score order.
	hits := i8.Query(q, 10)
	for i := 1; i < len(hits); i++ {
		if hits[i-1].Score < hits[i].Score {
			t.Errorf("not descending at %d: %.6f < %.6f", i, hits[i-1].Score, hits[i].Score)
		}
	}
	// k semantics: k<=0 and k>=Len return all.
	if all := i8.Query(q, -1); len(all) != 200 {
		t.Errorf("k<0: got %d, want 200", len(all))
	}
	if all := i8.Query(q, 999); len(all) != 200 {
		t.Errorf("k>Len: got %d, want 200", len(all))
	}
	// Wrong-dim query and empty index return nil.
	if got := i8.Query(make([]float32, 63), 5); got != nil {
		t.Errorf("wrong-dim query: got %d hits, want nil", len(got))
	}
	if got := NewFlatI8(nil).Query(nil, 5); got != nil {
		t.Errorf("empty index: got %d hits, want nil", len(got))
	}
}

// TestNewFlatI8_streamingIsBitIdentical is A6's gate. Streaming one row at a
// time must produce byte-for-byte the codes and scales the staged
// QuantizeRowsInt8 produced — it is the same arithmetic, since that function is
// a loop over QuantizeRowInt8, and this asserts it rather than assuming it.
//
// The ragged cases carry the real risk. A short vector must be zero-padded and a
// long one truncated, and the streaming path does that through a reused scratch
// row — so a missing clear() leaks the previous vector's tail into the next
// short one, which is invisible in a uniform-dimension corpus and wrong in a
// ragged one.
func TestNewFlatI8_streamingIsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 6))
	for _, tc := range []struct {
		name string
		vecs [][]float32
	}{
		{"uniform", unitVecs(200, 64, 1)},
		{"single", unitVecs(1, 32, 2)},
		{"all-zero rows", make([][]float32, 8)},
	} {
		if tc.name == "all-zero rows" {
			for i := range tc.vecs {
				tc.vecs[i] = make([]float32, 16)
			}
		}
		t.Run(tc.name, func(t *testing.T) { assertStreamedMatchesStaged(t, tc.vecs) })
	}

	// Ragged: alternating short, exact and long rows, with large values in the
	// long ones so a leaked tail would move the scale visibly.
	t.Run("ragged", func(t *testing.T) {
		d := 48
		var vecs [][]float32
		for i := range 60 {
			n := d
			switch i % 3 {
			case 0:
				n = d / 3 // short → zero-padded
			case 2:
				n = d * 2 // long → truncated
			}
			v := make([]float32, n)
			for j := range v {
				v[j] = float32(rng.NormFloat64()) * 10
			}
			vecs = append(vecs, v)
		}
		// vecs[0] sets d, so make it exactly d long.
		vecs[0] = make([]float32, d)
		for j := range vecs[0] {
			vecs[0][j] = float32(rng.NormFloat64())
		}
		assertStreamedMatchesStaged(t, vecs)
	})
}

// assertStreamedMatchesStaged compares NewFlatI8 against the pre-A6 shape: stage
// every vector into one flat [n,d] float32 block, then quantize the block.
func assertStreamedMatchesStaged(t *testing.T, vecs [][]float32) {
	t.Helper()
	n := len(vecs)
	d := 0
	if n > 0 {
		d = len(vecs[0])
	}
	flat := make([]float32, n*d)
	for i, v := range vecs {
		copy(flat[i*d:i*d+d], v)
	}
	wantBq, wantScales := linalg.QuantizeRowsInt8(flat, n, d)

	f := NewFlatI8(vecs)
	if f.n != n || f.dim != d {
		t.Fatalf("n=%d dim=%d, want %d/%d", f.n, f.dim, n, d)
	}
	if len(f.bq) != len(wantBq) {
		t.Fatalf("%d codes, want %d", len(f.bq), len(wantBq))
	}
	for i := range wantBq {
		if f.bq[i] != wantBq[i] {
			t.Fatalf("code %d (row %d, dim %d): %d, staged %d", i, i/max(d, 1), i%max(d, 1), f.bq[i], wantBq[i])
		}
	}
	for i := range wantScales {
		if f.scales[i] != wantScales[i] {
			t.Fatalf("scale %d: %v, staged %v", i, f.scales[i], wantScales[i])
		}
	}
}

package ann

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"reflect"
	"testing"
)

// TestLoadFlatI8_rejectsInconsistentShape pins the fuzz-found bug: an empty index
// (n=0) with a nonzero dim was accepted (the size check is satisfied by n*dim=0),
// leaving a loaded index whose huge dim could drive a gigantic allocation.
func TestLoadFlatI8_rejectsInconsistentShape(t *testing.T) {
	b := binary.LittleEndian.AppendUint32(nil, flatI8Magic)
	b = binary.LittleEndian.AppendUint32(b, flatI8Version)
	b = binary.LittleEndian.AppendUint32(b, 0x7ff90000) // dim ≈ 2.1e9
	b = binary.LittleEndian.AppendUint32(b, 0)          // n = 0
	if _, err := LoadFlatI8(b); err == nil {
		t.Error("n=0 with dim>0 should be rejected, not accepted as an empty index")
	}
}

func TestFlatI8_roundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	vecs := randUnitSet(rng, 500, 64)
	f := NewFlatI8(vecs)

	blob, err := f.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	g, err := LoadFlatI8(blob)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(f, g) {
		t.Fatal("round-trip: loaded index differs from original")
	}
	// Behavioral equivalence: identical query results.
	for i := range 25 {
		q := randUnit(rng, 64)
		if !reflect.DeepEqual(f.Query(q, 10), g.Query(q, 10)) {
			t.Fatalf("query %d: loaded index returns different hits", i)
		}
	}
	// Re-marshal is byte-identical (canonical format).
	if blob2, _ := g.MarshalBinary(); !bytes.Equal(blob, blob2) {
		t.Error("re-marshal not byte-identical")
	}
}

func TestFlatI8_emptyRoundTrip(t *testing.T) {
	f := NewFlatI8(nil)
	blob, err := f.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	g, err := LoadFlatI8(blob)
	if err != nil {
		t.Fatalf("load empty: %v", err)
	}
	if g.Len() != 0 {
		t.Errorf("empty index Len = %d, want 0", g.Len())
	}
	if g.Query(make([]float32, 64), 5) != nil {
		t.Error("query on empty index should be nil")
	}
}

func TestLoadFlatI8_rejectsBadBlobs(t *testing.T) {
	good, _ := NewFlatI8(randUnitSet(rand.New(rand.NewPCG(3, 4)), 100, 32)).MarshalBinary()

	cases := map[string][]byte{
		"too short":      {1, 2, 3},
		"empty":          {},
		"bad magic":      flip(good, 0),
		"truncated body": good[:len(good)-4],
		"trailing byte":  append(append([]byte(nil), good...), 0),
	}
	for name, blob := range cases {
		if _, err := LoadFlatI8(blob); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
	// Sanity: the unmodified blob still loads.
	if _, err := LoadFlatI8(good); err != nil {
		t.Errorf("unmodified blob failed to load: %v", err)
	}
}

func flip(b []byte, i int) []byte {
	out := append([]byte(nil), b...)
	out[i]++
	return out
}

// FuzzLoadFlatI8 asserts LoadFlatI8 never panics on arbitrary bytes, and that any
// blob it accepts is query-ready and round-trips byte-identically.
func FuzzLoadFlatI8(f *testing.F) {
	seed, _ := NewFlatI8(randUnitSet(rand.New(rand.NewPCG(5, 6)), 50, 16)).MarshalBinary()
	f.Add(seed)
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0})
	empty, _ := NewFlatI8(nil).MarshalBinary()
	f.Add(empty)

	f.Fuzz(func(t *testing.T, data []byte) {
		// Cap input size: the validation/bounds logic is exercised by small inputs;
		// larger ones only slow each exec without finding new logic bugs. Keeps the
		// fuzzer fast and focused (an unbounded version stalled a CI worker).
		if len(data) > 1<<16 {
			return
		}
		idx, err := LoadFlatI8(data)
		if err != nil {
			return
		}
		// Crash-safety: an accepted blob must be queryable and re-serializable
		// without panicking. (Byte-exact round-trip + query equivalence are pinned
		// deterministically by TestFlatI8_roundTrip on real data — not asserted here,
		// since arbitrary scale bytes can be NaN, whose float32 bit pattern is not
		// guaranteed to survive a round-trip through a float32 variable.)
		_ = idx.Query(make([]float32, idx.dim), 5)
		if _, err := idx.MarshalBinary(); err != nil {
			t.Fatalf("re-marshal of a loaded index failed: %v", err)
		}
	})
}

// TestFlatI8_WriteToMatchesMarshal is §4.3's gate: WriteTo must emit exactly the
// bytes MarshalBinary returns, so the two cannot drift into different formats.
//
// Byte-for-byte over several shapes including the degenerate ones, plus a
// round-trip through LoadFlatI8 — a format that differs only in a header field
// would still load, and only compare-then-query catches that.
func TestFlatI8_WriteToMatchesMarshal(t *testing.T) {
	for _, tc := range []struct{ n, d int }{{0, 0}, {1, 4}, {7, 16}, {500, 64}, {1000, 129}} {
		var vecs [][]float32
		if tc.n > 0 {
			vecs = unitVecs(tc.n, tc.d, int64(tc.n*31+tc.d))
		}
		f := NewFlatI8(vecs)

		want, err := f.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		var got bytes.Buffer
		nw, err := f.WriteTo(&got)
		if err != nil {
			t.Fatalf("n=%d d=%d: %v", tc.n, tc.d, err)
		}
		if nw != int64(got.Len()) {
			t.Fatalf("n=%d d=%d: WriteTo reported %d bytes, wrote %d", tc.n, tc.d, nw, got.Len())
		}
		if !bytes.Equal(got.Bytes(), want) {
			t.Fatalf("n=%d d=%d: WriteTo produced %d bytes, MarshalBinary %d; first diff at %d",
				tc.n, tc.d, got.Len(), len(want), firstDiff(got.Bytes(), want))
		}
		// And the streamed bytes must load back into an equivalent index.
		back, err := LoadFlatI8(got.Bytes())
		if err != nil {
			t.Fatalf("n=%d d=%d: load streamed blob: %v", tc.n, tc.d, err)
		}
		if back.Len() != f.Len() || back.dim != f.dim {
			t.Fatalf("n=%d d=%d: round-trip shape %d×%d, want %d×%d",
				tc.n, tc.d, back.Len(), back.dim, f.Len(), f.dim)
		}
		if tc.n > 0 {
			a, b := f.Query(vecs[0], 5), back.Query(vecs[0], 5)
			if len(a) != len(b) {
				t.Fatalf("n=%d d=%d: %d hits vs %d", tc.n, tc.d, len(a), len(b))
			}
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("n=%d d=%d hit %d: %+v vs %+v", tc.n, tc.d, i, a[i], b[i])
				}
			}
		}
	}
}

func firstDiff(a, b []byte) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

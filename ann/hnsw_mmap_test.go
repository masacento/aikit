package ann

import (
	"encoding/binary"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"unsafe"
)

func writeHNSWBlob(t *testing.T, h *HNSW) string {
	t.Helper()
	blob, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "index.hnsw")
	if err := os.WriteFile(p, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestLoadHNSWMmap_matchesCopy: the zero-copy mmap index must be
// behavior-identical to the in-memory one built from the same corpus, in both
// storage modes.
func TestLoadHNSWMmap_matchesCopy(t *testing.T) {
	for _, tc := range []struct {
		name string
		i8   bool
	}{
		{"f32", false},
		{"int8", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))
			orig := BuildHNSW(randUnitSet(rng, 600, 48),
				Config{M: 12, EfConstruction: 60, EfSearch: 40, Seed: 7, Int8: tc.i8})

			m, err := LoadHNSWMmap(writeHNSWBlob(t, orig))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()

			if m.Len() != orig.Len() || m.dim != orig.dim {
				t.Fatalf("shape: got (%d,%d) want (%d,%d)", m.Len(), m.dim, orig.Len(), orig.dim)
			}
			for i := range 25 {
				q := randUnit(rng, 48)
				got, want := m.Query(q, 10), orig.Query(q, 10)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("query %d: mmap result differs from in-memory\n got %v\nwant %v", i, got, want)
				}
			}
		})
	}
}

// TestHNSWMmap_vectorsAreAliasedNotCopied checks the actual point of this
// loader, not just its output: the f32 vector rows must point INTO the mmap'd
// byte range, not into a freshly allocated copy. A regression that silently
// fell back to copying (e.g. a broken alignment check reading garbage, then
// "fixing" it by copying) would pass every correctness test above and still
// defeat the whole reason LoadHNSWMmap exists.
func TestHNSWMmap_vectorsAreAliasedNotCopied(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	orig := BuildHNSW(randUnitSet(rng, 50, 16), Config{M: 8, EfConstruction: 40, EfSearch: 20, Seed: 3})
	m, err := LoadHNSWMmap(writeHNSWBlob(t, orig))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	lo := uintptr(unsafe.Pointer(&m.mmap[0]))
	hi := lo + uintptr(len(m.mmap))
	for i, row := range m.vecs {
		p := uintptr(unsafe.Pointer(&row[0]))
		if p < lo || p >= hi {
			t.Fatalf("vecs[%d] at %#x is outside the mapping [%#x, %#x) — not aliased", i, p, lo, hi)
		}
	}
}

// TestHNSWMmap_int8VectorsAreAliased is the int8-mode sibling: bq must alias
// the mapping too (this direction was already possible before the v4 bump —
// int8 has no alignment requirement — but LoadHNSWMmap should still get it
// right for both storage modes, not just the f32 one the bump unblocked).
func TestHNSWMmap_int8VectorsAreAliased(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	orig := BuildHNSW(randUnitSet(rng, 50, 16), Config{M: 8, EfConstruction: 40, EfSearch: 20, Seed: 3, Int8: true})
	m, err := LoadHNSWMmap(writeHNSWBlob(t, orig))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if len(m.bq) == 0 {
		t.Fatal("bq is empty")
	}
	lo := uintptr(unsafe.Pointer(&m.mmap[0]))
	hi := lo + uintptr(len(m.mmap))
	p := uintptr(unsafe.Pointer(&m.bq[0]))
	if p < lo || p >= hi {
		t.Fatalf("bq at %#x is outside the mapping [%#x, %#x) — not aliased", p, lo, hi)
	}
}

func TestHNSWMmap_closePanicsOnQueryAndAdd(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	orig := BuildHNSW(randUnitSet(rng, 100, 32), Config{M: 8, EfConstruction: 40, EfSearch: 20, Seed: 1})
	m, err := LoadHNSWMmap(writeHNSWBlob(t, orig))
	if err != nil {
		t.Fatal(err)
	}
	q := randUnit(rand.New(rand.NewPCG(5, 6)), 32)
	_ = m.Query(q, 5) // fine before Close

	if err := m.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second close (must be idempotent): %v", err)
	}

	t.Run("Query", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("Query after Close should panic")
			}
		}()
		m.Query(q, 5)
	})
	t.Run("Add", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("Add after Close should panic")
			}
		}()
		m.Add(q)
	})
}

// In-memory indexes have nothing to release: Close is a no-op and leaves them
// queryable (only a mmap-backed index becomes unusable after Close).
func TestHNSW_inMemoryCloseIsNoop(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	h := BuildHNSW(randUnitSet(rng, 50, 16), Config{M: 8, EfConstruction: 40, EfSearch: 20, Seed: 1})
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}
	if got := h.Query(randUnit(rng, 16), 5); got == nil {
		t.Error("in-memory index should still query after Close")
	}
	h.Add(randUnit(rng, 16)) // must not panic
}

// TestLoadHNSWMmap_addAfterLoad: Add-after-load is a documented HNSW
// capability (Load's own doc comment), and it must work identically whether
// the index came from Load (heap) or LoadHNSWMmap (aliased) — the append that
// grows a full-capacity aliased slice always reallocates fresh heap memory
// (never writes into the read-only mapping), so this should behave exactly
// like the heap-loaded case.
func TestLoadHNSWMmap_addAfterLoad(t *testing.T) {
	for _, tc := range []struct {
		name string
		i8   bool
	}{
		{"f32", false},
		{"int8", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(21, 22))
			vecs := randUnitSet(rng, 200, 24)
			cfg := Config{M: 8, EfConstruction: 40, EfSearch: 30, Seed: 9, Int8: tc.i8}
			built := BuildHNSW(vecs, cfg)
			blob, err := built.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}

			extra := randUnitSet(rng, 20, 24)

			heap, err := Load(blob)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range extra {
				heap.Add(v)
			}

			p := filepath.Join(t.TempDir(), "index.hnsw")
			if err := os.WriteFile(p, blob, 0o644); err != nil {
				t.Fatal(err)
			}
			mapped, err := LoadHNSWMmap(p)
			if err != nil {
				t.Fatal(err)
			}
			defer mapped.Close()
			for _, v := range extra {
				mapped.Add(v)
			}

			if mapped.Len() != heap.Len() {
				t.Fatalf("Len after Add: mmap=%d heap=%d", mapped.Len(), heap.Len())
			}
			for i := range 15 {
				q := randUnit(rng, 24)
				got, want := mapped.Query(q, 10), heap.Query(q, 10)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("query %d after Add: mmap result differs from heap\n got %v\nwant %v", i, got, want)
				}
			}
		})
	}
}

func TestLoadHNSWMmap_emptyAndBad(t *testing.T) {
	empty := NewHNSW(Config{M: 8, EfConstruction: 40, EfSearch: 20})
	m, err := LoadHNSWMmap(writeHNSWBlob(t, empty))
	if err != nil {
		t.Fatalf("empty index via mmap: %v", err)
	}
	if m.Len() != 0 {
		t.Errorf("empty len %d", m.Len())
	}
	_ = m.Close()

	bad := filepath.Join(t.TempDir(), "bad.hnsw")
	if err := os.WriteFile(bad, []byte("not a valid HNSW blob, but long enough to pass the size floor"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadHNSWMmap(bad); err == nil {
		t.Error("corrupt blob should error (and not leak the mapping)")
	}
	if _, err := LoadHNSWMmap(filepath.Join(t.TempDir(), "nope.hnsw")); err == nil {
		t.Error("missing file should error")
	}
}

// TestLoadHNSWMmap_rejectsOldVersion confirms LoadHNSWMmap enforces the same
// version check as Load — both route through loadHNSW, but this pins that a
// v3 blob (pre-alignment-pad) is rejected rather than misread as v4, which
// would silently parse a stale layout.
func TestLoadHNSWMmap_rejectsOldVersion(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	h := BuildHNSW(randUnitSet(rng, 40, 12), Config{M: 8, EfConstruction: 40, EfSearch: 20, Seed: 1})
	blob, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(blob[4:], 3) // stamp the version field as v3
	p := filepath.Join(t.TempDir(), "oldversion.hnsw")
	if err := os.WriteFile(p, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(blob); err == nil {
		t.Error("Load should reject a v3-stamped blob")
	}
	if _, err := LoadHNSWMmap(p); err == nil {
		t.Error("LoadHNSWMmap should reject a v3-stamped blob")
	}
}

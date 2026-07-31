package ann

import (
	"io"
	"math/rand/v2"
	"runtime"
	"testing"
)

// benchHNSW builds one index shared across the persist benchmarks. Building an
// HNSW is far more expensive than serializing it, so building per-iteration would
// bury the thing being measured.
func benchHNSW(b *testing.B, n, dim int, int8mode bool) *HNSW {
	b.Helper()
	rng := rand.New(rand.NewPCG(101, 102))
	return BuildHNSW(randUnitSet(rng, n, dim), Config{
		M: 16, EfConstruction: 100, EfSearch: 50, Seed: 77, Int8: int8mode,
	})
}

// BenchmarkHNSWPersist compares the two serialization surfaces at the same shape.
// This is a FOOTPRINT item: the number to read is B/op, not ns/op. MarshalBinary
// necessarily allocates the whole blob; WriteTo allocates one 64 KiB buffer
// regardless of index size. Time is reported to prove streaming does not cost
// anything, not because streaming is expected to win it.
func BenchmarkHNSWPersist(b *testing.B) {
	for _, mode := range []struct {
		name string
		i8   bool
	}{{"f32", false}, {"int8", true}} {
		h := benchHNSW(b, 50_000, 256, mode.i8)
		b.Run(mode.name+"/MarshalBinary", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				blob, err := h.MarshalBinary()
				if err != nil {
					b.Fatal(err)
				}
				if _, err := io.Discard.Write(blob); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(mode.name+"/WriteTo", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := h.WriteTo(io.Discard); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkHNSWLoad measures the load side the arenas changed.
func BenchmarkHNSWLoad(b *testing.B) {
	h := benchHNSW(b, 50_000, 256, false)
	blob, err := h.MarshalBinary()
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		idx, err := Load(blob)
		if err != nil {
			b.Fatal(err)
		}
		runtime.KeepAlive(idx)
	}
}

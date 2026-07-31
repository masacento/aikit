package ann

import (
	"math/rand/v2"
	"testing"
)

// BenchmarkD5HeapPooling is the in-process A/B for pooling searchLayer's two
// heaps (measuring-performance §1.6: alternate variants inside one process; a
// cross-invocation delta under ~5% is unmeasured on this box).
//
// One index is built once and shared by both arms — building a 50k HNSW is ~30 s,
// and rebuilding it per arm per -count is what made the first attempt at this
// measurement cost minutes rather than seconds.
//
// The arms differ in exactly one thing: whether the scratch's heap arrays keep
// the capacity a previous query grew, or start from nil and re-grow by append
// (~7 doublings each to reach ef). That is precisely the change under test.
func BenchmarkD5HeapPooling(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	h := BuildHNSW(randUnitSet(rng, 5000, 64), Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	q := randUnit(rng, 64)

	b.Run("cold-heaps", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			// Hand the pool a scratch whose heaps have no capacity, reproducing
			// the pre-change behaviour (the visitTracker was already pooled; the
			// heaps were fresh per searchLayer call).
			sc := h.getScratch()
			sc.cands.items, sc.results.items = nil, nil
			h.putScratch(sc)
			_ = h.Query(q, 10)
		}
	})
	b.Run("pooled-heaps", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_ = h.Query(q, 10)
		}
	})
}

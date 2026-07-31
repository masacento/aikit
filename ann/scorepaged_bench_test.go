package ann

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

var sinkPagedNbr any

// §3.5: paged Query cost with many blocks — the removed per-block query
// re-quantization was paid once per block. Budget large so paging itself doesn't
// dominate; this isolates the per-block quant cost.
func BenchmarkScorePagedQuery(b *testing.B) {
	rng := rand.New(rand.NewPCG(9, 9))
	const dim, n, blockRows = 768, 120_000, 32 // ~3750 blocks
	f := NewFlatI8(randUnitSet(rng, n, dim))
	blob, err := f.MarshalBinary()
	if err != nil {
		b.Fatal(err)
	}
	p := filepath.Join(b.TempDir(), "i.fi8")
	if err := os.WriteFile(p, blob, 0o644); err != nil {
		b.Fatal(err)
	}
	paged, err := loadFlatI8MmapPaged(p, int64(n)*dim, blockRows)
	if err != nil {
		b.Fatal(err)
	}
	defer paged.Close()
	q := randUnit(rand.New(rand.NewPCG(1, 1)), dim)
	b.ResetTimer()
	for b.Loop() {
		sinkPagedNbr = paged.Query(q, 10)
	}
}

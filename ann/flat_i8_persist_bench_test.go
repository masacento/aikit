package ann

import "testing"

// BenchmarkFlatI8Marshal exists because index (de)serialization had no benchmark, so
// nothing caught that MarshalBinary appended the code block one byte at a time and
// pushed every scale through a captured closure (perf-campaign-2026-07-28.md, item 5).
//
// Sized at a realistic embedded index: 50k vectors × 384 dims is ~19 MB of codes, the
// scale at which per-element overhead stops being invisible.
func BenchmarkFlatI8Marshal(b *testing.B) {
	const n, dim = 50_000, 384
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32((i*31+d*7)%199-99) / 99
		}
		vecs[i] = v
	}
	f := NewFlatI8(vecs)
	b.ResetTimer()
	var sink int
	for range b.N {
		blob, err := f.MarshalBinary()
		if err != nil {
			b.Fatal(err)
		}
		sink += len(blob)
	}
	_ = sink
	b.SetBytes(int64(16 + n*dim + n*4))
}

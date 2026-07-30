package ann

import (
	"fmt"
	"testing"
)

// BenchmarkFlatBinaryQuery is perf-campaign item 38's end-to-end arbiter: the
// two-stage query against the exact scan it replaces, at both real model
// dimensions and across corpus sizes.
//
// The kernel-level ratio (linalg.BenchmarkHammingRows) is 14× and is NOT the
// number to quote — it measures the prefilter alone. What a caller gets is that
// stage plus an exact rerank of overquery·k candidates plus the selection, and
// the rerank does not shrink with the corpus. This benchmark is what closes
// that gap, per measuring-performance.md §1.4.
func BenchmarkFlatBinaryQuery(b *testing.B) {
	const k = 10
	for _, dim := range []int{256, 768} {
		for _, n := range []int{100_000, 1_000_000} {
			vecs, qs := binClusteredCorpus(n/100, 100, dim, 8, 0.5, 38)
			flat := New(vecs)
			fb := NewFlatBinary(vecs)
			name := fmt.Sprintf("d%d/N%d", dim, n)

			b.Run(name+"/exact", func(b *testing.B) {
				b.ReportAllocs()
				i := 0
				for b.Loop() {
					_ = flat.Query(qs[i%len(qs)], k)
					i++
				}
			})
			b.Run(name+"/binary", func(b *testing.B) {
				b.ReportAllocs()
				i := 0
				for b.Loop() {
					_ = fb.Query(qs[i%len(qs)], k)
					i++
				}
			})
		}
	}
}

// BenchmarkFlatBinaryOverquery prices the recall knob. The prefilter scans the
// whole corpus regardless, so raising overquery adds only rerank work — the
// question is how steeply, and whether the default can afford the setting where
// the recall curve flattens (16 on both the synthetic and the real corpus).
//
// k is swept alongside because the rerank is overquery·k candidates: the knob
// is cheap at k=10 and cannot be assumed cheap at k=100.
func BenchmarkFlatBinaryOverquery(b *testing.B) {
	const dim, n = 768, 200_000
	vecs, qs := binClusteredCorpus(n/100, 100, dim, 8, 0.5, 38)
	for _, k := range []int{10, 100} {
		for _, over := range []int{4, 8, 16, 32} {
			fb := NewFlatBinaryOverquery(vecs, over)
			b.Run(fmt.Sprintf("k%d/over%d", k, over), func(b *testing.B) {
				b.ReportAllocs()
				i := 0
				for b.Loop() {
					_ = fb.Query(qs[i%len(qs)], k)
					i++
				}
			})
		}
	}
}

// BenchmarkFlatBinaryPrefilterPaths A/Bs the two prefilter implementations in
// one binary: Query takes the histogram path, QueryFilter with an always-true
// predicate takes the heap path over identical data.
//
// It exists because the histogram's advantage is not uniform. It is flat in
// overquery where the heap is not, but it materializes n distances and makes a
// second pass over them, which the heap avoids — so at a small candidate count
// the heap can win. This is the measurement that says which default is right,
// and it is the one to re-run before touching either path.
func BenchmarkFlatBinaryPrefilterPaths(b *testing.B) {
	const dim, n = 768, 200_000
	vecs, qs := binClusteredCorpus(n/100, 100, dim, 8, 0.5, 38)
	alwaysLive := func(int) bool { return true }
	for _, k := range []int{10, 100} {
		for _, over := range []int{4, 16} {
			fb := NewFlatBinaryOverquery(vecs, over)
			b.Run(fmt.Sprintf("k%d/over%d/hist", k, over), func(b *testing.B) {
				i := 0
				for b.Loop() {
					_ = fb.Query(qs[i%len(qs)], k)
					i++
				}
			})
			b.Run(fmt.Sprintf("k%d/over%d/heap", k, over), func(b *testing.B) {
				i := 0
				for b.Loop() {
					_ = fb.QueryFilter(qs[i%len(qs)], k, alwaysLive)
					i++
				}
			})
		}
	}
}

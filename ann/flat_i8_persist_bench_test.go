package ann

import (
	"fmt"
	"io"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

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

// BenchmarkFlatI8Persist_writeVsMarshal prices §4.3: writing an index to a
// sink through MarshalBinary against streaming it with WriteTo.
//
// It reports peakMiB — the transient heap high-water mark — because that is what
// the finding is about. B/op would show the difference too, but peak is the
// quantity that decides whether a machine can serialize an index at all, and
// bytes-allocated and peak are not the same number (a pool changes one and not
// the other).
//
// READ THE TIME RATIO WITH CARE. The sink is io.Discard, so neither arm pays for
// real I/O — which flatters WriteTo enormously, since MarshalBinary still has to
// materialize the blob while WriteTo has almost nothing left to do. Against a
// file both arms pay the same write cost and the ratio collapses toward 1. The
// numbers that survive a real sink are the allocation and peak ones.
func BenchmarkFlatI8Persist_writeVsMarshal(b *testing.B) {
	for _, tc := range []struct{ n, d int }{{20_000, 256}, {200_000, 256}} {
		f := NewFlatI8(unitVecs(tc.n, tc.d, int64(tc.n)))
		name := fmt.Sprintf("n%d", tc.n)

		b.Run(name+"/marshal", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				blob, err := f.MarshalBinary()
				if err != nil {
					b.Fatal(err)
				}
				sinkPersist, _ = io.Discard.Write(blob)
			}
			b.ReportMetric(peakHeapMiBAnn(func() {
				blob, _ := f.MarshalBinary()
				sinkPersist, _ = io.Discard.Write(blob)
			}), "peakMiB")
		})
		b.Run(name+"/writeTo", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				n, err := f.WriteTo(io.Discard)
				if err != nil {
					b.Fatal(err)
				}
				sinkPersist = int(n)
			}
			b.ReportMetric(peakHeapMiBAnn(func() {
				_, _ = f.WriteTo(io.Discard)
			}), "peakMiB")
		})
	}
}

// peakHeapMiBAnn is the sampled-peak helper (see bench/coldstart_bench_test.go
// for the discussion of what it does and does not measure).
func peakHeapMiBAnn(fn func()) float64 {
	runtime.GC()
	var stop atomic.Bool
	var peak atomic.Uint64
	done := make(chan struct{})
	go func() {
		defer close(done)
		var ms runtime.MemStats
		for !stop.Load() {
			runtime.ReadMemStats(&ms)
			for {
				cur := peak.Load()
				if ms.HeapInuse <= cur || peak.CompareAndSwap(cur, ms.HeapInuse) {
					break
				}
			}
			time.Sleep(200 * time.Microsecond)
		}
	}()
	fn()
	stop.Store(true)
	<-done
	return float64(peak.Load()) / (1 << 20)
}

var sinkPersist int

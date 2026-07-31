package encoder

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestQ8LoadFootprint settles lens §4.5, which claims LoadWeightsQ8
// "materializes the whole f32 model before quantizing" for a peak of
// "~547 MB f32 + ~140 MB int8 ≈ 690 MB to produce a 140 MB model".
//
// The claim needs a PEAK measurement, not an after-the-fact one, and it needs
// RSS as well as heap — because LoadWeights goes through
// embed.OpenSafetensorsMmap, so the f32 side is file-backed pages rather than Go
// heap. Those pages are clean and reclaimable under pressure, which is a
// materially milder problem than 547 MB of heap, and only RSS sees them at all.
//
// It reports rather than asserts. A threshold here would be a guess about the
// machine's page-cache behaviour, and the point is to have the number on record.
func TestQ8LoadFootprint(t *testing.T) {
	const dir = "../testdata/encoder-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no encoder model at %s — see scripts/README.md", dir)
	}
	fi, err := os.Stat(dir + "/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	onDiskMiB := float64(fi.Size()) / (1 << 20)

	runtime.GC()
	baseHeap, baseRSS := heapMiB(), rssMiB()

	var stop atomic.Bool
	var peakHeap, peakRSS atomic.Uint64
	done := make(chan struct{})
	go func() {
		defer close(done)
		for !stop.Load() {
			bump(&peakHeap, uint64(heapMiB()*1024))
			bump(&peakRSS, uint64(rssMiB()*1024))
			time.Sleep(200 * time.Microsecond)
		}
	}()

	q, err := LoadWeightsQ8(dir)
	stop.Store(true)
	<-done
	if err != nil {
		t.Skipf("q8 unsupported for this checkpoint: %v", err)
	}

	afterHeap, afterRSS := heapMiB(), rssMiB()
	t.Logf("checkpoint on disk:      %7.1f MiB", onDiskMiB)
	t.Logf("baseline:                heap %7.1f  RSS %7.1f MiB", baseHeap, baseRSS)
	t.Logf("PEAK during load:        heap %7.1f  RSS %7.1f MiB",
		float64(peakHeap.Load())/1024, float64(peakRSS.Load())/1024)
	t.Logf("after load (model live): heap %7.1f  RSS %7.1f MiB", afterHeap, afterRSS)
	runtime.KeepAlive(q)
}

func bump(dst *atomic.Uint64, v uint64) {
	for {
		cur := dst.Load()
		if v <= cur || dst.CompareAndSwap(cur, v) {
			return
		}
	}
}

func heapMiB() float64 {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return float64(ms.HeapInuse) / (1 << 20)
}

// rssMiB reads resident set size from /proc/self/statm. Returns 0 where that is
// unavailable (darwin, windows), which makes the RSS columns above blank rather
// than wrong.
func rssMiB() float64 {
	b, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	f := strings.Fields(string(b))
	if len(f) < 2 {
		return 0
	}
	pages, err := strconv.ParseFloat(f[1], 64)
	if err != nil {
		return 0
	}
	return pages * float64(os.Getpagesize()) / (1 << 20)
}

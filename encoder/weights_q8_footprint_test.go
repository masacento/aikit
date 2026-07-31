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
// Measured, on CodeRankEmbed (521.6 MiB F32 checkpoint → 199.5 MiB model):
//
//	peak RSS 727.6 MiB   before the per-tensor release
//	peak RSS 242.3 MiB   after it — 3.00×, at +0.1% load time (min-of-10)
//
// So §4.5's magnitude was right and its character was not: heap peaks at exactly
// its steady state, and the whole spike was the f32 mapping faulted in by
// quantization. The fix is to release each tensor's pages as it is consumed, not
// to restructure the load.
//
// It reports the numbers and asserts only the ratio the release controls: peak
// RSS within 1.5× of the loaded model's own RSS. An absolute threshold would be
// a guess about the machine's page-cache behaviour, but the ratio is the thing
// the release exists to hold — it was 3.5× before, and it goes straight back
// there if the release calls are dropped. Linux-only, because madvise cannot
// force a resident drop for this mapping type anywhere else (see mmap's docs).
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
	pkHeap, pkRSS := float64(peakHeap.Load())/1024, float64(peakRSS.Load())/1024
	t.Logf("checkpoint on disk:      %7.1f MiB", onDiskMiB)
	t.Logf("baseline:                heap %7.1f  RSS %7.1f MiB", baseHeap, baseRSS)
	t.Logf("PEAK during load:        heap %7.1f  RSS %7.1f MiB", pkHeap, pkRSS)
	t.Logf("after load (model live): heap %7.1f  RSS %7.1f MiB", afterHeap, afterRSS)
	if runtime.GOOS == "linux" && afterRSS > 0 {
		if ratio := pkRSS / afterRSS; ratio > 1.5 {
			t.Errorf("peak RSS %.1f MiB is %.2f× the loaded model's %.1f MiB — the "+
				"per-tensor release in LoadWeightsQ8 is not taking effect (it was 3.5× without it)",
				pkRSS, ratio, afterRSS)
		}
	}
	runtime.KeepAlive(q)
}

// TestLayerQ8TensorNamesResolve is the drift gate on layerQ8TensorNames: it
// duplicates the tensor names buildWeightsFromSafetensors uses, and a rename on
// one side would otherwise turn every per-layer release into a silent no-op —
// the footprint would quietly regress to 727 MiB with every test still green.
// ReleaseTensors reports an unknown name precisely so this can fail loudly.
func TestLayerQ8TensorNamesResolve(t *testing.T) {
	const dir = "../testdata/encoder-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no encoder model at %s — see scripts/README.md", dir)
	}
	w, err := LoadWeights(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.st.Close() }()
	if !w.Cfg.gatedMLP() || w.hasMoE() {
		t.Skip("checkpoint is not the gated-MLP shape LoadWeightsQ8 accepts")
	}
	for i := range w.Cfg.NumLayers {
		if err := w.releasePages(layerQ8TensorNames(i)...); err != nil {
			t.Errorf("layer %d: %v", i, err)
		}
	}
	if err := w.releasePages(
		"embeddings.word_embeddings.weight",
		"embeddings.token_type_embeddings.weight",
		"emb_ln.weight", "emb_ln.bias",
	); err != nil {
		t.Error(err)
	}
	// And the negative: a name that doesn't resolve must be reported, or the
	// check above proves nothing.
	if err := w.releasePages("encoder.layers.0.attn.Wqkv.weightX"); err == nil {
		t.Error("releasePages accepted an unknown tensor name")
	}
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

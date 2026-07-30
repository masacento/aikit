package encoder

import (
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

// TestBERTFamily_bracketsInFlight pins perf-campaign item 6: every BERT-family
// entry point must register itself in the in-flight counter, so the intra-op
// matmul gate can tell "I am the only forward running" from "N rerank goroutines
// are already using every core".
//
// It observes the counter from *inside* the forward — a peak of 0 means the
// bracket is missing, which is exactly the state the gate misreads as "run
// parallel", turning an N-goroutine rerank loop into NumCPU×NumCPU runnable
// goroutines.
func TestBERTFamily_bracketsInFlight(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatalf("LoadSPLADE: %v", err)
	}
	defer func() { _ = s.Close() }()

	if got := inflightForwards.Load(); got != 0 {
		t.Fatalf("counter not at rest before the test: %d", got)
	}

	// A single Expand must show exactly 1 in flight while the trunk runs — not 0
	// (bracket missing) and not 2 (double-counted).
	var peak, samples int32
	backend := &countingBackend{peak: &peak, samples: &samples}
	s.bert.be = backend
	if _, err := s.Expand("how do i parse json in go"); err != nil {
		t.Fatal(err)
	}
	if samples == 0 {
		t.Fatal("backend never invoked — the probe observed nothing, so this test proves nothing")
	}
	if peak != 1 {
		t.Errorf("in-flight count during a lone Expand = %d, want 1 "+
			"(0 = hiddenStates is not bracketed, so the intra-op gate always reads 'run parallel')", peak)
	}
	if got := inflightForwards.Load(); got != 0 {
		t.Errorf("counter leaked after Expand: %d, want 0", got)
	}

	// Concurrent Expands must see each other, so the gate serializes them.
	const goroutines = 4
	var conc int32
	s.bert.be = &countingBackend{peak: &conc, samples: &samples}
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range goroutines {
		wg.Go(func() {
			<-start
			if _, err := s.Expand("machine learning for semantic search"); err != nil {
				t.Error(err)
			}
		})
	}
	close(start)
	wg.Wait()
	if conc < 2 {
		t.Errorf("peak in-flight across %d concurrent Expands = %d, want ≥2 — "+
			"concurrent forwards must be visible to each other", goroutines, conc)
	}
	if got := inflightForwards.Load(); got != 0 {
		t.Errorf("counter leaked after concurrent Expands: %d, want 0", got)
	}
}

// countingBackend observes inflightForwards from inside a forward, then hands
// the matmul to the exact function the nil-backend path uses — so it changes no
// numerics. It is a probe, not a backend.
type countingBackend struct {
	peak    *int32
	samples *int32
}

func (c *countingBackend) Name() string { return "counting-probe" }

func (c *countingBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	atomic.AddInt32(c.samples, 1)
	n := inflightForwards.Load()
	for {
		old := atomic.LoadInt32(c.peak)
		if n <= old || atomic.CompareAndSwapInt32(c.peak, old, n) {
			break
		}
	}
	matmulBTInto(a, b, dst, M, K, N)
}

func (c *countingBackend) Close() error { return nil }

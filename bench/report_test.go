package bench

import (
	"strings"
	"testing"
)

// twoMachineRecords builds a realistic two-box corpus: an Apple box (cpu-simd + metal) and an
// NVIDIA box (cpu-simd + cuda), same ANN sweep, with a crossover at batch≥64 on each.
func twoMachineRecords() []Record {
	var recs []Record
	mk := func(machine, chip, gpu, arch, backend string, batch int, thr float64, recall float64) Record {
		r := Record{
			Workload: "ann.FlatI8.QueryBatch", Backend: backend, Precision: "int8",
			Device:     Device{Machine: machine, Chip: chip, GPU: gpu, GOARCH: arch, Cores: 10},
			Shape:      Shape{N: 100000, Dim: 256, Batch: batch, K: 10},
			Throughput: thr, ThroughputUnit: "queries/s",
			Quality: Quality{RecallAtK: &recall, ParityOK: true, MaxDeltaVsCPU: 1e-6},
			Meta:    Meta{AikitCommit: "deadbeef1234", Go: "go1.26"},
		}
		if backend != "cpu-simd" {
			r.Device.Chip = ""
		}
		return r
	}
	// Apple: cpu grows with batch; metal wins from batch 64.
	for _, b := range []struct {
		batch      int
		cpu, metal float64
	}{{1, 3000, 1200}, {8, 9000, 8000}, {64, 12000, 30000}, {256, 12500, 52000}} {
		recs = append(recs, mk("apple-m1pro", "Apple M1 Pro", "", "arm64", "cpu-simd", b.batch, b.cpu, 1.0))
		recs = append(recs, mk("apple-m1pro", "", "Apple M1 Pro", "arm64", "metal", b.batch, b.metal, 0.9997))
	}
	// NVIDIA: cpu weaker, cuda wins earlier.
	for _, b := range []struct {
		batch     int
		cpu, cuda float64
	}{{1, 2000, 4000}, {8, 6000, 20000}, {64, 8000, 90000}, {256, 8200, 210000}} {
		recs = append(recs, mk("nvidia-2070s", "EPYC", "", "amd64", "cpu-simd", b.batch, b.cpu, 1.0))
		recs = append(recs, mk("nvidia-2070s", "", "RTX 2070S", "amd64", "cuda", b.batch, b.cuda, 0.9996))
	}
	return recs
}

func TestReport_TwoMachines(t *testing.T) {
	out := Report(twoMachineRecords())

	// Both machines get their OWN per-machine table (apples-to-apples, same box).
	for _, m := range []string{"apple-m1pro", "nvidia-2070s"} {
		if !strings.Contains(out, m) {
			t.Errorf("report missing machine section for %s", m)
		}
	}
	// The normalized summary must use speedup columns, not absolute ms across machines.
	if !strings.Contains(out, "Normalized cross-platform summary") {
		t.Error("missing normalized summary")
	}
	if !strings.Contains(out, "metal ×vs-cpu") || !strings.Contains(out, "cuda ×vs-cpu") {
		t.Errorf("normalized summary should have per-backend speedup columns; got:\n%s", out)
	}
	// HONESTY RULE: never a raw table with both a metal-q/s and a cuda-q/s absolute column
	// side by side (two different CPUs/boxes). The per-machine headers each name exactly one
	// gpu backend; assert no single header line contains both "metal" and "cuda".
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "|") && strings.Contains(line, "q/s") &&
			strings.Contains(line, "metal") && strings.Contains(line, "cuda") {
			t.Errorf("found an absolute-numbers header mixing metal and cuda columns:\n%s", line)
		}
	}
	// Threshold derivation: GPU overtakes CPU at batch ≥ 64 on both boxes (from the fixture).
	if !strings.Contains(out, "batch ≥ 64") {
		t.Errorf("expected a derived crossover of batch ≥ 64; got:\n%s", out)
	}
	// Determinism: same records → byte-identical doc.
	if out2 := Report(twoMachineRecords()); out2 != out {
		t.Error("Report is not deterministic for identical records")
	}
}

func TestReport_Empty(t *testing.T) {
	out := Report(nil)
	if !strings.Contains(out, "No runs recorded yet") {
		t.Errorf("empty report should say so; got:\n%s", out)
	}
	if !strings.Contains(out, "GENERATED") {
		t.Error("even the empty report carries the generated banner")
	}
}

// TestReport_ParityFail proves a failed-parity row is rendered as a failure, not silently
// dropped — a fast-but-wrong number must be visible, per BENCH-gpu.md §3.
func TestReport_ParityFail(t *testing.T) {
	recs := []Record{
		{Workload: "ann.FlatI8.QueryBatch", Backend: "cpu-simd", Precision: "int8",
			Device: Device{Machine: "m", GOARCH: "arm64"}, Shape: Shape{N: 1000, Batch: 8, K: 10},
			Throughput: 1000, ThroughputUnit: "queries/s",
			Quality: Quality{RecallAtK: f64(1.0), ParityOK: true}},
		{Workload: "ann.FlatI8.QueryBatch", Backend: "metal", Precision: "int8",
			Device: Device{Machine: "m", GPU: "M1", GOARCH: "arm64"}, Shape: Shape{N: 1000, Batch: 8, K: 10},
			Throughput: 3000, ThroughputUnit: "queries/s",
			Quality: Quality{RecallAtK: f64(0.71), ParityOK: false}}, // a real recall gap
	}
	out := Report(recs)
	if !strings.Contains(out, "FAIL") {
		t.Errorf("a parity-failing row must surface as FAIL; got:\n%s", out)
	}
}

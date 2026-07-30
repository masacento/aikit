package bench

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestRecordRoundTrip(t *testing.T) {
	recs := []Record{
		{
			Workload: "ann.FlatI8.QueryBatch", Backend: "cpu-simd", Precision: "int8",
			Device:     Device{Machine: "apple-m1pro", Chip: "Apple M1 Pro", GOARCH: "arm64", Cores: 10},
			Shape:      Shape{N: 100000, Dim: 256, Batch: 64, K: 10},
			Timing:     Timing{Compute: 3.2, Wall: 3.5},
			Throughput: 18200, ThroughputUnit: "queries/s",
			Quality: Quality{RecallAtK: new(0.9997), ParityOK: true, MaxDeltaVsCPU: 1e-6},
			Meta:    Meta{AikitCommit: "abc123", Go: "go1.26", Seed: 1, Iters: 20},
		},
		{
			Workload: "ann.FlatI8.QueryBatch", Backend: "metal", Precision: "int8",
			Device:     Device{Machine: "apple-m1pro", GPU: "Apple M1 Pro", SMorFamily: "Apple7", GOARCH: "arm64", Cores: 10},
			Shape:      Shape{N: 100000, Dim: 256, Batch: 64, K: 10},
			Timing:     Timing{OneTime: 12, H2D: 0, Compute: 1.1, Wall: 1.3},
			Throughput: 47000, ThroughputUnit: "queries/s",
			Quality:      Quality{RecallAtK: new(0.9997), ParityOK: true, MaxDeltaVsCPU: 1e-6},
			SpeedupVsCPU: new(2.58),
			Meta:         Meta{AikitCommit: "abc123", Go: "go1.26"},
		},
	}
	path := filepath.Join(t.TempDir(), "records.jsonl")
	if err := AppendRecords(path, recs[0]); err != nil {
		t.Fatal(err)
	}
	if err := AppendRecords(path, recs[1]); err != nil { // append must add, not clobber
		t.Fatal(err)
	}
	got, err := LoadRecords(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, recs) {
		t.Fatalf("round-trip mismatch:\n got %+v\nwant %+v", got, recs)
	}
}

func TestPercentilesAndRecall(t *testing.T) {
	p50, p95, p99, mean := Percentiles([]float64{5, 1, 3, 2, 4})
	if p50 != 3 || mean != 3 {
		t.Fatalf("p50=%v mean=%v want 3,3", p50, mean)
	}
	if p95 < p50 || p99 < p95 {
		t.Fatalf("percentiles not monotone: %v %v %v", p50, p95, p99)
	}
}

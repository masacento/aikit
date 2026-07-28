package bench

import (
	"bufio"
	"encoding/json"
	"os"
	"runtime"
	"runtime/debug"
	"sort"

	"github.com/townsendmerino/aikit/ann"
)

// record.go is the SOURCE-OF-TRUTH layer for the GPU benchmarking methodology
// (docs/BENCH-gpu.md): one JSON record per (workload × backend × shape × precision),
// appended to a records.jsonl, from which the human-readable results doc is GENERATED
// (report.go / cmd/benchreport). A published number can then never drift from the run it
// came from — the same discipline benchmarks/README.md keeps.
//
// This layer has NO GPU dependency on purpose: it is pure schema + I/O so it stays in the
// root (cgo-free) module and is reusable by the CPU harness here AND by the device-gated GPU
// harnesses in gpu/annmetal (Metal) and gpu/anncuda (CUDA), which import it to emit records.

// Record is one measurement. The field layout mirrors docs/BENCH-gpu.md's schema so records
// from different machines and repos are comparable and joinable on (workload, shape, precision).
type Record struct {
	Workload       string   `json:"workload"`        // "ann.FlatI8.QueryBatch" | "vision.SigLIP" | ...
	Backend        string   `json:"backend"`         // "cpu-simd" | "metal" | "cuda"
	Device         Device   `json:"device"`          // the hardware this record ran on
	Precision      string   `json:"precision"`       // "f32" | "f16" | "int8" | "int4"
	Shape          Shape    `json:"shape"`           // only the relevant fields are set
	Timing         Timing   `json:"timing_ms"`       // the four cost components + wall, ms
	Throughput     float64  `json:"throughput"`      // items per second (unit below)
	ThroughputUnit string   `json:"throughput_unit"` // "queries/s" | "patches/s" | "tokens/s" | "GB/s"
	Quality        Quality  `json:"quality"`
	SpeedupVsCPU   *float64 `json:"speedup_vs_cpu"` // same box; nil when CPU-only or no CPU on this box
	Meta           Meta     `json:"meta"`
}

// Device is the run's hardware. Machine is the per-box grouping key the report tables split on
// (two different chips/machines must never share an absolute-numbers column); the rest is the
// device spec docs/BENCH-gpu.md requires so no cross-machine extrapolation is silent.
type Device struct {
	Machine    string `json:"machine"` // stable per-box label, e.g. "apple-m1pro-14c" / "nvidia-2070s"
	Chip       string `json:"chip"`    // CPU chip (baseline) or host chip
	GPU        string `json:"gpu"`     // "" for cpu-simd
	Driver     string `json:"driver,omitempty"`
	SMorFamily string `json:"sm_or_family,omitempty"`
	VRAMMB     int    `json:"vram_mb,omitempty"`
	GOARCH     string `json:"goarch"`
	Cores      int    `json:"cores"`
}

// Shape carries only the fields relevant to a workload (omitempty), so an ANN row and a ViT row
// share one struct without noise.
type Shape struct {
	N       int `json:"n,omitempty"`       // corpus size
	Dim     int `json:"dim,omitempty"`     // vector dimension
	Batch   int `json:"batch,omitempty"`   // query/batch size
	K       int `json:"k,omitempty"`       // top-k
	Seq     int `json:"seq,omitempty"`     // sequence length
	Patches int `json:"patches,omitempty"` // ViT patch count
}

// Timing breaks the cost apart (ms) — never a single fused "GPU time". one_time is amortized
// (context + pipeline + index residency, measured once); transfer is h2d+d2h and is
// backend-dependent (explicit copies on CUDA, ~free on Metal UMA); compute is the warm kernel.
type Timing struct {
	OneTime   float64 `json:"one_time"`
	PerLaunch float64 `json:"per_launch"`
	H2D       float64 `json:"h2d"`
	D2H       float64 `json:"d2h"`
	Compute   float64 `json:"compute"`
	Wall      float64 `json:"wall"`
}

// Quality couples perf to correctness: a fast-but-wrong row is not admissible. RecallAtK is nil
// where recall does not apply; ParityOK gates whether the row may be published at all.
type Quality struct {
	RecallAtK     *float64 `json:"recall_at_k"`
	ParityOK      bool     `json:"parity_ok"`
	MaxDeltaVsCPU float64  `json:"max_delta_vs_cpu"`
}

// Meta is the reproducibility context.
type Meta struct {
	AikitCommit    string `json:"aikit_commit"`
	Go             string `json:"go"`
	BuildFlags     string `json:"build_flags,omitempty"`
	Seed           int64  `json:"seed,omitempty"`
	WarmupLaunches int    `json:"warmup_launches,omitempty"`
	Iters          int    `json:"iters,omitempty"`
}

// CaptureMeta fills the reproducibility context that can be read from the binary itself: the Go
// version and the aikit VCS revision (from the build's embedded VCS info; "" if built without).
func CaptureMeta() Meta {
	m := Meta{Go: runtime.Version()}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" {
				if len(s.Value) > 12 {
					m.AikitCommit = s.Value[:12]
				} else {
					m.AikitCommit = s.Value
				}
			}
		}
	}
	return m
}

// HostDevice fills the host-derivable device fields (GOARCH, logical cores). The caller sets
// Machine/Chip and, for a GPU record, GPU/Driver/SMorFamily/VRAMMB — the harness knows the box.
func HostDevice() Device {
	return Device{GOARCH: runtime.GOARCH, Cores: runtime.NumCPU()}
}

// AppendRecords appends recs to a JSONL file (one compact JSON object per line), creating it if
// absent. Append-only so a periodic run adds to the corpus rather than clobbering prior machines.
func AppendRecords(path string, recs ...Record) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, r := range recs {
		if err := enc.Encode(r); err != nil { // Encode writes a trailing newline → JSONL
			return err
		}
	}
	return w.Flush()
}

// LoadRecords reads a JSONL file written by AppendRecords. Blank lines are skipped.
func LoadRecords(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Record
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Record
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, sc.Err()
}

// --- helpers the GPU harnesses reuse (so recall/latency are computed one way everywhere) ---

// RecallAt returns |got ∩ truth| / |truth| — mean recall@k of a hit list against the exact-CPU
// top-k index set (idxSet of ann.New(...).Query). Truth is the ground-truth index set.
func RecallAt(got []ann.Hit, truth map[int]bool) float64 {
	if len(truth) == 0 {
		return 1
	}
	hitCount := 0
	for _, h := range got {
		if truth[h.Index] {
			hitCount++
		}
	}
	return float64(hitCount) / float64(len(truth))
}

// TruthSet is the exact-CPU top-k index set for one query, the recall ground truth.
func TruthSet(hits []ann.Hit) map[int]bool { return idxSet(hits) }

// Percentiles returns p50/p95/p99/mean (ms) of the latencies; it sorts a copy, so the caller's
// slice is untouched.
func Percentiles(latenciesMs []float64) (p50, p95, p99, mean float64) {
	if len(latenciesMs) == 0 {
		return 0, 0, 0, 0
	}
	s := append([]float64(nil), latenciesMs...)
	sort.Float64s(s)
	return pct(s, 50), pct(s, 95), pct(s, 99), meanOf(s)
}

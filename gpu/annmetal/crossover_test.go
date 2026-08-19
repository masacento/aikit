//go:build darwin

package annmetal_test

// crossover_test.go is the Metal half of the ANN GPU first slice (docs/BENCH-gpu.md): the
// FlatI8 CPU-SIMD vs Metal QueryBatch (N × batch) crossover on REAL Model2Vec embeddings,
// gated on recall + parity against the exact-CPU top-k, emitting bench.Record rows.
//
// It lives here (not in the root bench package) so importing annmetal — the opt-in that
// registers the ann.Backend — never pulls the GPU into the root cgo-free module. The CUDA
// mirror is gpu/anncuda/crossover_test.go; both share bench's schema + report generator.
//
// It is NOT a default `go test`: it needs a Metal device, the Model2Vec checkpoint, and it
// is a documented PERIODIC pass, so it self-skips unless AIKIT_GPU_BENCH is set.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	_ "github.com/townsendmerino/aikit/gpu/annmetal" // registers the Metal ann.Backend via init

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bench"
	"github.com/townsendmerino/aikit/embed"
	gpu "github.com/townsendmerino/aikit/gpu"
)

const (
	modelDir  = "../../testdata/model" // potion-code-16M (Model2Vec), same as benchmarks/
	kTop      = 10
	queryPool = 256 // the largest batch; smaller batches use a prefix
	warmup    = 3
	iters     = 10
)

// envInts parses "1,8,64" from env, or returns the default.
func envInts(key string, def []int) []int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	var out []int
	for s := range strings.SplitSeq(v, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return def
	}
	return out
}

// TestMetalANNCrossover runs the sweep and appends records. Skips cleanly without the env
// gate, the model, or a device — so a normal `go test ./...` on any Mac stays green.
// NOTE (2026-08-12, from the Linux box): the timing loops here used a fixed 10
// iterations, which at small batch sizes is a window SHORTER THAN A GO GC CYCLE — the
// prelude above leaves ~120 MB live, and the cycle that lands during the first shapes
// measured halves their throughput. On CUDA that made the N=10k CPU baseline read 5.4k
// or 10.5k queries/s depending on where the cycle fell. They now use bench.MinDuration,
// which also runs for a minimum wall time. Diagnosis and the ruled-out alternatives
// (cold cache, clock/thermal) are in docs/internal/roofline-2026-08.md §3i.
//
// THE RECORDS IN docs/bench-records/crossover-metal.jsonl PREDATE THIS FIX and carry the
// same error at batch=1 and batch=8 — which is exactly where the "single-query GPU loses,
// EnableGPU pays only at batch >= 8" finding lives. Re-run on the device to regenerate.
func TestMetalANNCrossover(t *testing.T) {
	if os.Getenv("AIKIT_GPU_BENCH") == "" {
		t.Skip("periodic GPU pass — set AIKIT_GPU_BENCH=1 to run (docs/BENCH-gpu.md)")
	}
	if _, err := os.Stat(modelDir + "/model.safetensors"); err != nil {
		t.Skipf("no Model2Vec model at %s — see benchmarks/README.md", modelDir)
	}
	m, err := embed.LoadFromFS(os.DirFS(modelDir), ".")
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	dim := len(m.Encode("probe"))

	// Device spec: auto-fill the GPU name; the operator labels the box via env.
	gpuName := "Apple GPU (Metal)"
	if dev, derr := gpu.CreateSystemDefaultDevice(); derr == nil {
		gpuName = dev.Name()
		dev.ReleaseObjects()
	}
	// Default to the canonical M1 Pro label (matches vit-metal.jsonl and bench's
	// record_test fixture). machineKey groups report rows by Machine alone, so a bare
	// "apple" here renders this one box as a second machine next to "apple-m1pro" — the
	// label split a122098 had to repair by hand. Defaulting correctly stops a re-run from
	// re-introducing it. Override both for any other Apple box.
	machine := envStr("AIKIT_BENCH_MACHINE", "apple-m1pro")
	chip := envStr("AIKIT_BENCH_CHIP", "Apple M1 Pro")
	recordsPath := envStr("AIKIT_BENCH_RECORDS", "../../docs/bench-records/crossover-metal.jsonl")
	_ = os.MkdirAll(filepath.Dir(recordsPath), 0o755)
	_ = os.Remove(recordsPath) // fresh file per run; benchreport merges machines' files

	meta := bench.CaptureMeta()
	if meta.AikitCommit == "" { // module+replace builds strip VCS info; let the operator pass it
		meta.AikitCommit = envStr("AIKIT_BENCH_COMMIT", "unknown")
	}
	meta.WarmupLaunches, meta.Iters = warmup, iters
	cpuDev := bench.HostDevice()
	cpuDev.Machine, cpuDev.Chip = machine, chip
	gpuDev := bench.HostDevice()
	gpuDev.Machine, gpuDev.GPU, gpuDev.SMorFamily = machine, gpuName, "Apple"

	Ns := envInts("AIKIT_BENCH_N", []int{10_000, 100_000})
	batches := envInts("AIKIT_BENCH_BATCH", []int{1, 8, 64, 256})

	for _, N := range Ns {
		vecs := genCorpus(m, N)
		exact := ann.New(vecs) // f32 exact top-k = recall ground truth
		fi8 := ann.NewFlatI8(vecs)

		qpool := genQueries(m, queryPool)
		truth := make([]map[int]bool, len(qpool))
		for i, q := range qpool {
			truth[i] = bench.TruthSet(exact.Query(q, kTop))
		}

		// --- CPU-SIMD QueryBatch (before EnableGPU: per-query CPU path) ---
		cpuThr := map[int]float64{}
		cpuRecall := map[int]float64{}
		cpuHitsByBatch := map[int][][]ann.Hit{}
		for _, b := range batches {
			qs := qpool[:b]
			for range warmup {
				fi8.QueryBatch(qs, kTop)
			}
			var hits [][]ann.Hit
			best := bench.MinDuration(iters, func() { hits = fi8.QueryBatch(qs, kTop) })
			cpuThr[b] = float64(b) / best.Seconds()
			cpuRecall[b] = meanRecall(hits, truth)
			cpuHitsByBatch[b] = hits
		}

		// --- single-query FlatI8.Query — the path gemv_w8a8 serves (NOT QueryBatch, which
		// takes the tiled GEMM even at batch 1). This is what decides whether EnableGPU is
		// worth calling for one query. CPU baseline first. ---
		sq := qpool[:min(64, queryPool)]
		cpuQThr, cpuQHits := timeSingleQuery(fi8, sq, kTop)
		cpuQRecall := meanRecall(cpuQHits, truth[:len(sq)])

		// --- residency (one-time), then Metal QueryBatch ---
		t0 := time.Now()
		if err := fi8.EnableGPU(); err != nil {
			t.Skipf("EnableGPU: %v (no Metal device?)", err)
		}
		oneTimeMs := float64(time.Since(t0).Nanoseconds()) / 1e6

		// --- single-query FlatI8.Query on Metal (gemv_w8a8, SIMD-group per row) ---
		gpuQThr, gpuQHits := timeSingleQuery(fi8, sq, kTop)
		gpuQRecall := meanRecall(gpuQHits, truth[:len(sq)])
		qParityOK, qMaxDelta := parity(gpuQHits, cpuQHits, cpuQRecall, gpuQRecall)
		if !qParityOK {
			t.Errorf("PARITY(Query): N=%d single-query Metal ranking diverged from CPU int8 — a bug, not a knob.", N)
		}
		qsp := gpuQThr / cpuQThr
		qShape := bench.Shape{N: N, Dim: dim, Batch: 1, K: kTop}
		cpuQR, gpuQR := cpuQRecall, gpuQRecall
		cpuQRec := bench.Record{
			Workload: "ann.FlatI8.Query", Backend: "cpu-simd", Precision: "int8",
			Device: cpuDev, Shape: qShape,
			Timing:     bench.Timing{Compute: msPer(cpuQThr, 1), Wall: msPer(cpuQThr, 1)},
			Throughput: cpuQThr, ThroughputUnit: "queries/s",
			Quality: bench.Quality{RecallAtK: &cpuQR, ParityOK: true},
			Meta:    meta,
		}
		gpuQRec := bench.Record{
			Workload: "ann.FlatI8.Query", Backend: "metal", Precision: "int8",
			Device: gpuDev, Shape: qShape,
			Timing:     bench.Timing{OneTime: oneTimeMs, Compute: msPer(gpuQThr, 1), Wall: msPer(gpuQThr, 1)},
			Throughput: gpuQThr, ThroughputUnit: "queries/s",
			Quality:      bench.Quality{RecallAtK: &gpuQR, ParityOK: qParityOK, MaxDeltaVsCPU: qMaxDelta},
			SpeedupVsCPU: &qsp,
			Meta:         meta,
		}
		if err := bench.AppendRecords(recordsPath, cpuQRec, gpuQRec); err != nil {
			t.Fatalf("append query records: %v", err)
		}
		t.Logf("N=%-7d Query(1)    cpu %8.0f q/s  metal %8.0f q/s  %5.2f×  recall cpu %.4f metal %.4f  parity %v",
			N, cpuQThr, gpuQThr, qsp, cpuQR, gpuQR, qParityOK)

		for _, b := range batches {
			qs := qpool[:b]
			for range warmup {
				fi8.QueryBatch(qs, kTop)
			}
			var hits [][]ann.Hit
			best := bench.MinDuration(iters, func() { hits = fi8.QueryBatch(qs, kTop) })
			wallMs := float64(best.Nanoseconds()) / 1e6
			gpuThr := float64(b) / best.Seconds()
			gRecall := meanRecall(hits, truth)
			parityOK, maxDelta := parity(hits, cpuHitsByBatch[b], cpuRecall[b], gRecall)
			if !parityOK {
				t.Errorf("PARITY: N=%d batch=%d — Metal ranking diverged from CPU int8 (recall %.4f vs %.4f). This is a bug, not a knob.",
					N, b, gRecall, cpuRecall[b])
			}
			sp := gpuThr / cpuThr[b]
			shape := bench.Shape{N: N, Dim: dim, Batch: b, K: kTop}
			cr, gr := cpuRecall[b], gRecall

			cpuRec := bench.Record{
				Workload: "ann.FlatI8.QueryBatch", Backend: "cpu-simd", Precision: "int8",
				Device: cpuDev, Shape: shape,
				Timing:     bench.Timing{Compute: msPer(cpuThr[b], b), Wall: msPer(cpuThr[b], b)},
				Throughput: cpuThr[b], ThroughputUnit: "queries/s",
				Quality: bench.Quality{RecallAtK: &cr, ParityOK: true},
				Meta:    meta,
			}
			gpuRec := bench.Record{
				Workload: "ann.FlatI8.QueryBatch", Backend: "metal", Precision: "int8",
				Device: gpuDev, Shape: shape,
				// UMA: H2D/D2H are a memcpy into a shared buffer, folded into Wall (~free), so
				// they are reported 0 and Compute≈Wall; one_time is the residency upload.
				Timing:     bench.Timing{OneTime: oneTimeMs, Compute: wallMs, Wall: wallMs},
				Throughput: gpuThr, ThroughputUnit: "queries/s",
				Quality:      bench.Quality{RecallAtK: &gr, ParityOK: parityOK, MaxDeltaVsCPU: maxDelta},
				SpeedupVsCPU: &sp,
				Meta:         meta,
			}
			if err := bench.AppendRecords(recordsPath, cpuRec, gpuRec); err != nil {
				t.Fatalf("append records: %v", err)
			}
			t.Logf("N=%-7d batch=%-4d  cpu %8.0f q/s  metal %8.0f q/s  %5.2f×  recall cpu %.4f metal %.4f  parity %v",
				N, b, cpuThr[b], gpuThr, sp, cr, gr, parityOK)
		}
		fi8.Close()
	}
	t.Logf("records → %s ; generate the doc with:\n  go run ./bench/cmd/benchreport %s > docs/BENCH-gpu-results.md", recordsPath, recordsPath)
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func msPer(thr float64, batch int) float64 {
	if thr == 0 {
		return 0
	}
	return float64(batch) / thr * 1000
}

// timeSingleQuery times FlatI8.Query ONE query at a time (min per-query over iters),
// returning queries/s and the best pass's hits. This is the gemv_w8a8 path — distinct
// from QueryBatch, which takes the tiled GEMM even at batch 1.
func timeSingleQuery(f *ann.FlatI8, qs [][]float32, k int) (float64, [][]ann.Hit) {
	for range warmup {
		for _, q := range qs {
			f.Query(q, k)
		}
	}
	var hits [][]ann.Hit
	best := bench.MinDuration(iters, func() {
		h := make([][]ann.Hit, len(qs))
		for i, q := range qs {
			h[i] = f.Query(q, k)
		}
		hits = h
	})
	// MinDuration times the whole sweep of len(qs) queries; report per query.
	best /= time.Duration(len(qs))
	return 1.0 / best.Seconds(), hits
}

// genCorpus/genQueries produce distinct, REAL Model2Vec embeddings (not random vectors, which
// concentrate distances and make recall@k meaningless) by encoding templated code signatures.
var (
	verbs = []string{"get", "set", "make", "parse", "read", "write", "open", "close", "find", "build", "scan", "load", "save", "merge", "sort", "hash", "diff", "join"}
	nouns = []string{"User", "Index", "Buffer", "Token", "Vector", "Config", "Result", "Node", "Query", "Cache", "Graph", "Record", "Shard", "Frame"}
	types = []string{"int", "string", "[]byte", "error", "bool", "float64", "[]int", "rune"}
)

func genCorpus(m *embed.StaticModel, n int) [][]float32 {
	out := make([][]float32, n)
	for i := range n {
		v, nn, ty := verbs[i%len(verbs)], nouns[(i/len(verbs))%len(nouns)], types[i%len(types)]
		out[i] = m.Encode(fmt.Sprintf("func %s%s%d(in %s) (%s, error)", v, nn, i, ty, ty))
	}
	return out
}

func genQueries(m *embed.StaticModel, n int) [][]float32 {
	out := make([][]float32, n)
	for i := range n {
		v, nn, ty := verbs[(i*7)%len(verbs)], nouns[(i*3)%len(nouns)], types[(i*5)%len(types)]
		out[i] = m.Encode(fmt.Sprintf("%s the %s %s value at %d", v, nn, ty, i))
	}
	return out
}

func meanRecall(hits [][]ann.Hit, truth []map[int]bool) float64 {
	if len(hits) == 0 {
		return 0
	}
	var s float64
	for i, h := range hits {
		s += bench.RecallAt(h, truth[i])
	}
	return s / float64(len(hits))
}

// parity: does the Metal ranking match the CPU int8 ranking? They compute the same int32 dot,
// so the top-k SETS should be identical. Returns whether every query agrees, and the recall gap.
func parity(gpu, cpu [][]ann.Hit, cpuRecall, gpuRecall float64) (bool, float64) {
	ok := true
	for i := range gpu {
		if !sameSet(gpu[i], cpu[i]) {
			ok = false
			break
		}
	}
	d := gpuRecall - cpuRecall
	if d < 0 {
		d = -d
	}
	return ok, d
}

func sameSet(a, b []ann.Hit) bool {
	if len(a) != len(b) {
		return false
	}
	ai := make([]int, len(a))
	bi := make([]int, len(b))
	for i := range a {
		ai[i], bi[i] = a[i].Index, b[i].Index
	}
	sort.Ints(ai)
	sort.Ints(bi)
	for i := range ai {
		if ai[i] != bi[i] {
			return false
		}
	}
	return true
}

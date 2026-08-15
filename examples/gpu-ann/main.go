// Command gpu-ann is the native-GPU ANN path in one file: the same
// ann.FlatI8 int8 index, queried twice — once on the CPU, once with
// EnableGPU() — to show the ann.Backend seam working end to end and check
// that the two agree exactly.
//
// gpu/'s ~6,150 lines (the device substrate plus 8 one-backend-per-module
// leaves — docs/task-native-gpu.md) are otherwise only discoverable by reading
// godoc: no example or benchmark in the repo touches RegisterBackend,
// EnableGPU, or RegisterResident. This is that demo, scoped to ANN/FlatI8 —
// task-native-gpu.md itself calls that "the headline" ("ANN (Phase 2) alone
// is the headline; the rest compounds it"), and it needs no model checkpoint:
// FlatI8 quantizes whatever L2-normalized float32 vectors it's given, so this
// runs on synthetic data with zero setup beyond the GPU itself.
//
// register_darwin.go and register_linux.go blank-import gpu/annmetal /
// gpu/anncuda — the two backends this repo ships — so main.go itself never
// mentions Metal or CUDA: it only calls the platform-agnostic
// ann.HasBackend/EnableGPU/QueryBatch seam. On a machine with neither (no
// GPU, or Windows — no backend module exists for it yet), EnableGPU returns
// "no backend registered" and this falls back to a CPU-only run: same
// correctness story, no speedup to report.
//
//	go run ./examples/gpu-ann                    # defaults: n=1e6, 256 queries, k=10
//	go run ./examples/gpu-ann -n 50000 -queries 8 # a quick, smaller run
//
// This is its own Go module (like examples/embedded-corpus), not part of the
// root aikit module: gpu/annmetal and gpu/anncuda pull in purego/gocudrv,
// exactly the dependency the root module's cgo-free-by-default promise keeps
// out (docs/architecture.md's dependency-quarantine table).
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/townsendmerino/aikit/ann"
)

func main() {
	n := flag.Int("n", 1_000_000, "corpus size (synthetic vectors)")
	dim := flag.Int("dim", 256, "vector dimension")
	numQueries := flag.Int("queries", 256, "number of queries to benchmark")
	k := flag.Int("k", 10, "top-k per query")
	seed := flag.Int64("seed", 1, "RNG seed, for a reproducible corpus")
	flag.Parse()

	fmt.Printf("building a synthetic corpus: n=%d dim=%d (no model needed —\n", *n, *dim)
	fmt.Println("this demo is about the GPU path, not retrieval quality)")
	rng := rand.New(rand.NewSource(*seed))
	vecs := randomUnitVectors(rng, *n, *dim)
	queries := randomUnitVectors(rng, *numQueries, *dim)

	cpu := ann.NewFlatI8(vecs)
	gpuIdx := ann.NewFlatI8(vecs)

	if !ann.HasBackend() {
		fmt.Println("\nno GPU backend registered for this platform/build — running CPU-only.")
		fmt.Println("(built-in backends: gpu/annmetal on darwin, gpu/anncuda on linux, both")
		fmt.Println("blank-imported by this example's register_*.go — see their build tags.)")
	} else if err := gpuIdx.EnableGPU(); err != nil {
		fmt.Printf("\nbackend %q registered but EnableGPU failed: %v — running CPU-only.\n", ann.BackendName(), err)
	} else {
		fmt.Printf("\nGPU backend %q enabled.\n", ann.BackendName())
	}

	cpuHits, cpuElapsed := timeQueryBatch(cpu, queries, *k)
	gpuHits, gpuElapsed := timeQueryBatch(gpuIdx, queries, *k)

	mismatches := 0
	for i := range queries {
		if !hitsEqual(cpuHits[i], gpuHits[i]) {
			mismatches++
		}
	}

	fmt.Printf("\n%d queries, k=%d, corpus n=%d dim=%d:\n", *numQueries, *k, *n, *dim)
	fmt.Printf("  CPU  QueryBatch: %v\n", cpuElapsed)
	if gpuIdx.GPUEnabled() {
		fmt.Printf("  GPU  QueryBatch: %v (%.2fx)\n", gpuElapsed, float64(cpuElapsed)/float64(gpuElapsed))
	} else {
		fmt.Printf("  (GPU not enabled — this run is CPU vs CPU, mismatches should be exactly 0)\n")
	}
	if mismatches == 0 {
		fmt.Println("  results: IDENTICAL — every query's top-k matched exactly (index and score).")
	} else {
		fmt.Printf("  results: %d/%d queries DIFFERED between the two runs — see the package docs'\n", mismatches, *numQueries)
		fmt.Println("  bit-identity claim (gpu/annmetal's doc comment) before trusting either index.")
	}
}

func timeQueryBatch(idx *ann.FlatI8, queries [][]float32, k int) ([][]ann.Hit, time.Duration) {
	start := time.Now()
	hits := idx.QueryBatch(queries, k)
	return hits, time.Since(start)
}

func hitsEqual(a, b []ann.Hit) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Index != b[i].Index || a[i].Score != b[i].Score {
			return false
		}
	}
	return true
}

// randomUnitVectors returns n random L2-normalized float32 vectors of the given
// dimension — ann's package invariant (both Flat and FlatI8 assume unit
// vectors; see ann/flat.go's doc comment). Values are uniform in [-1,1] before
// normalizing, which is enough to exercise the int8 quantizer's full range —
// this demo is about the GPU path being correct and fast, not about
// retrieval quality, so unlike every other example here it needs no real
// embedding model.
func randomUnitVectors(rng *rand.Rand, n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		var normSq float64
		for j := range v {
			x := rng.Float64()*2 - 1
			v[j] = float32(x)
			normSq += x * x
		}
		norm := float32(1)
		if normSq > 0 {
			norm = float32(1 / math.Sqrt(normSq))
		}
		for j := range v {
			v[j] *= norm
		}
		out[i] = v
	}
	return out
}

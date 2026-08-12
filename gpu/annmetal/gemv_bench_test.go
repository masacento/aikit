//go:build darwin

package annmetal_test

// gemv_bench_test.go measures (1) the M1 Pro's streaming-read bandwidth ceiling in
// isolation, and (2) the single-query int8 GEMV (FlatI8.Query → gemv_w8a8) against it.
// No streaming-bandwidth number for this box existed in the repo; a "% of optimal" with a
// guessed denominator is worse than none (measuring-performance §1.x), so the ceiling is
// measured first, here, from a stream-read far past any cache.
//
// Env-gated like crossover_test (AIKIT_GPU_BENCH=1); a plain `go test` stays green.

import (
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"
	"time"

	_ "github.com/townsendmerino/aikit/gpu/annmetal" // registers the Metal ann.Backend

	"github.com/townsendmerino/aikit/ann"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// bwSrc streams a buffer with 16-byte (float4) grid-stride loads and reduces, so the read
// cannot be dead-code-eliminated. sink is tiny; every thread folds its accumulator into it.
const bwSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void bw_read(
    device const float4* src [[buffer(0)]],
    constant uint&       n4  [[buffer(1)]],   // float4 elements in src
    constant uint&       gsz [[buffer(2)]],   // total threads dispatched (grid stride)
    device float*        sink[[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    float4 acc = float4(0.0);
    for (uint i = gid; i < n4; i += gsz) acc += src[i];
    sink[gid & 1023u] = acc.x + acc.y + acc.z + acc.w;
}`

// measureBandwidth streams `bytes` far past any cache and returns the best (max) sustained
// GB/s over a small grid-size sweep — the ceiling to judge the GEMV against.
func measureBandwidth(t *testing.T, dev *gpu.Device, q gpu.Queue, bytes int) float64 {
	t.Helper()
	lib, err := dev.CompileLibrary(bwSrc, gpu.MSL3_1)
	if err != nil {
		t.Fatalf("compile bw kernel: %v", err)
	}
	p, err := dev.NewComputePipeline(lib, "bw_read")
	if err != nil {
		t.Fatalf("pipeline bw_read: %v", err)
	}
	nFloats := bytes / 4
	src := dev.NewBufferLen(nFloats)
	sink := dev.NewBufferLen(1024)
	defer dev.ReleaseBuf(src)
	defer dev.ReleaseBuf(sink)
	// Touch every page so reads hit real DRAM, not zero-fill-on-demand.
	fv := src.Floats()
	for i := range fv {
		fv[i] = 1.0
	}
	n4 := nFloats / 4
	n4buf := dev.NewBufferU32(uint32(n4))
	defer dev.ReleaseBuf(n4buf)

	best := 0.0
	for _, gsz := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 22} {
		gszBuf := dev.NewBufferU32(uint32(gsz))
		var secs float64 = math.Inf(1)
		for rep := 0; rep < 5; rep++ {
			runtime.LockOSThread()
			e := q.Begin()
			e.Dispatch(p, gsz, 256, src, n4buf, gszBuf, sink)
			e.End()
			runtime.UnlockOSThread()
			if err := e.Err(); err != nil {
				t.Fatalf("bw dispatch: %v", err)
			}
			if s := e.GPUEnd() - e.GPUStart(); s > 0 && s < secs {
				secs = s
			}
		}
		dev.ReleaseBuf(gszBuf)
		gbps := float64(bytes) / secs / 1e9
		t.Logf("  bw grid=%-9d %.1f GB/s (%.3f ms for %d MiB)", gsz, gbps, secs*1e3, bytes>>20)
		if gbps > best {
			best = gbps
		}
	}
	return best
}

// randCorpus builds n random f32 vectors of dim K (NewFlatI8 quantizes them to int8). Values
// are irrelevant to timing and to parity (GPU and CPU compute the same int32 dot); only recall
// would need real embeddings, which this measurement does not report.
func randCorpus(rng *rand.Rand, n, K int) [][]float32 {
	out := make([][]float32, n)
	for i := range n {
		v := make([]float32, K)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		out[i] = v
	}
	return out
}

// bestQuery times FlatI8.Query over `queries`, returning the min per-query wall time and the
// hits of the last pass (for parity).
func bestQuery(f *ann.FlatI8, queries [][]float32, k, iters int) (time.Duration, [][]ann.Hit) {
	for range 3 {
		for _, q := range queries {
			f.Query(q, k)
		}
	}
	best := time.Hour
	hits := make([][]ann.Hit, len(queries))
	for range iters {
		t0 := time.Now()
		for i, q := range queries {
			hits[i] = f.Query(q, k)
		}
		if d := time.Since(t0) / time.Duration(len(queries)); d < best {
			best = d
		}
	}
	return best, hits
}

// TestMetalGemvBandwidth measures the ceiling then the single-query GEMV against it. It reports
// (does not assert perf) and checks GPU/CPU score parity (worst Δ must be 0 — same int32 dot).
func TestMetalGemvBandwidth(t *testing.T) {
	if os.Getenv("AIKIT_GPU_BENCH") == "" {
		t.Skip("periodic GPU pass — set AIKIT_GPU_BENCH=1 to run")
	}
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer func() { dev.ReleaseObjects(); dev.ReleaseAll() }()
	q := dev.NewCommandQueue()

	t.Logf("device: %s", dev.Name())
	ceiling := measureBandwidth(t, dev, q, 512<<20) // 512 MiB, past any cache
	t.Logf("STREAMING-READ CEILING: %.1f GB/s\n", ceiling)

	rng := rand.New(rand.NewSource(1))
	const k, nq, iters = 10, 200, 20
	for _, N := range []int{200_000, 500_000} {
		for _, K := range []int{256, 768} {
			vecs := randCorpus(rng, N, K)
			qs := randCorpus(rng, nq, K)
			f := ann.NewFlatI8(vecs)

			cpuBest, cpuHits := bestQuery(f, qs, k, iters)
			if err := f.EnableGPU(); err != nil {
				t.Fatalf("EnableGPU: %v", err)
			}
			gpuBest, gpuHits := bestQuery(f, qs, k, iters)

			// Parity: same top-k index set AND identical scores (int32 dot ⇒ exact).
			worst := 0.0
			for i := range gpuHits {
				cm := map[int]float64{}
				for _, h := range cpuHits[i] {
					cm[h.Index] = h.Score
				}
				for _, h := range gpuHits[i] {
					cs, ok := cm[h.Index]
					if !ok {
						worst = math.Inf(1)
						continue
					}
					if d := math.Abs(h.Score - cs); d > worst {
						worst = d
					}
				}
			}

			gpuBytes := float64(N) * float64(K) // int8 codes streamed per query
			gpuGBps := gpuBytes / gpuBest.Seconds() / 1e9
			t.Logf("N=%-7d K=%-3d  cpu %7.3f ms/q  gpu %7.3f ms/q  %5.2f×  gpu %6.1f GB/s (%4.1f%% of ceil)  worstΔ %.3e",
				N, K, float64(cpuBest.Nanoseconds())/1e6, float64(gpuBest.Nanoseconds())/1e6,
				cpuBest.Seconds()/gpuBest.Seconds(), gpuGBps, 100*gpuGBps/ceiling, worst)
			if worst != 0 {
				t.Errorf("N=%d K=%d: GPU/CPU score parity Δ=%.3e, want 0 (same int32 dot)", N, K, worst)
			}
			f.Close()
		}
	}
}

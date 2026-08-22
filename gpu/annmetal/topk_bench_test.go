//go:build darwin

package annmetal

// topk_bench_test.go measures the on-device top-k selection (topk_rows) in ISOLATION —
// the second dispatch of FlatI8.TopKBatch, after the tiled GEMM has produced the M*N score
// matrix. The roofline campaign (docs/internal/roofline-2026-08.md §6) flagged the CUDA
// topk_rows as the batch path's largest remaining cost; this measures whether the Metal
// mirror has the same problem, against a ceiling measured on THIS box.
//
// topk_rows reads the whole M*N score matrix once (each thread strip-scans its slice), so
// its floor is set by streaming-read bandwidth: t_floor = 4*M*N bytes / BW. A run far below
// that fraction is spending its time somewhere other than the read — here, the serial merge
// thread 0 does over all topkTG*k partial candidates.
//
// Env-gated (AIKIT_GPU_BENCH=1) like the other GPU benches; a plain `go test` stays green.

import (
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// tkBwSrc streams a buffer with float4 grid-stride loads and reduces (cannot be DCE'd) — the
// read-bandwidth ceiling topk_rows is judged against. Same probe shape as gemv_bench's bwSrc.
const tkBwSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void bw_read(
    device const float4* src [[buffer(0)]],
    constant uint&       n4  [[buffer(1)]],
    constant uint&       gsz [[buffer(2)]],
    device float*        sink[[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    float4 acc = float4(0.0);
    for (uint i = gid; i < n4; i += gsz) acc += src[i];
    sink[gid & 1023u] = acc.x + acc.y + acc.z + acc.w;
}`

// tkMeasureBW returns the best sustained GB/s of a stream-read far past cache.
func tkMeasureBW(t *testing.T, dev *gpu.Device, q gpu.Queue, bytes int) float64 {
	t.Helper()
	lib, err := dev.CompileLibrary(tkBwSrc, gpu.MSL3_1)
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
	fv := src.Floats()
	for i := range fv {
		fv[i] = 1.0
	}
	n4 := nFloats / 4
	n4buf := gpu.NewBufferOf(dev, []uint32{uint32(n4)})
	defer dev.ReleaseBuf(n4buf)
	best := 0.0
	for _, gsz := range []int{1 << 18, 1 << 20, 1 << 22} {
		gszBuf := gpu.NewBufferOf(dev, []uint32{uint32(gsz)})
		secs := math.Inf(1)
		for range 5 {
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
		if gbps := float64(bytes) / secs / 1e9; gbps > best {
			best = gbps
		}
	}
	return best
}

// TestMetalTopKBandwidth measures topk_rows in isolation against the streaming ceiling. It
// fills a device score matrix with random floats (so the merge does real work and no strip is
// degenerate), times the topk dispatch alone, and reports GB/s vs the 4*M*N read floor.
func TestMetalTopKBandwidth(t *testing.T) {
	if os.Getenv("AIKIT_GPU_BENCH") == "" {
		t.Skip("periodic GPU pass — set AIKIT_GPU_BENCH=1 to run")
	}
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer func() { dev.ReleaseObjects(); dev.ReleaseAll() }()
	q := dev.NewCommandQueue()

	lib, err := dev.CompileLibrary(w8a8Src, gpu.MSL3_1)
	if err != nil {
		t.Fatalf("compile w8a8Src: %v", err)
	}
	topk, err := dev.NewComputePipeline(lib, "topk_rows")
	if err != nil {
		t.Fatalf("pipeline topk_rows: %v", err)
	}

	t.Logf("device: %s", dev.Name())
	ceiling := tkMeasureBW(t, dev, q, 512<<20)
	t.Logf("STREAMING-READ CEILING: %.1f GB/s", ceiling)

	rng := rand.New(rand.NewSource(1))
	const k = 10
	for _, N := range []int{200_000, 500_000} {
		for _, M := range []int{16, 64, 256} {
			scores := make([]float32, M*N)
			for i := range scores {
				scores[i] = rng.Float32()*2 - 1
			}
			scoreBuf := gpu.NewBufferOf(dev, scores)
			nBuf := gpu.NewBufferOf(dev, []uint32{uint32(N)})
			kBuf := gpu.NewBufferOf(dev, []uint32{uint32(k)})
			idxOut := gpu.NewBufferOf(dev, make([]uint32, M*k))
			scoreOut := dev.NewBufferLen(M * k)

			secs := math.Inf(1)
			for range 8 {
				runtime.LockOSThread()
				e := q.Begin()
				e.Dispatch(topk, M*topkTG, topkTG, scoreBuf, nBuf, kBuf, idxOut, scoreOut)
				e.End()
				runtime.UnlockOSThread()
				if err := e.Err(); err != nil {
					t.Fatalf("topk dispatch: %v", err)
				}
				if s := e.GPUEnd() - e.GPUStart(); s > 0 && s < secs {
					secs = s
				}
			}
			bytes := float64(M) * float64(N) * 4 // M*N f32 scores read once
			gbps := bytes / secs / 1e9
			t.Logf("N=%-7d M=%-3d  topk %7.3f ms  %6.1f GB/s (%4.1f%% of ceil)", N, M, secs*1e3, gbps, 100*gbps/ceiling)

			for _, b := range []gpu.Buffer{scoreBuf, nBuf, kBuf, idxOut, scoreOut} {
				dev.ReleaseBuf(b)
			}
		}
	}
}

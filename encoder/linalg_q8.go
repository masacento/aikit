package encoder

import (
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// matmulBTQ8 is the M8 int8-weight variant of matmulBT. Same shape:
// dst = a · bᵀ where a is [M, K] f32 (activations) and b is logically
// [N, K] f32 stored as int8 quantized rows + per-row f32 scales:
//
//	a       — [M, K] f32 row-major (activations, NOT quantized)
//	bQ      — [N*K] int8 row-major (the quantized [N, K] matrix)
//	bScales — [N] f32 (per-row scale: row n's f32 value ≈ float32(bQ[n,k]) * bScales[n])
//	dst     — [M, N] f32 row-major, freshly allocated
//
// The kernel is the M3 blocked-matmul with one twist: weight reads come
// from a tightly-packed int8 array (4× less memory bandwidth than f32),
// the multiply-accumulate happens in f32 (each int8 weight gets
// converted to f32 inside the inner loop), and the final accumulator
// is scaled by the row's bScale once per (i, n) tile cell at write-back.
//
// Why this saves time: at M3's blocked-kernel measurement, the GEMM
// was bandwidth-bound on weight reads at ~6.5 GFLOP/s. Reducing weight
// bytes 4× pushes the bound out by ~3× (memory subsystem can deliver
// 4× more weights per cycle, but per-multiply work is unchanged; the
// f32×f32 multiply itself doesn't get faster). Empirically the win
// lands closer to 2× than 4× because Go's compiler doesn't auto-SIMD
// the int8-to-f32 conversion as tightly as the pure-f32 inner loop.
//
// Dispatch: matmulBTQ8 is always blocked (the M8 path is only
// triggered for the big linear layers Wqkv/OutProj/fc11/fc12/fc2,
// all of which have M*K*N ≫ the matmulBT small-shape threshold).
func matmulBTQ8(a []float32, bQ []int8, bScales []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	// The arm64 fused path widens inside the pack and needs no deqW; only the
	// dequant-then-GEMM fallback does, so don't allocate it when it won't be read.
	var deqW []float32
	if !linalg.FusedQ8Applies(K, N) {
		deqW = make([]float32, N*K)
	}
	matmulBTQ8Into(dst, a, bQ, bScales, M, K, N, deqW)
	return dst
}

// matmulBTQ8Into is matmulBTQ8 writing into a caller-supplied dst[:M*N] and using a
// caller-supplied deqW[:N*K] weight-dequant buffer — both pooled in the q8 forward's
// scratch. It widens each int8 weight to f32 ONCE (N*K) into deqW, then runs the
// vectorized f32 matmulBTInto.
//
// This replaced a scalar blocked kernel that did the int8→f32 widen INLINE in the
// GEMM (so the conversion ran M times per weight), which measured ~26× slower than
// the f32 SIMD matmul and ~36× slower than the SDOT W8A8 kernel on the Wqkv shape —
// the actual reason LoadQ8 was ~5× slower than Load, NOT the allocation churn the
// pooling fix already removed. Dequant-then-SIMD keeps the weight-only numerics
// exactly (cosine vs f32 unchanged at 0.997), unlike full W8A8 which quantizes
// activations and fell below the 0.97 reranker bar; the weights stay int8 in storage
// (¼ the //go:embed footprint) — deqW is transient runtime scratch only.
func matmulBTQ8Into(dst, a []float32, bQ []int8, bScales []float32, M, K, N int, deqW []float32) {
	if len(a) != M*K || len(bQ) != N*K || len(bScales) != N || len(dst) < M*N {
		panic("encoder: matmulBTQ8Into shape mismatch")
	}
	// FUSED (arm64): widen int8→f32 INSIDE the b-panel pack (perf-campaign item 22
	// fix (b)), so the full [N,K] f32 weight is never materialized — killing the
	// ~0.9 GB/forward deqW round-trip through DRAM that fix (a) still paid. Needs the
	// NEON packed kernel and N%8==0 (every encoder Q8 shape: N∈{768,2304,3072}).
	if linalg.FusedQ8Applies(K, N) {
		matmulBTQ8FusedDispatch(dst, a, bQ, bScales, M, K, N)
		return
	}
	// FALLBACK (amd64 / odd N): widen the whole matrix into pooled deqW ONCE, then run
	// the vectorized f32 matmulBTInto over it (fix (a): VPMOVSXBD/VCVTDQ2PS/VMULPS AVX2
	// widen). The widen is O(N*K), INDEPENDENT of M, so it does not amortize with
	// sequence length; both paths are bit-identical — float32(int8) is exact and the
	// scale is one f32 multiply either way.
	if len(deqW) < N*K {
		panic("encoder: matmulBTQ8Into deqW too small")
	}
	w := deqW[:N*K]
	linalg.DequantizeRowsInt8Into(w, bQ, bScales, N, K)
	matmulBTInto(a, w, dst, M, K, N)
}

// matmulBTQ8FusedDispatch mirrors matmulBTInto's intra-op parallelism for the fused
// int8 kernel — and it must mirror it EXACTLY, both axes. wantParallelCols is only true
// for SMALL M (M < minRowsPerWorker·numCPU); above that it hands off to the row split.
// Dropping the row branch made large-M fc2 (w256, the long-sequence rerank) run SERIAL
// while the f32 baseline row-parallelised — measured 43% slower end-to-end even though
// the fused kernel itself is faster. So: columns for small M, rows for large M, serial
// only when neither clears its threshold. The in-flight guard inside both predicates
// keeps EncodeBatch (every core already busy with sibling forwards) on the serial path.
func matmulBTQ8FusedDispatch(dst, a []float32, bQ []int8, bScales []float32, M, K, N int) {
	if wantParallelCols(M, K, N) {
		linalg.MatmulBTQ8Fused(dst, a, bQ, bScales, M, K, N)
		return
	}
	if wantParallelMatmul(M, K, N) {
		matmulBTQ8FusedRowsParallel(dst, a, bQ, bScales, M, K, N)
		return
	}
	linalg.MatmulBTQ8FusedInto(dst, a, bQ, bScales, M, K, N)
}

// matmulBTQ8FusedRowsParallel is the fused twin of matmulBTBlockedIntoParallel: split
// the M output rows across workers, each running the serial fused kernel over its
// disjoint row block (reading a[iStart:iEnd] and the shared read-only int8 weights,
// writing dst[iStart:iEnd] — no overlap, no locking). Each worker widen-packs the whole
// [N,K] weight for its rows; that re-widens per worker, but the row split is the
// large-M regime where the widen is a vanishing fraction of the GEMM anyway.
func matmulBTQ8FusedRowsParallel(dst, a []float32, bQ []int8, bScales []float32, M, K, N int) {
	workers := runtime.NumCPU()
	maxByRows := (M + minRowsPerWorker - 1) / minRowsPerWorker
	if workers > maxByRows {
		workers = maxByRows
	}
	if workers <= 1 {
		linalg.MatmulBTQ8FusedInto(dst, a, bQ, bScales, M, K, N)
		return
	}
	rowsPer := (M + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		iStart := w * rowsPer
		if iStart >= M {
			break
		}
		iEnd := min(iStart+rowsPer, M)
		wg.Go(func() {
			m := iEnd - iStart
			linalg.MatmulBTQ8FusedInto(dst[iStart*N:iEnd*N], a[iStart*K:iEnd*K], bQ, bScales, m, K, N)
		})
	}
	wg.Wait()
}

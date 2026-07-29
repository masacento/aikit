package encoder

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// Intra-op (within-one-forward) matmul parallelism for the standalone-
// Encode latency case.
//
// The hard constraint: EncodeBatch already saturates every core by
// running runtime.NumCPU() forward workers concurrently (model.go).
// Parallelizing the matmul *inside* a forward would oversubscribe in
// that mode. But a lone Encode() call — one text, nothing else running
// — leaves N-1 cores idle, and a single large forward's matmuls (fc11,
// fc2, wqkv at L≥~256) are big enough to split across them profitably.
//
// We distinguish the two modes WITHOUT threading a flag through the
// hot, shared selfAttention/swigluMLP code (which both forward and
// forwardBatch call): every forward entry point bumps inflightForwards
// on the way in. The matmul parallelizes its row dimension only when
// the count is ≤1 (this forward is the only one in flight) AND the
// shape clears parallelThreshold. Under EncodeBatch the count is
// NumCPU>1, so the matmul stays byte-for-byte on the serial blocked
// path — the batch path is behaviorally unchanged.
//
// Numerics: splitting the M (row) dimension is exact. Each output row
// depends only on its own a-row and the shared b; the row range a
// worker computes is tiled identically to the serial path, so results
// are bit-identical to matmulBTBlockedInto (no new f32 reduction-order
// noise — unlike changing the K-tiling would introduce).
var inflightForwards atomic.Int32

// enterForward/leaveForward bracket one forward pass for the in-flight
// accounting above. Every forward variant (f32/q8, single/batch) must
// pair these so the parallelism gate sees an accurate concurrent-
// forward count.
func enterForward() { inflightForwards.Add(1) }
func leaveForward() { inflightForwards.Add(-1) }

// parallelThreshold is the per-call FLOP count (M*K*N) at/above which
// row-splitting pays off. Tuned on an M1 Pro (8 core) against
// parallel_test's BenchmarkMatmul{Parallel,Serial}_*; goroutine spawn
// is cheap enough (~µs, vs sub-ms matmuls) that even small shapes win:
//
//	shape (M,K,N)        FLOP    serial     parallel   speedup
//	L80 outproj 80,768,768   47M  2.02 ms    0.76 ms    2.6×
//	L80 fc11 80,768,3072    188M  8.59 ms    2.92 ms    2.9×
//	L256 fc11 256,768,3072  604M  27.6 ms    6.37 ms    4.3×
//	L512 fc11 512,768,3072  1.2G  54.5 ms    12.1 ms    4.5×
//
// 32M sits above the per-head attention QKᵀ shape (L512: 512·64·512 ≈
// 17M), keeping those serial: they win in isolation too (~3.4×) but
// recur 12 heads × 12 layers = 144×/forward (1000+ goroutine spawns),
// whose NET effect needs an end-to-end forward benchmark on real
// weights to judge — a follow-up, not guessed at here. Every f32
// linear layer (wqkv/fc11/fc12/fc2/outproj at L≥80) clears 32M and
// parallelizes. Sits well above matmulBT's 4M naive/blocked cutoff.
const parallelThreshold = 32_000_000

// minRowsPerWorker keeps each goroutine's slice fat enough to be worth
// a spawn (and ≥ the linalg blocked-GEMM mBlock tile, so no worker gets a
// sub-tile sliver).
const minRowsPerWorker = 32

// wantParallelMatmul reports whether matmulBTInto should row-split this
// call across cores: only when no other forward is in flight and the
// shape is both large enough (FLOPs) and tall enough (rows to split).
func wantParallelMatmul(M, K, N int) bool {
	if inflightForwards.Load() > 1 {
		return false
	}
	if int64(M)*int64(K)*int64(N) < parallelThreshold {
		return false
	}
	return M >= 2*minRowsPerWorker
}

// wantParallelCols reports whether a matmul the ROW split declined should
// instead fan across output columns.
//
// It is deliberately the row gate's complement, not its rival: where M is tall
// enough to split, the row split wins (it is the tuned, measured path, and its
// workers share no output cache lines). wantParallelCols only picks up the
// shapes the row split structurally cannot serve — M < 2*minRowsPerWorker.
//
// Those shapes are not exotic. A SPLADE query or a reranker pair is often ~20
// tokens, and M is the token count: at L=22 every projection in the BERT trunk
// (wqkv 22·768·2304, fc11 22·768·3072, fc2 22·3072·768) clears the FLOP
// threshold by 1.5-2× yet fails M ≥ 64, so the ENTIRE forward ran on one core
// while 15 sat idle. Measured on a 3700X, that left the trunk at 128 ms of a
// 153 ms SPLADE.Expand.
//
// It also removes an interference effect that is invisible in a per-matmul
// benchmark. Mixing an all-core burst into an otherwise serial forward costs the
// serial part real time: a memory-free all-core burst alone slowed the following
// serial trunk 12.8% (boost clock), and the real column-parallel vocabulary
// projection slowed it 32% (boost plus cache). That is why parallelizing only
// the vocabulary projection made short-query Expand 6.7% SLOWER end to end even
// though the projection itself got 5× faster — the fix is for the trunk to be
// parallel too, not for the projection to stay serial.
//
// N is checked against the row gate's per-worker minimum for symmetry: a
// fan-out needs enough columns to be worth the spawn, and linalg.MatmulBT
// applies its own threshold below this one.
func wantParallelCols(M, K, N int) bool {
	if inflightForwards.Load() > 1 {
		return false
	}
	if M >= 2*minRowsPerWorker {
		return false // the row split serves this shape; prefer it
	}
	if int64(M)*int64(K)*int64(N) < parallelThreshold {
		return false
	}
	return N >= 2*minRowsPerWorker
}

// matmulBTColsInto computes dst[M,N] = a[M,K]·b[N,K]ᵀ, fanning across the N
// (output-column) dimension instead of the M (row) dimension.
//
// This is the counterpart to matmulBTInto for the shape where a row split
// cannot help. The SPLADE vocabulary projection is [L,768]·[30522,768]ᵀ: N is
// the 30k vocabulary, but M is a *query* length — often ~20 tokens. That fails
// wantParallelMatmul's M ≥ 2*minRowsPerWorker test, which is the right test for
// a row split and exactly wrong here, and left a ~500 MFLOP matmul on one core
// for every short query (perf-campaign item 7).
//
// The in-flight guard is the same contract wantParallelMatmul documents: under
// EncodeBatch every core is already busy with sibling forwards, so this stays on
// the serial fill and the batch path is behaviorally unchanged. No FLOP gate is
// needed — linalg.MatmulBT applies its own (parallelCols) and runs the span
// inline below it.
//
// Numerics: linalg.MatmulBT shards output columns 8-aligned and runs the same
// blockedFill per shard as the serial path, so each dst[i,j] is bit-identical to
// matmulBTBlockedInto at any fan-out width — linalg's documented M-invariance
// plus TestParallelWidth_bitIdentical; re-gated end-to-end here by
// TestSpladeVocabProj_colParallelIsBitIdentical.
func matmulBTColsInto(a, b, dst []float32, M, K, N int) {
	if inflightForwards.Load() > 1 {
		matmulBTBlockedInto(a, b, dst, M, K, N)
		return
	}
	linalg.MatmulBT(a, b, dst, M, K, N)
}

// matmulBTBlockedIntoParallel splits the M (row) dimension across
// NumCPU goroutines, each computing a disjoint, contiguous block of
// output rows via the serial blocked fill. dst MUST have len ≥ M*N.
//
// Each worker writes dst[iStart*N:iEnd*N] and reads a[iStart*K:iEnd*K]
// + the shared read-only b — disjoint writes, no locking. dst is
// zeroed once up front (the blocked fill accumulates into a zeroed
// region per its k-tile contract).
func matmulBTBlockedIntoParallel(a, b, dst []float32, M, K, N int) {
	if len(dst) < M*N {
		panic("encoder: matmulBTBlockedIntoParallel dst too small")
	}
	workers := runtime.NumCPU()
	maxByRows := (M + minRowsPerWorker - 1) / minRowsPerWorker
	if workers > maxByRows {
		workers = maxByRows
	}
	if workers <= 1 {
		matmulBTBlockedInto(a, b, dst, M, K, N)
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
		wg.Add(1)
		go func(iStart, iEnd int) {
			defer wg.Done()
			m := iEnd - iStart
			// Each worker owns a disjoint row block; MatmulBTInto zeroes and
			// fills its own dst sub-slice, so no shared up-front zeroing.
			linalg.MatmulBTInto(dst[iStart*N:iEnd*N], a[iStart*K:iEnd*K], b, m, K, N)
		}(iStart, iEnd)
	}
	wg.Wait()
}

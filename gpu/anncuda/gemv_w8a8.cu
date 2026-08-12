// gemv_w8a8.cu — the W8A8 scoring kernels behind the CUDA ann.Backend
// (docs/task-native-gpu.md, Phase 1b). `gemv_w8a8` scores ONE query against the whole
// index (FlatI8.Query); `gemv_w8a8_batch8`/`gemv_w8a8_batch16` score M queries in one
// launch (FlatI8.QueryBatch, FlatI8.TopKBatch); `topk_rows` selects on the device.
//
// This no longer mirrors gpu/annmetal/backend.go's MSL kernel-for-kernel. Both
// backends started from the same two-kernel sketch, and both have since been
// measured against their own device's ceilings, which pushed them apart — the batch
// path here is a lane-group GEMV where Metal's is a threadgroup-tiled GEMM. The
// PARITY contract is what the two share, not the kernel shape.
//
// BIT-IDENTITY-EXEMPT: structurally immune rather than merely uncontracted. The dot
// products accumulate into an `int` — integer addition is associative, so the reduction
// is exact and order-independent — and the float tail is `(float)acc * qscale * scale`,
// two multiplies with NO add. There is no multiply-accumulate here for a compiler to
// contract, which the shipped PTX confirms: 0 fma.rn.f32 against 50 mul.f32 (the count
// rose with the batch kernels' unrolled epilogues; what matters is that the fma count
// is ZERO, since there is nothing to fuse). Adding a float accumulate or a bias term
// would end that, so re-classify if this kernel grows one.
//
// All compute the EXACT int32 dot of the host-quantized int8 query against each
// row, then apply the query/row rescale — the same value linalg.MatmulBTW8A8
// produces on the CPU. Integer accumulation is exact, so GPU and CPU rank
// identically (backend_test.go gates that, break-it-first).
//
// BOTH SCORING KERNELS WERE ONCE DELIBERATELY UNTUNED — "no __dp4a, no warp-per-row
// reduction, no vectorized loads", on the reasoning that the tuned blob is goinfer's
// and this is only the proving kernel. Measurement retired that twice, for the same
// underlying reason: a proving kernel is fine, but a proving kernel wired to a
// production entry point is a pessimization.
//
//   - gemv_w8a8 (FlatI8.Query) was thread-per-row, so adjacent threads read addresses
//     K bytes apart and a warp's load touched 32 cache lines to use one byte of each.
//     25-28 GB/s against a measured 412 GB/s streaming ceiling. Now warp-per-row with
//     __dp4a: 3.3-6.0x (be90aec).
//   - The batch path (FlatI8.QueryBatch) ran gpu/vit.cu's 16x16 gemm_w8a8_tiled, whose
//     cost depends on ceil(M/16) rather than M — 6.79 ms at N=200k for every M from 1
//     to 16. Now the lane-group batch GEMV below: 6.4-12.7x, and the tiled kernel is
//     no longer reachable from this backend at any M.
//
// The lesson both share is in docs/internal/roofline-2026-08.md: the denominator has
// to be measured before the kernel can be called fast or slow.
//
// Two shape conventions carried over from the Metal side:
//   - Scalars arrive as 1-element BUFFERS, not by-value kernel params, mirroring
//     MSL's `constant uint&` binds. That is what lets gpu.Queue.Run1D keep Metal's
//     `bufs ...Buffer` signature on both platforms.
//   - Each kernel takes its own thread count as a trailing param and guards on it:
//     a CUDA launch rounds up to whole blocks, where Metal's dispatchThreads
//     launches exactly n (gpu/cuda.go, divergence 2).

// gemv_w8a8: out[j] = dot(qi8, codes[j]) * qscale * scales[j], one WARP per row.
extern "C" __global__ void gemv_w8a8(
    const signed char* __restrict__ codes,  // [N*K] int8, row-major
    const signed char* __restrict__ qi8,    // [K]   int8 query (host-quantized)
    const float* __restrict__ scales,       // [N]   per-row scale
    const unsigned int* __restrict__ K,
    const float* __restrict__ qscale,       // query scale (0 => all-zero query)
    float* __restrict__ out,                // [N]   scores
    const unsigned int* __restrict__ N)     // row count (the launch bound)
{
    // ONE WARP PER ROW. The 32 lanes walk the SAME row, so a warp's load covers 32
    // consecutive addresses — one fully-used transaction, against the 32 partly-used
    // ones the thread-per-row form issued. The launch therefore dispatches N*32
    // threads (see cudaI8Index.Score); j is the global WARP index, not thread index.
    unsigned int warpsPerBlock = blockDim.x >> 5;
    unsigned int j = blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    if (j >= *N) return;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int k = *K;
    const signed char* row = codes + (unsigned long long)j * k;

    int acc = 0;
    if ((k & 3u) == 0) {
        // 4 int8 MACs per instruction via __dp4a, and 4 bytes per lane per load, so a
        // warp moves 128 bytes at a time. Safe because k%4==0 makes every row start
        // 4-byte aligned (the base allocation is 256-byte aligned).
        const int* row4 = (const int*)row;
        const int* q4 = (const int*)qi8;
        unsigned int k4 = k >> 2;
        for (unsigned int i = lane; i < k4; i += 32) {
            acc = __dp4a(row4[i], q4[i], acc);
        }
    } else {
        // Byte path for k%4 != 0: still coalesced (32 consecutive bytes per warp),
        // just one MAC per lane per step instead of four.
        for (unsigned int i = lane; i < k; i += 32) {
            acc += (int)qi8[i] * (int)row[i];
        }
    }

    // Warp reduction. Integer addition is associative, so this is EXACTLY the
    // sequential sum — the reason this kernel can be reshaped at all, and why it
    // stays BIT-IDENTITY-EXEMPT rather than needing a contract.
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[j] = (float)acc * (*qscale) * scales[j];
    }
}

// ---------------------------------------------------------------------------
// gemv_w8a8_batch{8,16}: the BATCHED form — out[m*N+j] for M queries — as one
// group of lanes per corpus row, with the query tile staged in shared memory so
// every loaded corpus row is reused across all QTILE queries in the tile.
//
// WHY THIS REPLACED THE TILED GEMM. This path used to run gpu/vit.cu's
// gemm_w8a8_tiled, which stages a 16x16 output tile, so its wall time depends on
// ceil(M/16) rather than on M: measured on nvidia-rtx2070s at N=200k K=768 it took
// 6.79 ms for EVERY M from 1 to 16, and 109 ms at M=256. Looping the single-query
// gemv_w8a8 costs 0.47 ms per query, so the tiled kernel did not break even until
// M ~ 15 and never beat that loop by more than 9%. Both sat far below both roofs
// (360 GMAC/s against a measured 4876 GMAC/s int32-MAD ceiling; 45 GB/s against a
// measured 412 GB/s streaming ceiling) because a 16x16 byte-wise tile spends its
// time on shared-memory traffic and __syncthreads, not on the dot. The constant
// that selected it (gemmTileMinM = 2) was calibrated when the single-query GEMV was
// still thread-per-row and 5.6x slower — the tiled kernel only ever looked good
// against a broken baseline. See docs/internal/roofline-2026-08.md.
//
// THE TWO PARAMETERS WERE SWEPT, NOT REASONED. LANES (lanes cooperating on one row)
// and QTILE (queries per tile) trade off against each other: the shuffle reduction
// costs QTILE * log2(LANES) per row, so a wider query tile wants FEWER lanes, while
// coalescing wants more (8 lanes * 4 B = 32 B, exactly one sector). Measured at
// N=200k K=768, ms per launch, best in each column marked *:
//
//        M=1     M=2     M=4     M=8    M=16    M=64   M=256
//  L8Q8   0.571*  0.547*  0.587*  0.664*  1.267   4.858  19.450
//  L4Q16  0.908   0.914   0.922   0.956   1.082*  3.935*  15.513*
//  L8Q16  0.919   0.921   0.944   1.026   1.206   4.465  17.360
//  L2Q8   0.978   0.981   0.982   0.993   1.940   7.531  24.970
//  L32Q16 1.893   —       —       —       2.402   9.622  —
//
// Hence L8Q8 for M <= 8 and L4Q16 above it. The all-32-lane form at the bottom is
// what a straight port of the single-query kernel would have given: 2.2x off.
//
// QTILE ACCUMULATORS ARE ALWAYS COMPUTED, EVEN WHEN M < QTILE. The alternative — a
// runtime-bounded loop over mc — indexes acc[] dynamically, which spills the array
// from registers to local memory and costs far more than the padded MACs. The
// staging loop zero-fills the unused query rows, so the padded accumulators compute
// a defined zero and are simply not stored. That is what the M=1..8 column above is
// paying for, and it is still an order of magnitude better than the tiled kernel.
//
// BIT-IDENTITY: same structural exemption as gemv_w8a8 — int accumulation is
// associative, so the reduction over LANES equals the sequential sum, and the float
// tail is two multiplies with no add. out[m*N+j] is exactly the value gemv_w8a8
// writes for query m, which TestCUDAGEMM_batchMatchesQuery gates as exact equality.
// ---------------------------------------------------------------------------

template <int LANES, int QTILE>
__device__ __forceinline__ void batch_body(
    const signed char* __restrict__ codes, const signed char* __restrict__ qi8,
    const float* __restrict__ scales, const unsigned int* __restrict__ K,
    const float* __restrict__ qscale, float* __restrict__ out,
    const unsigned int* __restrict__ N, const unsigned int* __restrict__ M)
{
    // Dynamic shared memory, declared as int so the 4-byte reads below are aligned;
    // the launch sizes it at QTILE*K bytes (gpu.Buffer-free, so it costs no upload).
    extern __shared__ int qsmem[];
    signed char* qs = (signed char*)qsmem;

    unsigned int k = *K, n = *N, m = *M;
    unsigned int m0 = blockIdx.y * QTILE;
    unsigned int mc = (m - m0 < QTILE) ? (m - m0) : QTILE;

    // Stage this tile's queries, zero-filling rows past mc. EVERY thread in the block
    // runs this loop and reaches the barrier, which is why the row-bound return below
    // is placed after __syncthreads() — returning first would strand the survivors.
    for (unsigned int i = threadIdx.x; i < (unsigned int)QTILE * k; i += blockDim.x) {
        unsigned int t = i / k;
        qs[i] = (t < mc) ? qi8[(unsigned long long)(m0 + t) * k + (i - t * k)] : (signed char)0;
    }
    __syncthreads();

    unsigned int groupsPerBlock = blockDim.x / LANES;
    unsigned int j = blockIdx.x * groupsPerBlock + (threadIdx.x / LANES);
    if (j >= n) return;
    unsigned int lane = threadIdx.x & (unsigned int)(LANES - 1);
    const signed char* row = codes + (unsigned long long)j * k;

    int acc[QTILE];
    #pragma unroll
    for (int t = 0; t < QTILE; t++) acc[t] = 0;

    if ((k & 3u) == 0) {
        // 4 int8 MACs per instruction via __dp4a, 4 bytes per lane per load. k%4==0
        // makes every row start 4-byte aligned (the base allocation is 256-byte
        // aligned), and the shared tile inherits qsmem's alignment.
        const int* row4 = (const int*)row;
        const int* qs4 = (const int*)qs;
        unsigned int k4 = k >> 2;
        for (unsigned int i = lane; i < k4; i += LANES) {
            int rv = row4[i]; // ONE global load feeds all QTILE dp4a's — the whole point
            #pragma unroll
            for (int t = 0; t < QTILE; t++) {
                acc[t] = __dp4a(rv, qs4[(unsigned int)t * k4 + i], acc[t]);
            }
        }
    } else {
        // Byte path for k%4 != 0: still coalesced, one MAC per lane per step.
        for (unsigned int i = lane; i < k; i += LANES) {
            int rv = (int)row[i];
            #pragma unroll
            for (int t = 0; t < QTILE; t++) {
                acc[t] += rv * (int)qs[(unsigned int)t * k + i];
            }
        }
    }

    // One reduction per query. The shuffles are warp-wide but the tree is only
    // log2(LANES) deep, so the lanes of the other row-groups sharing this warp
    // reduce their own rows in the same instructions.
    #pragma unroll
    for (int t = 0; t < QTILE; t++) {
        int a = acc[t];
        for (int off = LANES / 2; off > 0; off >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, off);
        }
        if (lane == 0 && (unsigned int)t < mc) {
            out[(unsigned long long)(m0 + t) * n + j] = (float)a * qscale[m0 + t] * scales[j];
        }
    }
}

// gemv_w8a8_batch8 — 8 lanes per row, 8 queries per tile. For M <= 8.
extern "C" __global__ void gemv_w8a8_batch8(
    const signed char* __restrict__ codes,  // [N*K] int8, row-major
    const signed char* __restrict__ qi8,    // [M*K] int8 queries (host-quantized)
    const float* __restrict__ scales,       // [N]   per-row scale
    const unsigned int* __restrict__ K,
    const float* __restrict__ qscale,       // [M]   per-query scale
    float* __restrict__ out,                // [M*N] scores, row-major
    const unsigned int* __restrict__ N,
    const unsigned int* __restrict__ M)     // query count
{
    batch_body<8, 8>(codes, qi8, scales, K, qscale, out, N, M);
}

// gemv_w8a8_batch16 — 4 lanes per row, 16 queries per tile. For M > 8.
extern "C" __global__ void gemv_w8a8_batch16(
    const signed char* __restrict__ codes,
    const signed char* __restrict__ qi8,
    const float* __restrict__ scales,
    const unsigned int* __restrict__ K,
    const float* __restrict__ qscale,
    float* __restrict__ out,
    const unsigned int* __restrict__ N,
    const unsigned int* __restrict__ M)
{
    batch_body<4, 16>(codes, qi8, scales, K, qscale, out, N, M);
}

// topk_rows — per-query top-k selection ON THE DEVICE, so a batch returns M*k hits
// instead of the whole M*N score matrix. On a discrete GPU that readback is a real
// PCIe copy (at N=1e6, batch=256 it is ~1 GB), which is the cost this exists to avoid.
//
// THE ORDER MUST MATCH ann.FlatI8.topHits EXACTLY, or the GPU returns a different
// (still plausible) set and parity silently degrades: score DESCENDING, and on equal
// scores the LOWER INDEX wins. That tie-break is not decoration — an all-equal-scores
// index must yield [0..k-1], and the test fixture asserts precisely that.
//
// Algorithm: k passes of a block-wide argmax over the query's row, each pass writing
// -FLT_MAX into the winner's slot so the next pass cannot re-select it. The scores
// buffer is therefore MUTATED — it is per-call scratch, never the caller's data. k is
// small (a retrieval k, not a sort), so k*N/threads work per query is cheap next to the
// GEMM that produced the row.
//
// A slot already consumed reads back as -FLT_MAX and can never win: `v > best` is false
// against the initial -FLT_MAX sentinel, and the tie-break branch is guarded on a valid
// index. Real dot-product scores do not reach -FLT_MAX.
#define TKBLOCK 256

extern "C" __global__ void topk_rows(
    float* __restrict__ scores,      // [M*N] — MUTATED (winners consumed)
    int* __restrict__ outIdx,        // [M*k]
    float* __restrict__ outScore,    // [M*k]
    const unsigned int* __restrict__ Mp,
    const unsigned int* __restrict__ Np,
    const unsigned int* __restrict__ kp)
{
    unsigned int M = *Mp, N = *Np, k = *kp;
    unsigned int m = blockIdx.x;
    if (m >= M) return;
    float* row = scores + (long)m * N;
    __shared__ float sVal[TKBLOCK];
    __shared__ int sIdx[TKBLOCK];

    for (unsigned int j = 0; j < k; j++) {
        float bv = -3.402823466e+38f;
        int bi = -1;
        for (unsigned int i = threadIdx.x; i < N; i += blockDim.x) {
            float v = row[i];
            // score DESC, then index ASC — the topHits order.
            if (bi < 0 || v > bv || (v == bv && (int)i < bi)) {
                if (v != -3.402823466e+38f || bi < 0) { bv = v; bi = (int)i; }
            }
        }
        sVal[threadIdx.x] = bv;
        sIdx[threadIdx.x] = bi;
        __syncthreads();
        for (unsigned int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                float ov = sVal[threadIdx.x + off];
                int oi = sIdx[threadIdx.x + off];
                float cv = sVal[threadIdx.x];
                int ci = sIdx[threadIdx.x];
                if (oi >= 0 && (ci < 0 || ov > cv || (ov == cv && oi < ci))) {
                    sVal[threadIdx.x] = ov;
                    sIdx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            outIdx[m * k + j] = sIdx[0];
            outScore[m * k + j] = sVal[0];
            if (sIdx[0] >= 0) row[sIdx[0]] = -3.402823466e+38f; // consume
        }
        __syncthreads();
    }
}

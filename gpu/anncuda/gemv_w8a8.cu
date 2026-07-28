// gemv_w8a8.cu — the minimal, correctness-only W8A8 kernels behind the CUDA
// ann.Backend (docs/task-native-gpu.md, Phase 1b). Line-for-line the CUDA twin of
// the MSL in gpu/annmetal/backend.go: `gemv_w8a8` scores ONE query against the whole
// index (FlatI8.Query), `gemm_w8a8` scores M queries in one launch (FlatI8.QueryBatch).
//
// Both compute the EXACT int32 dot of the host-quantized int8 query against each
// row, then apply the query/row rescale — the same value linalg.MatmulBTW8A8
// produces on the CPU. Integer accumulation is exact, so GPU and CPU rank
// identically (backend_test.go gates that, break-it-first).
//
// These are deliberately NOT the tuned decode kernels — no __dp4a, no warp-per-row
// reduction, no vectorized loads. That blob is goinfer's and stays there; this is
// the proving kernel that shows the device layer computes the right numbers.
//
// Two shape conventions carried over from the Metal side:
//   - Scalars arrive as 1-element BUFFERS, not by-value kernel params, mirroring
//     MSL's `constant uint&` binds. That is what lets gpu.Queue.Run1D keep Metal's
//     `bufs ...Buffer` signature on both platforms.
//   - Each kernel takes its own thread count as a trailing param and guards on it:
//     a CUDA launch rounds up to whole blocks, where Metal's dispatchThreads
//     launches exactly n (gpu/cuda.go, divergence 2).

// gemv_w8a8: out[j] = dot(qi8, codes[j]) * qscale * scales[j], one thread per row.
extern "C" __global__ void gemv_w8a8(
    const signed char* __restrict__ codes,  // [N*K] int8, row-major
    const signed char* __restrict__ qi8,    // [K]   int8 query (host-quantized)
    const float* __restrict__ scales,       // [N]   per-row scale
    const unsigned int* __restrict__ K,
    const float* __restrict__ qscale,       // query scale (0 => all-zero query)
    float* __restrict__ out,                // [N]   scores
    const unsigned int* __restrict__ N)     // row count (the launch bound)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= *N) return;
    unsigned int k = *K;
    const signed char* row = codes + (unsigned long long)j * k;
    int acc = 0;
    for (unsigned int i = 0; i < k; i++) {
        acc += (int)qi8[i] * (int)row[i];
    }
    out[j] = (float)acc * (*qscale) * scales[j];
}

// gemm_w8a8: out[m*N+j] = dot(qi8[m], codes[j]) * qscale[m] * scales[j],
// one thread per (query, row) pair.
extern "C" __global__ void gemm_w8a8(
    const signed char* __restrict__ codes,  // [N*K] int8
    const signed char* __restrict__ qi8,    // [M*K] int8 queries (host-quantized)
    const float* __restrict__ scales,       // [N]
    const unsigned int* __restrict__ K,
    const unsigned int* __restrict__ N,
    const float* __restrict__ qscale,       // [M] per-query scale
    float* __restrict__ out,                // [M*N] scores, row-major
    const unsigned int* __restrict__ total) // M*N (the launch bound)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= *total) return;
    unsigned int n = *N, k = *K;
    unsigned int m = g / n, j = g % n;
    const signed char* qrow = qi8   + (unsigned long long)m * k;
    const signed char* crow = codes + (unsigned long long)j * k;
    int acc = 0;
    for (unsigned int i = 0; i < k; i++) {
        acc += (int)qrow[i] * (int)crow[i];
    }
    out[g] = (float)acc * qscale[m] * scales[j];
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

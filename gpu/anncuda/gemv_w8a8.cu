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

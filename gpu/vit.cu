// vit.cu — the kernel set for a native, cgo-free ViT forward (docs/task-native-gpu.md,
// Phase 3). These are aikit's own compute, built on the cuda.go device layer, and they
// exist to run vision.ResidentEncoder on NVIDIA the way goinfer's WebGPU backend runs it
// on any GPU — but cgo-free.
//
// PARITY IS THE POINT, SO THE MATH MIRRORS THE CPU TOWER EXACTLY
// --------------------------------------------------------------
// Every kernel here is written against vision/encoder.go's arithmetic, not against
// "standard" formulations, because the gate is cosine ≈ 1.0 against that CPU forward:
//   - layernorm accumulates mean and variance in DOUBLE and applies
//     (x-mean)*rsqrt(var+eps)*w + b, matching layerNormInto. eps is added to the
//     VARIANCE, not to the deviation.
//   - gelu is the TANH approximation with c = sqrt(2/pi), evaluated in double —
//     geluTanh. Not erf-gelu; they differ enough to move a cosine.
//   - softmax subtracts the row max, exponentiates and sums in DOUBLE — softmaxRow.
//   - attention is BIDIRECTIONAL (no causal mask) and scales scores by 1/sqrt(hd)
//     BEFORE the softmax.
// Turing runs fp64 at 1/32 rate, so the double accumulations are deliberately confined
// to the per-row reductions (a vanishing fraction of a tower's FLOPs) where they buy
// parity; every bulk matmul stays in int32/f32.
//
// Conventions from gpu/cuda.go: scalars by value, each kernel bounds-checks its own
// global index (a CUDA launch rounds up to whole blocks), and weight matrices are
// ROW-MAJOR [N,K] — i.e. the B-transposed layout linalg.MatmulBT uses, so a GEMM here
// is C[m,n] = dot(A[m,:], B[n,:]).

#define LNBLOCK 256

// quant_rows: per-row symmetric int8 quantization, byte-identical to linalg's
// quantizeRowInt8 — scale = maxAbs/127, q = round(x/scale) clamped to +-127, and an
// all-zero row yields scale 0. One block per row. roundf is round-half-away-from-zero,
// which is what Go's math.Round does; rintf (half-to-even) would NOT match.
extern "C" __global__ void quant_rows(
    const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ s,
    int rows, int dim)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * dim;
    __shared__ float sm[LNBLOCK];
    float m = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = fabsf(xr[i]);
        if (a > m) m = a;
    }
    sm[threadIdx.x] = m;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off && sm[threadIdx.x + off] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + off];
        __syncthreads();
    }
    float maxAbs = sm[0];
    float scale = maxAbs / 127.f;
    if (threadIdx.x == 0) s[r] = (maxAbs == 0.f) ? 0.f : scale;
    signed char* qr = q + (long)r * dim;
    if (maxAbs == 0.f) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) qr[i] = 0;
        return;
    }
    float inv = 1.f / scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = roundf(xr[i] * inv);
        if (v > 127.f) v = 127.f;
        else if (v < -127.f) v = -127.f;
        qr[i] = (signed char)v;
    }
}

// gemm_w8a8: C[M,N] = (A_q[M,K] . B_q[N,K]) * aScale[M] * bScale[N]. The int8 dot is
// EXACT integer arithmetic, so this contributes no error of its own — the only rounding
// is the final rescale. One thread per output element; correctness-first, not tiled.
extern "C" __global__ void gemm_w8a8(
    const signed char* __restrict__ A, const float* __restrict__ aScale,
    const signed char* __restrict__ B, const float* __restrict__ bScale,
    float* __restrict__ C, int M, int N, int K)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)M * N) return;
    int m = (int)(g / N), n = (int)(g % N);
    const signed char* ar = A + (long)m * K;
    const signed char* br = B + (long)n * K;
    int acc = 0;
    for (int k = 0; k < K; k++) acc += (int)ar[k] * (int)br[k];
    C[g] = (float)acc * aScale[m] * bScale[n];
}

// gemm_f32: C[M,N] = A[M,K] . B[N,K] in f32 (B row-major = B-transposed), for the
// patch-embed projection, whose weights are f32 rather than quantized.
extern "C" __global__ void gemm_f32(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)M * N) return;
    int m = (int)(g / N), n = (int)(g % N);
    const float* ar = A + (long)m * K;
    const float* br = B + (long)n * K;
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += ar[k] * br[k];
    C[g] = acc;
}

// add_bias: x[r,d] += bias[d] — a per-row broadcast (projection biases).
extern "C" __global__ void add_bias(
    float* __restrict__ x, const float* __restrict__ bias, int rows, int dim)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)rows * dim) return;
    x[g] += bias[g % dim];
}

// add_vec: x[i] += v[i] — elementwise, for the positional embedding and the two
// residual adds per layer.
extern "C" __global__ void add_vec(
    float* __restrict__ x, const float* __restrict__ v, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    x[g] += v[g];
}

// layernorm: standard mean/variance LayerNorm (NOT RMS), one block per row, double
// accumulation to match layerNormInto. Two passes over the row: mean, then variance.
extern "C" __global__ void layernorm(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out, int rows, int dim, float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * dim;
    float* dst = out + (long)r * dim;
    __shared__ double sm[LNBLOCK];

    double acc = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += (double)xr[i];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    double mean = sm[0] / (double)dim;
    __syncthreads();

    acc = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        double d = (double)xr[i] - mean;
        acc += d * d;
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    double var = sm[0] / (double)dim;
    double inv = 1.0 / sqrt(var + (double)eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = (float)(((double)xr[i] - mean) * inv) * w[i] + b[i];
    }
}

// gelu_tanh: the tanh approximation, in double, matching geluTanh exactly.
extern "C" __global__ void gelu_tanh(float* __restrict__ x, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    const double c = 0.7978845608028654; // sqrt(2/pi)
    double v = (double)x[g];
    x[g] = (float)(0.5 * v * (1.0 + tanh(c * (v + 0.044715 * v * v * v))));
}

// attention: bidirectional multi-head self-attention over np patches, ONE BLOCK per
// (head, query). The block stages that query's np scores in dynamic shared memory,
// softmaxes them (max-subtract, double-accumulated sum — softmaxRow), then each thread
// accumulates one output dimension. No causal mask: every query attends to every patch.
//
// q/k/v are [np, nH*hd] with head h at column offset h*hd, so no gather/transpose pass
// is needed — the kernel indexes the heads in place.
//
// Shared memory required: np * sizeof(float), sized by the caller.
extern "C" __global__ void attention(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, int np, int nH, int hd, float scale)
{
    int blk = blockIdx.x;
    if (blk >= nH * np) return;
    int h = blk / np, i = blk % np;
    int hidden = nH * hd, off = h * hd;
    extern __shared__ float sc[];

    const float* qi = q + (long)i * hidden + off;
    // scores[j] = dot(q_i, k_j) * scale
    for (int j = threadIdx.x; j < np; j += blockDim.x) {
        const float* kj = k + (long)j * hidden + off;
        float acc = 0.f;
        for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
        sc[j] = acc * scale;
    }
    __syncthreads();

    // row max, then exp/sum in double — both reduced through shared memory.
    __shared__ float smax[LNBLOCK];
    __shared__ double ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int j = threadIdx.x; j < np; j += blockDim.x) if (sc[j] > m) m = sc[j];
    smax[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && smax[threadIdx.x + o] > smax[threadIdx.x]) smax[threadIdx.x] = smax[threadIdx.x + o];
        __syncthreads();
    }
    float mx = smax[0];
    __syncthreads();

    double su = 0.0;
    for (int j = threadIdx.x; j < np; j += blockDim.x) {
        double e = exp((double)sc[j] - (double)mx);
        sc[j] = (float)e;
        su += e;
    }
    ssum[threadIdx.x] = su;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) ssum[threadIdx.x] += ssum[threadIdx.x + o];
        __syncthreads();
    }
    double inv = 1.0 / ssum[0];
    __syncthreads();
    for (int j = threadIdx.x; j < np; j += blockDim.x) sc[j] = (float)((double)sc[j] * inv);
    __syncthreads();

    // out[i, off+d] = sum_j softmax[j] * v[j, off+d]
    float* oi = out + (long)i * hidden + off;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < np; j++) acc += sc[j] * v[(long)j * hidden + off + d];
        oi[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Qwen2.5-VL ViT additions (Phase 3). The SigLIP set above covers the shared ops;
// these are the five Qwen needs on top, and they are written against
// vision/qwen_encoder.go's arithmetic for the same parity reason.
//
// Note gelu_erf below is a DIFFERENT function from gelu_tanh above. Qwen's patch
// merger uses nn.GELU()'s exact erf form while its SigLIP counterpart uses the tanh
// approximation; they differ by ~5e-4, which is far more than the parity bar. Both
// are shipped, and callers must pick deliberately.
// ---------------------------------------------------------------------------

// rmsnorm: weight-only RMS normalization (no mean subtraction, no bias) —
// x * rsqrt(mean(x^2) + eps) * w, double-accumulated to match rmsNorm. One block
// per row.
extern "C" __global__ void rmsnorm(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, int rows, int dim, float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * dim;
    float* dst = out + (long)r * dim;
    __shared__ double sm[LNBLOCK];
    double acc = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        double v = (double)xr[i];
        acc += v * v;
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sm[threadIdx.x] += sm[threadIdx.x + o];
        __syncthreads();
    }
    double inv = 1.0 / sqrt(sm[0] / (double)dim + (double)eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = (float)((double)xr[i] * inv) * w[i];
    }
}

// rope_qk: NeoX rotate_half 2D rotary applied IN PLACE to the q and k thirds of a
// fused qkv buffer [seq, 3*hidden] — row layout [3, nH, hd], so q is at offset 0 and
// k at offset hidden. v is untouched. Splitting qkv into three buffers first would be
// a pure copy; indexing the thirds here avoids it.
//
// Each thread owns one (patch, head, d) pair with d < hd/2 and rotates BOTH q and k,
// reading x and y before writing either — the pairs are disjoint, so this is
// race-free and bit-identical to the CPU's in-place pairwise form.
extern "C" __global__ void rope_qk(
    float* __restrict__ qkv, const float* __restrict__ cos, const float* __restrict__ sin,
    int seq, int nH, int hd)
{
    int half = hd / 2;
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)seq * nH * half;
    if (g >= total) return;
    int d = (int)(g % half);
    long rest = g / half;
    int head = (int)(rest % nH);
    int i = (int)(rest / nH);
    int hidden = nH * hd;

    const float* co = cos + (long)i * hd;
    const float* si = sin + (long)i * hd;
    long qoff = (long)i * 3 * hidden + (long)head * hd;
    long koff = qoff + hidden;

    float x = qkv[qoff + d], y = qkv[qoff + d + half];
    qkv[qoff + d] = x * co[d] - y * si[d];
    qkv[qoff + d + half] = y * co[d + half] + x * si[d + half];

    x = qkv[koff + d]; y = qkv[koff + d + half];
    qkv[koff + d] = x * co[d] - y * si[d];
    qkv[koff + d + half] = y * co[d + half] + x * si[d + half];
}

// attention_seg: bidirectional MHA restricted to each patch's cu_seqlens SEGMENT —
// a window for most blocks, a whole image for the fullatt blocks. The caller passes
// per-patch segment bounds (segStart/segEnd) rather than the cu_seqlens prefix array,
// so the kernel needs no search.
//
// Reads q/k/v from the FUSED qkv buffer [seq, 3*hidden] at offsets 0/hidden/2*hidden.
// One block per (head, query); the segment's scores stage in dynamic shared memory,
// so the caller sizes it to maxSegment*4 bytes.
extern "C" __global__ void attention_seg(
    const float* __restrict__ qkv, float* __restrict__ out,
    const int* __restrict__ segStart, const int* __restrict__ segEnd,
    int seq, int nH, int hd, float scale)
{
    int blk = blockIdx.x;
    if (blk >= nH * seq) return;
    int h = blk / seq, i = blk % seq;
    int hidden = nH * hd, off = h * hd;
    int s0 = segStart[i], s1 = segEnd[i], n = s1 - s0;
    extern __shared__ float sc[];

    const float* qi = qkv + (long)i * 3 * hidden + off;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const float* kj = qkv + (long)(s0 + t) * 3 * hidden + hidden + off;
        float acc = 0.f;
        for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
        sc[t] = acc * scale;
    }
    __syncthreads();

    __shared__ float smax[LNBLOCK];
    __shared__ double ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int t = threadIdx.x; t < n; t += blockDim.x) if (sc[t] > m) m = sc[t];
    smax[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && smax[threadIdx.x + o] > smax[threadIdx.x]) smax[threadIdx.x] = smax[threadIdx.x + o];
        __syncthreads();
    }
    float mx = smax[0];
    __syncthreads();

    double su = 0.0;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        double e = exp((double)sc[t] - (double)mx);
        sc[t] = (float)e;
        su += e;
    }
    ssum[threadIdx.x] = su;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) ssum[threadIdx.x] += ssum[threadIdx.x + o];
        __syncthreads();
    }
    double inv = 1.0 / ssum[0];
    __syncthreads();
    for (int t = threadIdx.x; t < n; t += blockDim.x) sc[t] = (float)((double)sc[t] * inv);
    __syncthreads();

    float* oi = out + (long)i * hidden + off;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < n; t++) acc += sc[t] * qkv[(long)(s0 + t) * 3 * hidden + 2 * hidden + off + d];
        oi[d] = acc;
    }
}

// silu_mul: gate = silu(gate) * up, the gated-MLP activation. silu in double to match
// the CPU's float64 exp.
extern "C" __global__ void silu_mul(
    float* __restrict__ gate, const float* __restrict__ up, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    double v = (double)gate[g];
    gate[g] = (float)(v / (1.0 + exp(-v))) * up[g];
}

// gelu_erf: the EXACT (erf) GELU — nn.GELU()'s default, used by Qwen's patch merger.
// Distinct from gelu_tanh above; see the header note.
extern "C" __global__ void gelu_erf(float* __restrict__ x, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    double v = (double)x[g];
    x[g] = (float)(0.5 * v * (1.0 + erf(v / 1.4142135623730951)));
}

// ---------------------------------------------------------------------------
// TILED GEMMs (throughput). The gemm_w8a8 / gemm_f32 above are correctness-first:
// one thread per output element, each re-reading a whole row of A and of B from
// global memory. At ViT shapes that is the dominant cost — every A row is re-read N
// times and every B row M times.
//
// These tile through shared memory instead: a TILE×TILE block cooperatively stages
// one A-tile and one B-tile per K-chunk, so each element is read from global memory
// once per tile rather than once per output. Same arithmetic, same operand order
// within a thread's accumulation.
//
// The W8A8 variant is expected to be BIT-IDENTICAL to the untiled one, and its test
// asserts exactly that: the accumulator is int32 and integer addition is associative,
// so re-chunking the K loop cannot change the result. That is a much sharper gate than
// a tolerance, and it is available here precisely because the dot is integer.
// gemm_f32_tiled cannot make that claim — f32 addition reassociates — so its gate is a
// tight relative bound instead.
//
// Launch with a 2-D grid: grid = (ceil(N/TILE), ceil(M/TILE)), block = (TILE, TILE).
// ---------------------------------------------------------------------------

#define TILE 16

extern "C" __global__ void gemm_w8a8_tiled(
    const signed char* __restrict__ A, const float* __restrict__ aScale,
    const signed char* __restrict__ B, const float* __restrict__ bScale,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ signed char As[TILE][TILE];
    __shared__ signed char Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.y * TILE + ty;
    int n = blockIdx.x * TILE + tx;
    int bRow = blockIdx.x * TILE + ty; // the B row this thread STAGES (not the one it uses)
    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(long)m * K + k] : (signed char)0;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(long)bRow * K + k] : (signed char)0;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) acc += (int)As[ty][kk] * (int)Bs[tx][kk];
        __syncthreads();
    }
    if (m < M && n < N) C[(long)m * N + n] = (float)acc * aScale[m] * bScale[n];
}

extern "C" __global__ void gemm_f32_tiled(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.y * TILE + ty;
    int n = blockIdx.x * TILE + tx;
    int bRow = blockIdx.x * TILE + ty;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(long)m * K + k] : 0.f;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(long)bRow * K + k] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    if (m < M && n < N) C[(long)m * N + n] = acc;
}

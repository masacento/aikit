// gemv_quant.cu — the TUNED generic quantized GEMV: aikit's GPU twin of linalg's
// W4A8/W8A8 matmul (docs/task-native-gpu.md, the Phase-1b blob-split).
//
// PROVENANCE AND THE BIT-IDENTITY RULE
// -----------------------------------
// These two kernels are lifted VERBATIM from goinfer/cuda/gemv_fwd.cu, which held them
// alongside two LLM-specific kernels (kv_store, rope_kv) purely by co-location — there
// were no shared __device__ helpers and no common state, so the split is a clean cut
// along the entry-point boundary. The LLM pair stays in goinfer; the generic pair lives
// here, where every consumer can reach it.
//
// The bodies are byte-identical to the originals ON PURPOSE. goinfer's CUDA decode is
// parity-gated against HF, and that gate is the tripwire for this extraction: same
// source + same NVRTC + same --gpu-architecture must yield the same instructions, so the
// lift changes nothing numerically. Do NOT "clean up" a kernel here — a reformat that
// changes instruction selection is a silent numeric change wearing a refactor's clothes.
// Tune deliberately, re-run goinfer's parity suite, and say so.
//
// Both kernels follow this layer's conventions (gpu/cuda.go): scalars are passed BY VALUE
// (gpu.ArgValue), each kernel bounds-checks its own row index because a CUDA launch rounds
// up to whole blocks, and an absent optional buffer is bound with gpu.ArgNull so the
// `bias ? ... : 0.f` guard sees a real null pointer.
#include <cuda_fp16.h>

// gemv_w4a8_fwd: dst[n] = (sum_k dequant(W[n,k]) * a[k]) * aScale + bias[n], int4 weights
// with f16 per-group scales. `accum` selects dst[n] += val over dst[n] = val, so an
// out-projection can accumulate straight into a residual stream.
//
// Args: W [N,Kwords] u32 (8 int4 per word, nibble-permuted at pack time), a [2*Kwords] int
// (4 int8 per word), gs [N,Kgroups] f16 group scales, aScalePtr f32* (on-device activation
// scale), bias f32* or null, N, Kwords, Kgroups, dst [N] f32, accum.
extern "C" __global__ void gemv_w4a8_fwd(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    const float* __restrict__ aScalePtr, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, float* __restrict__ dst, int accum)
{
    // COALESCED + 2x ILP unroll: consecutive lanes read consecutive words (128B warp
    // transactions), two independent word loads in flight per lane to saturate the
    // byte-light int4 stream (43%->80% peak in isolation). even/odd + __vsub4 unpack on the
    // fast nibble-permuted layout (permuteFast at pack time); each word's int partial is
    // scaled by its group's f32 scale and float-accumulated (the group sum falls out of the
    // final warp reduce), so no per-word segmented reduction. 32-stride remainder tail.
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi0 = base + lane, wi1 = base + 32 + lane;
        unsigned int w0 = wr[wi0], w1 = wr[wi1];
        int p0 = 0, p1 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0 + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1 + 1], p1);
        facc += (float)p0 * __half2float(sr[wi0 >> 2]);
        facc += (float)p1 * __half2float(sr[wi1 >> 2]);
    }
    // Tail: Kwords need NOT be a multiple of 32 (e.g. qwen2.5-0.5B H=896 → Kwords=112).
    // Guard per lane — the scale-per-word float accumulate has no cross-lane dependency, so
    // out-of-range lanes simply contribute nothing. Without this they read past the row.
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) {
            unsigned int word = wr[wi];
            int p = 0;
            p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p);
            p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p);
            facc += (float)p * __half2float(sr[wi >> 2]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) {
        float val = facc * (*aScalePtr) + (bias ? bias[n] : 0.f);
        dst[n] = accum ? dst[n] + val : val;
    }
}

// W8A8 forward GEMV: per-row f32 weight scale, on-device activation scale ptr, per-row bias.
// W is int8x4-packed (4 int8 per word, row-major), a is the same int8 activation (4/word).
extern "C" __global__ void gemv_w8a8_fwd(
    const int* __restrict__ W, const int* __restrict__ a, const float* __restrict__ wScale,
    const float* __restrict__ aScalePtr, const float* __restrict__ bias,
    int N, int Kdiv4, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int* wr = W + (long)n * Kdiv4;
    int acc = 0;
    for (int k = lane; k < Kdiv4; k += 32) acc = __dp4a(wr[k], a[k], acc);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) {
        float val = (float)acc * wScale[n] * (*aScalePtr) + (bias ? bias[n] : 0.f);
        dst[n] = accum ? dst[n] + val : val;
    }
}

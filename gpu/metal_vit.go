//go:build darwin

package gpu

import "fmt"

// metal_vit.go is the Metal mirror of cuda_vit.go — the transformer-encoder kernel
// set a ViT forward needs on top of the metal.go device layer (docs/task-native-gpu.md,
// Phase 3). The exported surface (ViT, NewViT, the kernel-name consts, ViTBlock) is
// identical to the CUDA side: they are build-tag mutually exclusive, so a consumer that
// only touches this surface stays platform-agnostic — the whole point of the substrate.
//
// Every kernel mirrors vision/encoder.go's arithmetic exactly, EXCEPT for the one thing
// that does not port: MSL has no `double`. The CUDA kernels accumulate LayerNorm
// mean/variance, softmax sums, and GELU in double to match the CPU tower's float64 math;
// here those reductions run in f32 with a pairwise (tree) reduction over threadgroup
// memory, which is the best f32 can do. That costs a little precision — see
// metal_vit_test.go for the achieved per-kernel tolerances and why they differ from the
// CUDA bars. The tanh-GELU formula, the max-subtract softmax, and the bidirectional
// 1/sqrt(hd) attention are otherwise identical.
//
// Metal ↔ CUDA divergences baked into these kernels (inverses of gpu/cuda.go's):
//   - dispatchThreads launches EXACTLY n threads, so no per-kernel bounds check.
//   - per-row / per-(head,query) kernels use one threadgroup per unit: the CUDA
//     blockIdx.x is threadgroup_position_in_grid, threadIdx.x is
//     thread_position_in_threadgroup, blockDim.x is threads_per_threadgroup.
//   - scalars bind as 1-element buffers (there is no by-value arg on Metal).

// vitMSL is the encoder kernel set, compiled at load by NewViT (Metal compiles MSL at
// runtime, so unlike the CUDA side there is no embedded PTX). LNBLOCK matches ViTBlock.
const vitMSL = `
#include <metal_stdlib>
using namespace metal;
#define LNBLOCK 256

// quant_rows: per-row symmetric int8 quant, byte-identical to linalg.quantizeRowInt8.
// One threadgroup per row. round() is half-away-from-zero, matching Go's math.Round.
kernel void quant_rows(
    device const float* x [[buffer(0)]], device char* q [[buffer(1)]], device float* s [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& dim [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)r * (uint)dim;
    threadgroup float sm[LNBLOCK];
    float m = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float a = abs(xr[i]); if (a > m) m = a; }
    sm[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) {
        if (tid < o && sm[tid + o] > sm[tid]) sm[tid] = sm[tid + o];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxAbs = sm[0];
    float scale = maxAbs / 127.0f;
    if (tid == 0) s[r] = (maxAbs == 0.0f) ? 0.0f : scale;
    device char* qr = q + (uint)r * (uint)dim;
    if (maxAbs == 0.0f) { for (int i = tid; i < dim; i += tgsz) qr[i] = 0; return; }
    float inv = 1.0f / scale;
    for (int i = tid; i < dim; i += tgsz) {
        float v = round(xr[i] * inv);
        if (v > 127.0f) v = 127.0f; else if (v < -127.0f) v = -127.0f;
        qr[i] = (char)v;
    }
}

// gemm_w8a8: C[M,N] = (Aq[M,K].Bq[N,K]) * aScale[M] * bScale[N]. Exact int32 dot.
kernel void gemm_w8a8(
    device const char* A [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const char* B [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device float* C [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& N [[buffer(6)]], constant int& K [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int g = (int)gid, m = g / N, n = g % N;
    device const char* ar = A + (uint)m * (uint)K;
    device const char* br = B + (uint)n * (uint)K;
    int acc = 0;
    for (int k = 0; k < K; k++) acc += (int)ar[k] * (int)br[k];
    C[g] = (float)acc * aScale[m] * bScale[n];
}

// gemm_f32: C[M,N] = A[M,K].B[N,K] in f32 (B row-major = B-transposed).
kernel void gemm_f32(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int g = (int)gid, m = g / N, n = g % N;
    device const float* ar = A + (uint)m * (uint)K;
    device const float* br = B + (uint)n * (uint)K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += ar[k] * br[k];
    C[g] = acc;
}

// add_bias: x[r,d] += bias[d].
kernel void add_bias(
    device float* x [[buffer(0)]], device const float* bias [[buffer(1)]],
    constant int& rows [[buffer(2)]], constant int& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    x[gid] += bias[(int)gid % dim];
}

// add_vec: x[i] += v[i].
kernel void add_vec(
    device float* x [[buffer(0)]], device const float* v [[buffer(1)]], constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    x[gid] += v[gid];
}

// layernorm: mean/variance LayerNorm (not RMS), one threadgroup per row, f32 pairwise
// reduction (the double the CUDA kernel uses does not exist in MSL). eps is added to the
// variance. rsqrt(var+eps) == 1/sqrt(var+eps).
kernel void layernorm(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& rows [[buffer(4)]], constant int& dim [[buffer(5)]], constant float& eps [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)r * (uint)dim;
    device float* dst = out + (uint)r * (uint)dim;
    threadgroup float sm[LNBLOCK];

    float acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) acc += xr[i];
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean = sm[0] / (float)dim;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float d = xr[i] - mean; acc += d * d; }
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float var = sm[0] / (float)dim;
    float inv = rsqrt(var + eps);

    for (int i = tid; i < dim; i += tgsz) dst[i] = ((xr[i] - mean) * inv) * w[i] + b[i];
}

// gelu_tanh: the tanh approximation, in f32 (precise::tanh), matching geluTanh's formula.
kernel void gelu_tanh(device float* x [[buffer(0)]], constant int& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    float v = x[gid];
    x[gid] = 0.5f * v * (1.0f + precise::tanh(c * (v + 0.044715f * v * v * v)));
}

// attention: bidirectional multi-head self-attention, ONE threadgroup per (head, query).
// The per-query score row is staged in dynamic threadgroup memory (sc, np floats, bound
// at index 0 by the caller); max and sum reduce through static threadgroup arrays.
// Softmax sum is f32 (the CUDA double does not port).
kernel void attention(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& np [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    threadgroup float* sc [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    int h = (int)blk / np, i = (int)blk % np;
    int hidden = nH * hd, off = h * hd;
    device const float* qi = q + (uint)i * (uint)hidden + off;
    for (int j = tid; j < np; j += tgsz) {
        device const float* kj = k + (uint)j * (uint)hidden + off;
        float acc = 0.0f;
        for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
        sc[j] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float smax[LNBLOCK];
    threadgroup float ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int j = tid; j < np; j += tgsz) if (sc[j] > m) m = sc[j];
    smax[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o && smax[tid + o] > smax[tid]) smax[tid] = smax[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx = smax[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float su = 0.0f;
    for (int j = tid; j < np; j += tgsz) { float e = exp(sc[j] - mx); sc[j] = e; su += e; }
    ssum[tid] = su;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) ssum[tid] += ssum[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f / ssum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = tid; j < np; j += tgsz) sc[j] = sc[j] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* oi = out + (uint)i * (uint)hidden + off;
    for (int d = tid; d < hd; d += tgsz) {
        float acc = 0.0f;
        for (int j = 0; j < np; j++) acc += sc[j] * v[(uint)j * (uint)hidden + off + d];
        oi[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Qwen2.5-VL ViT additions (Phase 3) — the Metal mirror of vit.cu's Qwen section,
// the five kernels Qwen needs on top of the shared SigLIP set. Written against
// vision/qwen_encoder.go's arithmetic, same as the CUDA side.
//
// gelu_erf below is a DIFFERENT function from gelu_tanh above: Qwen's patch merger
// uses nn.GELU()'s exact erf form while SigLIP uses the tanh approximation. They
// differ by ~5e-4, far above any parity bar — shipping one for both is a silent
// numeric bug, so both are present and callers pick deliberately.
// ---------------------------------------------------------------------------

// rmsnorm: weight-only RMS normalization (no mean subtraction, no bias) —
// x * rsqrt(mean(x^2) + eps) * w, one threadgroup per row. The mean-square reduces
// in f32 (the double vit.cu uses does not port); the divide/rsqrt stay exact because
// the library is compiled fast-math OFF (CompileLibraryPrecise) — the same guard that
// kept the quant scale from drifting a ULP on the SigLIP set.
kernel void rmsnorm(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)r * (uint)dim;
    device float* dst = out + (uint)r * (uint)dim;
    threadgroup float sm[LNBLOCK];
    float acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float vv = xr[i]; acc += vv * vv; }
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = rsqrt(sm[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += tgsz) dst[i] = (xr[i] * inv) * w[i];
}

// rope_qk: NeoX rotate_half 2D rotary applied IN PLACE to the q and k thirds of a fused
// qkv buffer [seq, 3*hidden] (row layout [3, nH, hd]); v is untouched. One thread per
// (patch, head, d<hd/2), rotating both q and k. dispatchThreads launches exactly the
// seq*nH*(hd/2) threads, so no bounds check.
kernel void rope_qk(
    device float* qkv [[buffer(0)]], device const float* cos [[buffer(1)]], device const float* sin [[buffer(2)]],
    constant int& seq [[buffer(3)]], constant int& nH [[buffer(4)]], constant int& hd [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int hf = hd / 2; // named hf, not half: half is MSL's f16 type keyword.
    int g = (int)gid;
    int d = g % hf;
    int rest = g / hf;
    int head = rest % nH;
    int i = rest / nH;
    int hidden = nH * hd;
    device const float* co = cos + (uint)i * (uint)hd;
    device const float* si = sin + (uint)i * (uint)hd;
    uint qoff = (uint)i * 3u * (uint)hidden + (uint)head * (uint)hd;
    uint koff = qoff + (uint)hidden;
    float x = qkv[qoff + d], y = qkv[qoff + d + hf];
    qkv[qoff + d] = x * co[d] - y * si[d];
    qkv[qoff + d + hf] = y * co[d + hf] + x * si[d + hf];
    x = qkv[koff + d]; y = qkv[koff + d + hf];
    qkv[koff + d] = x * co[d] - y * si[d];
    qkv[koff + d + hf] = y * co[d + hf] + x * si[d + hf];
}

// attention_seg: bidirectional MHA restricted to each patch's segment (a window for
// most blocks, a whole image for the fullatt blocks). Reads q/k/v from the FUSED qkv
// buffer [seq, 3*hidden] at offsets 0/hidden/2*hidden; the per-query score row stages
// in dynamic threadgroup memory (sc, maxSeg floats, bound at index 0 by the caller).
// segStart/segEnd are PER-PATCH bounds, not cu_seqlens. One threadgroup per (head,
// query). Softmax sum is f32 (the vit.cu double does not port).
kernel void attention_seg(
    device const float* qkv [[buffer(0)]], device float* out [[buffer(1)]],
    device const int* segStart [[buffer(2)]], device const int* segEnd [[buffer(3)]],
    constant int& seq [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    threadgroup float* sc [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    int h = (int)blk / seq, i = (int)blk % seq;
    int hidden = nH * hd, off = h * hd;
    int s0 = segStart[i], s1 = segEnd[i], n = s1 - s0;
    device const float* qi = qkv + (uint)i * 3u * (uint)hidden + off;
    for (int t = tid; t < n; t += tgsz) {
        device const float* kj = qkv + (uint)(s0 + t) * 3u * (uint)hidden + (uint)hidden + off;
        float acc = 0.0f;
        for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
        sc[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float smax[LNBLOCK];
    threadgroup float ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int t = tid; t < n; t += tgsz) if (sc[t] > m) m = sc[t];
    smax[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o && smax[tid + o] > smax[tid]) smax[tid] = smax[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx = smax[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float su = 0.0f;
    for (int t = tid; t < n; t += tgsz) { float e = exp(sc[t] - mx); sc[t] = e; su += e; }
    ssum[tid] = su;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) ssum[tid] += ssum[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f / ssum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = tid; t < n; t += tgsz) sc[t] = sc[t] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* oi = out + (uint)i * (uint)hidden + off;
    for (int d = tid; d < hd; d += tgsz) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++) acc += sc[t] * qkv[(uint)(s0 + t) * 3u * (uint)hidden + 2u * (uint)hidden + off + d];
        oi[d] = acc;
    }
}

// silu_mul: gate = silu(gate) * up, the gated-MLP activation. silu in f32 (vit.cu's
// double does not port), which is why the per-kernel bar is looser than CUDA's 1e-6.
kernel void silu_mul(
    device float* gate [[buffer(0)]], device const float* up [[buffer(1)]], constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float vv = gate[gid];
    gate[gid] = (vv / (1.0f + exp(-vv))) * up[gid];
}

// erf_approx: Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7 — at the f32 floor.
// MSL's standard library has NO erf (unlike CUDA), so the EXACT GELU below supplies its
// own. This is what makes gelu_erf a genuine f32 approximation of the float64 reference,
// hence its looser (but stated) per-kernel bar; the discrimination from gelu_tanh
// (~5e-4) is unaffected and remains the proof of correct FORM.
inline float erf_approx(float x) {
    float s = sign(x);
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

// gelu_erf: the EXACT (erf) GELU — nn.GELU()'s default, used by Qwen's patch merger.
// Distinct from gelu_tanh; see the header note. The divide matches vit.cu (exact under
// fast-math-off); erf is erf_approx above because MSL lacks a stdlib erf.
kernel void gelu_erf(device float* x [[buffer(0)]], constant int& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float vv = x[gid];
    x[gid] = 0.5f * vv * (1.0f + erf_approx(vv / 1.4142135623730951f));
}`

// Kernel entry points in vitMSL (identical names to the CUDA side).
const (
	KernelQuantRows = "quant_rows"
	KernelGEMMW8A8  = "gemm_w8a8"
	KernelGEMMF32   = "gemm_f32"
	KernelAddBias   = "add_bias"
	KernelAddVec    = "add_vec"
	KernelLayerNorm = "layernorm"
	KernelGELUTanh  = "gelu_tanh"
	KernelAttention = "attention"

	// --- Qwen2.5-VL additions (identical names to the CUDA side) ---
	KernelRMSNorm      = "rmsnorm"
	KernelRopeQK       = "rope_qk"
	KernelAttentionSeg = "attention_seg"
	KernelSiLUMul      = "silu_mul"
	KernelGELUErf      = "gelu_erf"
)

// ViTBlock is the threadgroup width the per-row/attention kernels reduce at; it must
// match vitMSL's LNBLOCK (the static threadgroup reduction arrays are sized to it).
const ViTBlock = 256

// ViT holds the compiled encoder kernel pipelines — identical shape to the CUDA ViT.
type ViT struct {
	QuantRows Pipeline
	GEMMW8A8  Pipeline
	GEMMF32   Pipeline
	AddBias   Pipeline
	AddVec    Pipeline
	LayerNorm Pipeline
	GELUTanh  Pipeline
	Attention Pipeline

	// Qwen2.5-VL additions.
	RMSNorm      Pipeline
	RopeQK       Pipeline
	AttentionSeg Pipeline
	SiLUMul      Pipeline
	GELUErf      Pipeline
}

// NewViT compiles vitMSL on this device and builds every encoder pipeline. The library
// is tracked by the Device, so ReleaseObjects frees it.
func (d *Device) NewViT() (ViT, error) {
	lib, err := d.CompileLibraryPrecise(vitMSL, MSL3_1)
	if err != nil {
		return ViT{}, fmt.Errorf("metal: compile ViT library: %w", err)
	}
	var v ViT
	for _, bind := range []struct {
		name string
		dst  *Pipeline
	}{
		{KernelQuantRows, &v.QuantRows},
		{KernelGEMMW8A8, &v.GEMMW8A8},
		{KernelGEMMF32, &v.GEMMF32},
		{KernelAddBias, &v.AddBias},
		{KernelAddVec, &v.AddVec},
		{KernelLayerNorm, &v.LayerNorm},
		{KernelGELUTanh, &v.GELUTanh},
		{KernelAttention, &v.Attention},
		{KernelRMSNorm, &v.RMSNorm},
		{KernelRopeQK, &v.RopeQK},
		{KernelAttentionSeg, &v.AttentionSeg},
		{KernelSiLUMul, &v.SiLUMul},
		{KernelGELUErf, &v.GELUErf},
	} {
		p, err := d.NewComputePipeline(lib, bind.name)
		if err != nil {
			return ViT{}, err
		}
		*bind.dst = p
	}
	return v, nil
}

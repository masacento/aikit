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
	} {
		p, err := d.NewComputePipeline(lib, bind.name)
		if err != nil {
			return ViT{}, err
		}
		*bind.dst = p
	}
	return v, nil
}

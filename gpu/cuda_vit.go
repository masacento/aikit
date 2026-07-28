//go:build linux

package gpu

import (
	_ "embed"
	"fmt"
)

// cuda_vit.go publishes the transformer-encoder kernel set (docs/task-native-gpu.md,
// Phase 3) — the compute a ViT forward needs on top of the cuda.go device layer.
//
// These live in gpu/ rather than beside a vision backend because none of them is
// vision-specific: a quantized GEMM, a LayerNorm, a GELU, a bidirectional attention
// and a per-row int8 quantizer are the same ops a text encoder needs (Phase 4). The
// vision-specific part — how a SigLIP tower is wired out of these — lives in
// gpu/visioncuda, which is also where the aikit/vision import lives, so this package
// stays aikit-import-free like the rest of the device layer.
//
// Every kernel mirrors vision/encoder.go's arithmetic exactly (double-accumulated
// reductions, tanh-GELU, max-subtract softmax); see vit.cu for why that matters and
// where it is deliberately not the "standard" formulation.

// ViTPTX is the compiled vit.cu. Exported so a consumer can load it into a module it
// already owns. Regenerate with ./build_ptx.sh — never hand-edit.
//
//go:embed testdata/vit.ptx
var ViTPTX []byte

// Kernel entry points in ViTPTX.
const (
	// KernelQuantRows quantizes [rows,dim] f32 to int8 with a per-row scale, exactly
	// as linalg's quantizeRowInt8 does. Launch ONE BLOCK PER ROW.
	//   (x f32*, q int8*, s f32*, rows i32, dim i32)
	KernelQuantRows = "quant_rows"

	// KernelGEMMW8A8 is C[M,N] = (Aq[M,K]·Bq[N,K]) * aScale[M] * bScale[N]; B is
	// row-major [N,K], the B-transposed layout linalg.MatmulBT uses.
	//   (A int8*, aScale f32*, B int8*, bScale f32*, C f32*, M, N, K i32)
	KernelGEMMW8A8 = "gemm_w8a8"

	// KernelGEMMF32 is C[M,N] = A[M,K]·B[N,K] in f32, for unquantized weights.
	//   (A f32*, B f32*, C f32*, M, N, K i32)
	KernelGEMMF32 = "gemm_f32"

	// KernelAddBias does x[r,d] += bias[d] over [rows,dim].
	//   (x f32*, bias f32*, rows i32, dim i32)
	KernelAddBias = "add_bias"

	// KernelAddVec does x[i] += v[i] — positional embeddings and residual adds.
	//   (x f32*, v f32*, n i32)
	KernelAddVec = "add_vec"

	// KernelLayerNorm is mean/variance LayerNorm (not RMS) with weight and bias,
	// double-accumulated. Launch ONE BLOCK PER ROW.
	//   (x f32*, w f32*, b f32*, out f32*, rows i32, dim i32, eps f32)
	KernelLayerNorm = "layernorm"

	// KernelGELUTanh applies the tanh-approximation GELU in place.
	//   (x f32*, n i32)
	KernelGELUTanh = "gelu_tanh"

	// KernelAttention is bidirectional multi-head self-attention over np patches.
	// Launch ONE BLOCK PER (head, query) — i.e. nH*np blocks — with np*4 bytes of
	// dynamic shared memory for the score row.
	//   (q f32*, k f32*, v f32*, out f32*, np, nH, hd i32, scale f32)
	KernelAttention = "attention"

	// --- Qwen2.5-VL additions ---

	// KernelRMSNorm is weight-only RMS normalization (no mean subtraction, no bias).
	// Launch ONE BLOCK PER ROW.
	//   (x f32*, w f32*, out f32*, rows i32, dim i32, eps f32)
	KernelRMSNorm = "rmsnorm"

	// KernelRopeQK applies NeoX rotate_half 2D rotary IN PLACE to the q and k thirds
	// of a FUSED qkv buffer [seq, 3*hidden] (row layout [3, nH, hd]); v is untouched.
	// One thread per (patch, head, d<hd/2).
	//   (qkv f32*, cos f32*, sin f32*, seq, nH, hd i32)
	KernelRopeQK = "rope_qk"

	// KernelAttentionSeg is bidirectional MHA restricted to each patch's segment —
	// a window, or a whole image for the fullatt blocks. Reads q/k/v from the fused
	// qkv buffer. Launch ONE BLOCK PER (head, query) with maxSegment*4 bytes of
	// dynamic shared memory. segStart/segEnd are PER-PATCH bounds, not cu_seqlens.
	//   (qkv f32*, out f32*, segStart i32*, segEnd i32*, seq, nH, hd i32, scale f32)
	KernelAttentionSeg = "attention_seg"

	// KernelSiLUMul computes gate = silu(gate) * up — the gated-MLP activation.
	//   (gate f32*, up f32*, n i32)
	KernelSiLUMul = "silu_mul"

	// KernelGELUErf is the EXACT (erf) GELU that nn.GELU() defaults to, used by
	// Qwen's patch merger. NOT interchangeable with KernelGELUTanh — they differ by
	// ~5e-4, far above any parity bar. Pick deliberately.
	//   (x f32*, n i32)
	KernelGELUErf = "gelu_erf"

	// --- tiled GEMMs (throughput) ---

	// KernelGEMMW8A8Tiled is gemm_w8a8 staged through shared memory. Same signature,
	// and BIT-IDENTICAL output: the accumulator is int32 and integer addition is
	// associative, so re-chunking the K loop cannot change the result.
	// Launch with TileGrid(M, N).
	KernelGEMMW8A8Tiled = "gemm_w8a8_tiled"

	// KernelGEMMF32Tiled is gemm_f32 staged through shared memory. Same signature.
	// NOT bit-identical to the untiled kernel — f32 addition reassociates — so it is
	// gated on a tight relative bound instead. Launch with TileGrid(M, N).
	KernelGEMMF32Tiled = "gemm_f32_tiled"
)

// GEMMTile is the tiled GEMMs' tile width; it must match vit.cu's TILE, which sizes
// their shared-memory staging arrays statically.
const GEMMTile = 16

// ViTBlock is the block width the per-row kernels reduce at. It must match vit.cu's
// LNBLOCK, since those kernels size their shared-memory reduction arrays statically.
const ViTBlock = 256

// ViT holds the compiled encoder kernel pipelines.
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

	// Tiled GEMMs — same math, staged through shared memory.
	GEMMW8A8Tiled Pipeline
	GEMMF32Tiled  Pipeline
}

// NewViT loads ViTPTX on this device and builds every encoder pipeline. The module is
// tracked by the Device, so ReleaseObjects frees it.
func (d *Device) NewViT() (ViT, error) {
	lib, err := d.CompileLibrary(ViTPTX)
	if err != nil {
		return ViT{}, fmt.Errorf("cuda: load ViT module: %w", err)
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
		{KernelGEMMW8A8Tiled, &v.GEMMW8A8Tiled},
		{KernelGEMMF32Tiled, &v.GEMMF32Tiled},
	} {
		p, err := d.NewComputePipeline(lib, bind.name)
		if err != nil {
			return ViT{}, err
		}
		*bind.dst = p
	}
	return v, nil
}

// RowGrid is the geometry the per-row kernels (quant_rows, layernorm) require: one
// block per row, ViTBlock threads cooperating on that row's reduction.
func RowGrid(rows int) LaunchConfig {
	if rows <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32(rows), GridY: 1, GridZ: 1,
		BlockX: ViTBlock, BlockY: 1, BlockZ: 1,
	}
}

// SegAttentionGrid is attention_seg's geometry: one block per (head, query), with the
// segment's score row in dynamic shared memory sized to the LARGEST segment (a window,
// or a whole image on the fullatt blocks).
func SegAttentionGrid(seq, heads, maxSeg int) LaunchConfig {
	if seq <= 0 || heads <= 0 || maxSeg <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32(seq * heads), GridY: 1, GridZ: 1,
		BlockX: ViTBlock, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32(maxSeg * 4),
	}
}

// AttentionGrid is attention's geometry: one block per (head, query), with the score
// row staged in dynamic shared memory. np*4 bytes must fit the device's per-block
// shared limit (48 KB by default ⇒ np ≤ 12288), which every SigLIP/Qwen grid does.
func AttentionGrid(np, heads int) LaunchConfig {
	if np <= 0 || heads <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32(np * heads), GridY: 1, GridZ: 1,
		BlockX: ViTBlock, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32(np * 4),
	}
}

// TileGrid is the tiled GEMMs' 2-D geometry: one GEMMTile×GEMMTile output tile per
// block. Both kernels bounds-check, so a grid that overhangs M or N is safe.
func TileGrid(M, N int) LaunchConfig {
	if M <= 0 || N <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX:  uint32((N + GEMMTile - 1) / GEMMTile),
		GridY:  uint32((M + GEMMTile - 1) / GEMMTile),
		GridZ:  1,
		BlockX: GEMMTile, BlockY: GEMMTile, BlockZ: 1,
	}
}

//go:build amd64

package linalg

// AVX-512 VNNI tier for the W4A8 (int4 weight × int8 activation) fused dot,
// ahead of dotW4A8FoldAVX2 in the dotW4A8 dispatch cascade
// (quant_w4a8_amd64.go). Mirrors dot_i8_avx512vnni_amd64.go's VPDPBUSD
// approach, adapted to the nibble-packed W4A8 layout the way
// dotW4A8FoldAVX2 already adapts dotI8AVX2's sign-extend body.

// dotW4A8FoldAVX512VNNI returns the per-group-scaled f32 dot
// Σ_g scale[g]·(act·w)_g of one int4 weight row against the int8 activation
// row — same contract as dotW4A8FoldAVX2 (group fixed at 32; nGroups = K/32;
// ragged K%32 handled by the Go caller), same in-register f32 fold (no
// per-group horizontal reduce), just a different inner computation for the
// per-group 8×int32 lane-partials. Implemented in
// dot_w4a8_avx512vnni_amd64.s.
//
// dotW4A8FoldAVX2's nibble-unpack prologue centers each nibble to a signed
// weight (nibble−8) via VPSUBB before sign-extending and feeding VPMADDWD —
// centering is necessary there because the AVX2 path has no unsigned×signed
// primitive to exploit. VPDPBUSD is exactly that primitive: it wants an
// UNSIGNED operand, and a raw nibble (0..15) already IS one, with no shift
// needed (unlike dotI8AVX512VNNI's a-operand, which starts signed and needs
// the XOR-0x80 trick to become unsigned). So the prologue here drops the two
// VPSUBB's entirely and feeds the raw nibble straight to VPDPBUSD:
//
//	Σ (nib[k]-8)·act[k] = Σ nib[k]·act[k] - 8·Σ act[k]
//
// The first term is one VPDPBUSD(nib_u8, act_s8) per group; the second is a
// correction sum computed the same way dotI8AVX512VNNI computes Σb — a
// second VPDPBUSD, this time against an all-ones unsigned vector. Both are
// folded in-register (converted to f32, multiplied by the group's broadcast
// scale, accumulated into an 8-lane f32 accumulator) exactly like
// dotW4A8FoldAVX2's single fold, just two parallel accumulators instead of
// one; the correction accumulator's per-iteration multiplier is
// pre-negated by 8 so combining the two at the end is addition, not
// subtraction — sidesteps any doubt about operand order on a non-commutative
// op, since the two folds are then combined with a single commutative add.
//
// Unlike dotI8AVX512VNNI, this kernel's result is NOT bit-exact against the
// scalar reference — it inherits the same f32-fold rounding
// dotW4A8FoldAVX2 already has (that kernel's own doc comment: the per-group
// scale multiply-accumulate happens in f32, so reassociation across groups
// changes the last bit or two). Tested to the repo's existing W4A8 tolerance
// (relative error ≤ 1e-5, TestW4A8_dotMatchesScalar's bar), not equality.
//
// Uses the 256-bit (YMM) VPDPBUSD form — one 32-byte quant group per
// instruction, a direct 1:1 loop-iteration match with dotW4A8FoldAVX2 and
// with the ARM64 dotW4A8FoldSDOT family's per-group structure — rather than
// packing two groups into one 512-bit (ZMM) instruction and splitting the
// 16-lane result at the group boundary. That ZMM-packed form is a real lever
// (twice the bytes per VPDPBUSD, half the loop overhead) left for later if
// profiling calls for it; this is the correctness-first cut, scoped the same
// way dot_w4a8_arm64.s's baseline dotW4A8FoldSDOT preceded its own later
// SplitHalf/2Acc/4Acc variants.
//
//go:noescape
func dotW4A8FoldAVX512VNNI(act *int8, packed *byte, scales *float32, nGroups int) float32

// hasAVX512VNNIVL is hasAVX512VNNI (dot_i8_avx512vnni_amd64.go) plus
// AVX512VL (CPUID leaf 7, EBX bit 31). dotI8AVX512VNNI only ever touches
// full ZMM registers, which needs just AVX512F+BW+VNNI; this file's kernel
// uses the 256-bit (YMM) VPDPBUSD encoding to match one 32-byte quant group
// per instruction, and that encoding is only legal with VL present. Every
// shipping VNNI CPU (Cascade Lake onward on Intel, Zen 4 onward on AMD)
// bundles VL in as part of the same AVX-512 baseline, so this is not
// expected to ever differ from hasAVX512VNNI in practice — but dispatch
// shouldn't rely on an unchecked assumption for an instruction that SIGILLs
// if it's wrong, so it's checked explicitly rather than reusing
// hasAVX512VNNI directly. Computed once at init, same pattern as
// hasAVX512VNNI/hasAVX2.
var hasAVX512VNNIVL = hasAVX512VNNI && detectAVX512VL()

func detectAVX512VL() bool {
	_, ebx7, _, _ := cpuid(7, 0)
	const bitAVX512VL = 1 << 31
	return ebx7&bitAVX512VL != 0
}

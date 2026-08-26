// AVX-512 VNNI W4A8 fold kernel for amd64. See dot_w4a8_avx512vnni_amd64.go
// for the uncentered-nibble derivation (raw 0..15 nibble is already a valid
// VPDPBUSD unsigned operand, so no VPSUBB centering pass is needed here the
// way dotW4A8FoldAVX2 needs one) and for why the sum-fold's per-iteration
// multiplier is pre-negated by 8 (makes the final combine an add, not a
// subtract).
//
// One 32-byte quant group per loop iteration (256-bit VPDPBUSD, AVX512VL —
// see hasAVX512VNNIVL): unpack 16 packed bytes to 32 unsigned nibble bytes
// (same VPAND/VPSRLW/VPUNPCKLBW/VPUNPCKHBW prologue as dotW4A8FoldAVX2, minus
// its two VPSUBB's), VPDPBUSD against the group's 32 activations for the raw
// nib·act partials, VPDPBUSD against an all-ones vector for the Σact
// correction partials — both 8×int32, unreduced. Each is converted to f32,
// multiplied by the group's broadcast scale (negated ×8 for the correction
// side), and folded into its own persistent 8-lane f32 accumulator; only
// after the loop are the two accumulators each reduced to a scalar and
// added.
//
// group fixed at 32 (16 packed bytes / 32 activations per iteration); the Go
// caller (dotW4A8 in quant_w4a8_amd64.go) routes nGroups=K/32 and mops up any
// K%32 remainder in scalar. Only called when hasAVX512VNNIVL.

#include "textflag.h"

DATA mask0FVNNI<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA mask0FVNNI<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL mask0FVNNI<>(SB), RODATA|NOPTR, $16

DATA onesU8VNNI<>+0(SB)/8, $0x0101010101010101
DATA onesU8VNNI<>+8(SB)/8, $0x0101010101010101
DATA onesU8VNNI<>+16(SB)/8, $0x0101010101010101
DATA onesU8VNNI<>+24(SB)/8, $0x0101010101010101
GLOBL onesU8VNNI<>(SB), RODATA|NOPTR, $32

DATA negEightVNNI<>+0(SB)/4, $0xC1000000 // float32(-8.0)
GLOBL negEightVNNI<>(SB), RODATA|NOPTR, $4

// func dotW4A8FoldAVX512VNNI(act *int8, packed *byte, scales *float32, nGroups int) float32
TEXT ·dotW4A8FoldAVX512VNNI(SB), NOSPLIT, $0-36
	MOVQ act+0(FP), SI     // &act[0]    (int8, 32 per group)
	MOVQ packed+8(FP), DI  // &packed[0] (16 bytes per group)
	MOVQ scales+16(FP), BX // &scales[0] (f32, one per group)
	MOVQ nGroups+24(FP), CX

	LEAQ    mask0FVNNI<>(SB), AX
	VMOVDQU (AX), X14      // low-nibble mask, hoisted
	LEAQ    onesU8VNNI<>(SB), AX
	VMOVDQU (AX), Y15      // unsigned-1s vector (32B), hoisted
	LEAQ         negEightVNNI<>(SB), AX
	VBROADCASTSS (AX), Y12 // -8.0 broadcast to all 8 lanes, hoisted

	VXORPS Y8, Y8, Y8 // foldDot: Σ_g scale[g]·Σ_lane(nib·act)_g        (8×f32)
	VXORPS Y9, Y9, Y9 // foldSum: Σ_g -8·scale[g]·Σ_lane(1·act)_g       (8×f32)

	TESTQ CX, CX
	JE    reduce

loop:
	// Unpack 16 packed bytes → k0..k15 (X3) and k16..k31 (X4), UNCENTERED
	// (raw nibble 0..15 — no VPSUBB, unlike dotW4A8FoldAVX2).
	VMOVDQU     (DI), X0
	VPAND       X14, X0, X1     // low nibbles  (k0,k2,…,k30)
	VPSRLW      $4, X0, X2
	VPAND       X14, X2, X2     // high nibbles (k1,k3,…,k31)
	VPUNPCKLBW  X2, X1, X3      // [k0,k1,…,k15]  (becomes Y3's low 128 bits)
	VPUNPCKHBW  X2, X1, X4      // [k16,…,k31]
	VINSERTI128 $1, X4, Y3, Y3  // Y3 = 32 unsigned nibble bytes, k-order

	VMOVDQU (SI), Y6 // 32 int8 activations

	VPXOR    Y5, Y5, Y5  // accDot = 0 (8×int32, this group only)
	VPXOR    Y7, Y7, Y7  // accSum = 0
	VPDPBUSD Y6, Y3, Y5  // accDot = Σ_4 nib·act  per lane (unreduced)
	VPDPBUSD Y6, Y15, Y7 // accSum = Σ_4 1·act    per lane (unreduced)

	VCVTDQ2PS Y5, Y5 // → f32
	VCVTDQ2PS Y7, Y7

	VBROADCASTSS (BX), Y10      // scale[g]
	VFMADD231PS  Y10, Y5, Y8    // foldDot += accDot_f32 · scale[g]
	VMULPS       Y12, Y10, Y11  // Y11 = scale[g] · (-8)
	VFMADD231PS  Y11, Y7, Y9    // foldSum += accSum_f32 · (-8·scale[g])

	ADDQ $32, SI
	ADDQ $16, DI
	ADDQ $4, BX
	DECQ CX
	JNE  loop

reduce:
	VEXTRACTF128 $1, Y8, X0
	VADDPS       X0, X8, X8
	VHADDPS      X8, X8, X8
	VHADDPS      X8, X8, X8
	// X8[0] = Σ_g scale[g]·Σnib·act  (mainFold)

	VEXTRACTF128 $1, Y9, X1
	VADDPS       X1, X9, X9
	VHADDPS      X9, X9, X9
	VHADDPS      X9, X9, X9
	// X9[0] = Σ_g -8·scale[g]·Σact   (already carries the -8 factor)

	VADDSS X9, X8, X8 // commutative — order-safe regardless of operand convention
	MOVSS  X8, ret+32(FP)
	VZEROUPPER
	RET

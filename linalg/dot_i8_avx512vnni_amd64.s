// AVX-512 VNNI int8 dot product for amd64. See dot_i8_avx512vnni_amd64.go for
// the sign-correction derivation (VPDPBUSD is u8×s8; a is shifted unsigned via
// XOR 0x80 and the +128·Σb correction is itself a second VPDPBUSD, against an
// all-ones vector, rather than a separate widen/sum pass).
//
// 128 int8 (two ZMM) per main-loop iteration, TWO independent (dot, sum)
// accumulator pairs so the four VPDPBUSD chains issue independently — the
// same reasoning dotI8AVX2's four accumulators use, scaled to VNNI's native
// 64-byte width. A 64-byte single-chunk step (not a real loop — the 128-wide
// loop above it guarantees the remainder here is < 64 already, i.e. it can
// fire at most once) covers the odd 64-block, then the caller handles
// whatever is left below 64 (dotI8 in quant_i8_amd64.go routes that to
// dotI8AVX2 or the scalar tail).
//
// Reduction: combine the two chunks' accumulators, fold the high 256 bits of
// each ZMM into its low 256 (VEXTRACTI64X4), then finish with the same
// YMM->XMM->scalar reduce dotI8AVX2 already uses (VEXTRACTI128 + VPADDD +
// VPHADDD×2). Two reductions (dot and sum), then one IMULL/SUBL combines them.

#include "textflag.h"

DATA bias80<>+0(SB)/8, $0x8080808080808080
DATA bias80<>+8(SB)/8, $0x8080808080808080
DATA bias80<>+16(SB)/8, $0x8080808080808080
DATA bias80<>+24(SB)/8, $0x8080808080808080
DATA bias80<>+32(SB)/8, $0x8080808080808080
DATA bias80<>+40(SB)/8, $0x8080808080808080
DATA bias80<>+48(SB)/8, $0x8080808080808080
DATA bias80<>+56(SB)/8, $0x8080808080808080
GLOBL bias80<>(SB), RODATA|NOPTR, $64

DATA onesU8<>+0(SB)/8, $0x0101010101010101
DATA onesU8<>+8(SB)/8, $0x0101010101010101
DATA onesU8<>+16(SB)/8, $0x0101010101010101
DATA onesU8<>+24(SB)/8, $0x0101010101010101
DATA onesU8<>+32(SB)/8, $0x0101010101010101
DATA onesU8<>+40(SB)/8, $0x0101010101010101
DATA onesU8<>+48(SB)/8, $0x0101010101010101
DATA onesU8<>+56(SB)/8, $0x0101010101010101
GLOBL onesU8<>(SB), RODATA|NOPTR, $64

// func dotI8AVX512VNNI(a, b *int8, n int) int32
TEXT ·dotI8AVX512VNNI(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX

	LEAQ     bias80<>(SB), AX
	VMOVDQU8 (AX), Z31 // sign-flip mask, held for the whole function
	LEAQ     onesU8<>(SB), AX
	VMOVDQU8 (AX), Z30 // unsigned-1s vector, held for the whole function

	VPXORD Z0, Z0, Z0 // accDot, chunk 0 (16×int32 lanes)
	VPXORD Z1, Z1, Z1 // accSum, chunk 0 (Σ b, for the correction)
	VPXORD Z2, Z2, Z2 // accDot, chunk 1
	VPXORD Z3, Z3, Z3 // accSum, chunk 1

loop128:
	CMPQ CX, $128
	JL   loop64
	VMOVDQU8 0(SI), Z4
	VMOVDQU8 0(DI), Z5
	VMOVDQU8 64(SI), Z6
	VMOVDQU8 64(DI), Z7
	VPXORD   Z31, Z4, Z8 // au0 = a0 ^ 0x80 (unsigned view, +128)
	VPXORD   Z31, Z6, Z9 // au1 = a1 ^ 0x80
	VPDPBUSD Z5, Z8, Z0  // accDot0 += au0 · b0
	VPDPBUSD Z5, Z30, Z1 // accSum0 += 1  · b0
	VPDPBUSD Z7, Z9, Z2  // accDot1 += au1 · b1
	VPDPBUSD Z7, Z30, Z3 // accSum1 += 1  · b1
	ADDQ     $128, SI
	ADDQ     $128, DI
	SUBQ     $128, CX
	JMP      loop128

loop64:
	// Fires at most once: loop128 only falls through here once CX<128, and
	// this step consumes exactly 64, leaving CX<64 — never enough to repeat.
	CMPQ CX, $64
	JL   reduce
	VMOVDQU8 0(SI), Z4
	VMOVDQU8 0(DI), Z5
	VPXORD   Z31, Z4, Z8
	VPDPBUSD Z5, Z8, Z0
	VPDPBUSD Z5, Z30, Z1
	ADDQ     $64, SI
	ADDQ     $64, DI
	SUBQ     $64, CX

reduce:
	VPADDD Z2, Z0, Z0 // combine chunk0+chunk1 dot accumulators
	VPADDD Z3, Z1, Z1 // combine chunk0+chunk1 sum accumulators

	// Reduce Z0 (16×int32 across the full ZMM) to one scalar in AX.
	VEXTRACTI64X4 $1, Z0, Y4
	VPADDD        Y4, Y0, Y0
	VEXTRACTI128  $1, Y0, X4
	VPADDD        X4, X0, X0
	VPHADDD       X0, X0, X0
	VPHADDD       X0, X0, X0
	MOVL          X0, AX

	// Same reduction for Z1 (the Σ b correction term) into DX.
	VEXTRACTI64X4 $1, Z1, Y5
	VPADDD        Y5, Y1, Y1
	VEXTRACTI128  $1, Y1, X5
	VPADDD        X5, X1, X1
	VPHADDD       X1, X1, X1
	VPHADDD       X1, X1, X1
	MOVL          X1, DX

	IMULL $128, DX  // DX = 128 · Σ b
	SUBL  DX, AX    // AX = (Σ (a+128)·b) − 128·Σ b = Σ a·b
	MOVL  AX, ret+24(FP)
	VZEROUPPER
	RET

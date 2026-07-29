// AVX2 int8 -> float32 weight widen for amd64.
//
//   dequantI8AVX2 — dst[i] = float32(q[i]) * scale, for n elements (n a
//                   multiple of 8; the Go wrapper handles any tail).
//
// Per iteration: VPMOVSXBD sign-extends 8 int8 (an m64 load) to 8 int32 in a
// YMM, VCVTDQ2PS converts them to float32, VMULPS applies the broadcast scale,
// VMOVUPS stores 8 floats. Four such chains are interleaved (32 elements per
// iteration) so the convert latency overlaps. VZEROUPPER on every exit avoids
// the AVX/SSE transition penalty.
//
// The result is bit-identical to the scalar Go loop: float32(int8) is exact and
// the scale is one f32 multiply in both.

#include "textflag.h"

// func dequantI8AVX2(q *int8, dst *float32, n int, scale float32)
TEXT ·dequantI8AVX2(SB), NOSPLIT, $0-28
	MOVQ         q+0(FP), SI
	MOVQ         dst+8(FP), DI
	MOVQ         n+16(FP), CX
	VBROADCASTSS scale+24(FP), Y15

	// 32 elements per iteration while at least 32 remain.
	CMPQ CX, $32
	JL   tail8

loop32:
	VPMOVSXBD (SI), Y0
	VPMOVSXBD 8(SI), Y1
	VPMOVSXBD 16(SI), Y2
	VPMOVSXBD 24(SI), Y3

	VCVTDQ2PS Y0, Y0
	VCVTDQ2PS Y1, Y1
	VCVTDQ2PS Y2, Y2
	VCVTDQ2PS Y3, Y3

	VMULPS Y15, Y0, Y0
	VMULPS Y15, Y1, Y1
	VMULPS Y15, Y2, Y2
	VMULPS Y15, Y3, Y3

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(DI)
	VMOVUPS Y2, 64(DI)
	VMOVUPS Y3, 96(DI)

	ADDQ $32, SI
	ADDQ $128, DI
	SUBQ $32, CX
	CMPQ CX, $32
	JGE  loop32

tail8:
	// 8 at a time for what is left (n is a multiple of 8 by contract).
	CMPQ CX, $8
	JL   done

loop8:
	VPMOVSXBD (SI), Y0
	VCVTDQ2PS Y0, Y0
	VMULPS    Y15, Y0, Y0
	VMOVUPS   Y0, (DI)

	ADDQ $8, SI
	ADDQ $32, DI
	SUBQ $8, CX
	CMPQ CX, $8
	JGE  loop8

done:
	VZEROUPPER
	RET

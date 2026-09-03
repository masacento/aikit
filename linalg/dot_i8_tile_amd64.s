// AVX2 activation-blocked W8A8 tile — docs/task-simd-audit.md S-01b's amd64 half.
// FOUR activation rows against ONE weight row, four int32 accumulators.
//
// THE SHAPE IS 4x1 BECAUSE 4x2 WAS MEASURED AND REJECTED. A 4-activation x
// 2-weight version fits AVX2's 16 YMM registers and costs fewer instructions per
// MAC (0.172 against this shape's 0.203), and on cache-resident B it duly measured
// 1.34-1.52x. On streamed B it measured 0.70x and 0.57x -- a 1.75x REGRESSION -- 
// because two weight rows K bytes apart are two interleaved DRAM streams where the
// span it replaces walks one. That is the same failure that got an eight-column
// W8A8 kernel deleted in v1.17.0 after it was shipped on a single cache-resident
// measurement; see w8a8Span's comment. This shape blocks ONLY the activation
// dimension, so B is still read strictly linearly, one stream, exactly as before.
//
// What it saves: dotI8AVX2 sign-extends BOTH operands for every (activation row,
// weight row) pair -- 4 instructions per 16 MACs. Here the weight chunk is widened
// once and four activation rows consume it: 1 + 4x(VPMOVSXBW, VPMADDWD, VPADDD) =
// 13 instructions per 64 MACs, 0.203 per MAC against 0.25.
//
// The ceiling is low and known in advance: VPMADDWD stays at one per 16 MACs and
// issues on a single Zen 2 port, so ~67 GMAC/s at 4.2 GHz binds no matter how the
// loop is blocked, and dotI8AVX2 already sits at 72-76% of it. arm64's 3.5-3.9x
// does not transfer: SDOT does 16 MACs on four pipes, VPMADDWD does 16 on one.
//
// BIT-IDENTICAL FOR FREE: every partial is int32, integer addition is exact and
// associative. n must be a multiple of 16; the caller mops up the remainder.

#include "textflag.h"

// func dotI8Tile4x1AVX2(act *int8, actStride int, w *int8, dst *int32, n int)
TEXT ·dotI8Tile4x1AVX2(SB), NOSPLIT, $0-40
	MOVQ act+0(FP), SI
	MOVQ actStride+8(FP), R12
	MOVQ w+16(FP), DI
	MOVQ dst+24(FP), DX
	MOVQ n+32(FP), CX

	LEAQ (SI)(R12*1), R8
	LEAQ (R8)(R12*1), R9
	LEAQ (R9)(R12*1), R10

	VPXOR Y0, Y0, Y0 // acc[act 0]
	VPXOR Y1, Y1, Y1 // acc[act 1]
	VPXOR Y2, Y2, Y2 // acc[act 2]
	VPXOR Y3, Y3, Y3 // acc[act 3]

i8tile4x1loop:
	VPMOVSXBW (DI), Y4 // the weight row chunk, widened ONCE for all four rows
	VPMOVSXBW (SI), Y5
	VPMADDWD  Y4, Y5, Y6
	VPADDD    Y6, Y0, Y0 // acc[act 0]
	VPMOVSXBW (R8), Y5
	VPMADDWD  Y4, Y5, Y6
	VPADDD    Y6, Y1, Y1 // acc[act 1]
	VPMOVSXBW (R9), Y5
	VPMADDWD  Y4, Y5, Y6
	VPADDD    Y6, Y2, Y2 // acc[act 2]
	VPMOVSXBW (R10), Y5
	VPMADDWD  Y4, Y5, Y6
	VPADDD    Y6, Y3, Y3 // acc[act 3]

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, DI
	SUBQ $16, CX
	JNZ  i8tile4x1loop

	// Four horizontal int32 reduces, the same tree dotI8AVX2 ends with.
	VEXTRACTI128 $1, Y0, X5
	VPADDD       X5, X0, X0
	VPHADDD      X0, X0, X0
	VPHADDD      X0, X0, X0
	MOVD         X0, AX
	MOVL         AX, 0(DX) // dst[act 0]
	VEXTRACTI128 $1, Y1, X5
	VPADDD       X5, X1, X1
	VPHADDD      X1, X1, X1
	VPHADDD      X1, X1, X1
	MOVD         X1, AX
	MOVL         AX, 4(DX) // dst[act 1]
	VEXTRACTI128 $1, Y2, X5
	VPADDD       X5, X2, X2
	VPHADDD      X2, X2, X2
	VPHADDD      X2, X2, X2
	MOVD         X2, AX
	MOVL         AX, 8(DX) // dst[act 2]
	VEXTRACTI128 $1, Y3, X5
	VPADDD       X5, X3, X3
	VPHADDD      X3, X3, X3
	VPHADDD      X3, X3, X3
	MOVD         X3, AX
	MOVL         AX, 12(DX) // dst[act 3]
	VZEROUPPER
	RET

// AVX2 register-blocked W4A8 tile — docs/task-simd-audit.md S-01's amd64 half.
// Counterpart of dot_w4a8_tile_arm64.s, but a DIFFERENT SHAPE for a hardware
// reason: AVX2 has 16 YMM registers where NEON has 32, so the arm64 4-activation
// x 4-weight tile (16 live f32 accumulators) does not fit. This blocks the
// activation dimension only — ONE weight row against FOUR activation rows, four
// f32 accumulators — which is exactly the reuse that matters, since the nibble
// unpack belongs to the weight row and is what was being repeated per activation
// row.
//
// Per 32-k group, dotW4A8FoldAVX2 spends 25 instructions for 32 MACs, of which
// 10 are the unpack prologue (load, 2x VPAND, VPSRLW, 2x VPUNPCK, 2x VPSUBB, 2x
// VPMOVSXBW) and 1 is the scale broadcast. At M>1 the canonical span pays all 11
// again for every activation row over the identical weight bytes. Here they are
// paid once and four activation rows consume them:
//
//     (10 unpack + 1 broadcast + 4 x 9 per-row + 5 loop) / 4 rows
//       = 13 instructions per 32 MACs per row, against 25.
//
// The VPMADDWD count per MAC is unchanged at 1 per 16 MACs, so this does not move
// the Zen 2 port floor (~67 GMAC/s at 4.2 GHz); it moves the kernel toward it
// from 17.1.
//
// BIT-IDENTICAL BY CONSTRUCTION. Each output's per-group sequence is
// dotW4A8FoldAVX2's, instruction for instruction -- VPMADDWD low, VPMADDWD high,
// VPADDD, VCVTDQ2PS, VFMADD231PS by the broadcast group scale into that output's
// own 8-lane f32 accumulator, in ascending group -- and the epilogue is the same
// VEXTRACTF128 / VADDPS / VHADDPS / VHADDPS reduce. Only the unpack and the
// broadcast are shared; no partial sum is regrouped.
//
// Register budget is all 16 YMM and that is what fixes the tile at four rows:
//   Y14/Y15 the 0x0F mask and the 8 bias, hoisted
//   Y3/Y4   the unpacked, sign-extended weight halves, live across the four rows
//   Y11     the group's broadcast scale
//   Y8/Y9/Y10/Y13  the four f32 accumulators, live for the whole call
//   Y0/Y1/Y2/Y5/Y6/Y7/Y12  unpack and per-row scratch

#include "textflag.h"

DATA tmask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA tmask0F<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL tmask0F<>(SB), RODATA|NOPTR, $16

DATA tconst8<>+0(SB)/8, $0x0808080808080808
DATA tconst8<>+8(SB)/8, $0x0808080808080808
GLOBL tconst8<>(SB), RODATA|NOPTR, $16

// func dotW4A8Tile4RowAVX2(act *int8, actStride int, packed *byte, scales *float32, dst *float32, nGroups int)
TEXT ·dotW4A8Tile4RowAVX2(SB), NOSPLIT, $0-48
	MOVQ act+0(FP), SI
	MOVQ actStride+8(FP), R11
	MOVQ packed+16(FP), DI
	MOVQ scales+24(FP), BX
	MOVQ dst+32(FP), DX
	MOVQ nGroups+40(FP), CX

	// The other three activation rows. Each of the four advances by 32 per
	// group independently, so no index arithmetic runs in the loop.
	LEAQ (SI)(R11*1), R8
	LEAQ (R8)(R11*1), R9
	LEAQ (R9)(R11*1), R10

	LEAQ    tmask0F<>(SB), AX
	VMOVDQU (AX), X14
	LEAQ    tconst8<>(SB), AX
	VMOVDQU (AX), X15

	VXORPS Y8, Y8, Y8   // acc row 0
	VXORPS Y9, Y9, Y9   // acc row 1
	VXORPS Y10, Y10, Y10 // acc row 2
	VXORPS Y13, Y13, Y13 // acc row 3

tile4rowloop:
	// ---- the weight row's 16 packed bytes: unpacked ONCE for all four rows ----
	VMOVDQU    (DI), X0
	VPAND      X14, X0, X1
	VPSRLW     $4, X0, X2
	VPAND      X14, X2, X2
	VPUNPCKLBW X2, X1, X3
	VPUNPCKHBW X2, X1, X4
	VPSUBB     X15, X3, X3
	VPSUBB     X15, X4, X4
	VPMOVSXBW  X3, Y3
	VPMOVSXBW  X4, Y4
	VBROADCASTSS (BX), Y11

	// ---- activation row 0 ----
	VMOVDQU     (SI), X5
	VMOVDQU     16(SI), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y8

	// ---- activation row 1 ----
	VMOVDQU     (R8), X5
	VMOVDQU     16(R8), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y9

	// ---- activation row 2 ----
	VMOVDQU     (R9), X5
	VMOVDQU     16(R9), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y10

	// ---- activation row 3 ----
	VMOVDQU     (R10), X5
	VMOVDQU     16(R10), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y13

	ADDQ $16, DI
	ADDQ $4, BX
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $1, CX
	JNZ  tile4rowloop

	// Four horizontal f32 reduces, the same tree dotW4A8FoldAVX2 ends with.
	VEXTRACTF128 $1, Y8, X11
	VADDPS       X11, X8, X8
	VHADDPS      X8, X8, X8
	VHADDPS      X8, X8, X8
	MOVSS        X8, (DX)

	VEXTRACTF128 $1, Y9, X11
	VADDPS       X11, X9, X9
	VHADDPS      X9, X9, X9
	VHADDPS      X9, X9, X9
	MOVSS        X9, 4(DX)

	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, 8(DX)

	VEXTRACTF128 $1, Y13, X11
	VADDPS       X11, X13, X13
	VHADDPS      X13, X13, X13
	VHADDPS      X13, X13, X13
	MOVSS        X13, 12(DX)

	VZEROUPPER
	RET

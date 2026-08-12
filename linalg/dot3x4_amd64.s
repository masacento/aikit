#include "textflag.h"

// func dotFMA3x4(a0, a1, a2, b0, b1, b2, b3 *float32, n int, out *[12]float32)
//
// The 3×4 register microkernel: three shared a-rows against four b-rows, TWELVE live
// YMM accumulators. out is [a0·b0..a0·b3, a1·b0..a1·b3, a2·b0..a2·b3].
//
// WHY TWELVE, WHICH IS THE WHOLE POINT. Zen 2 issues 2 FMAs per cycle at ~5 cycles of
// latency, so a kernel needs at least 5×2 = 10 INDEPENDENT accumulator chains before
// the FMA pipes, rather than the dependency chain, set the pace. Measured with the
// load-free fmaPeak probe on this box:
//
//	 8 chains → 108.5 GFLOPS   (80% — exactly 8/10 of peak, latency-bound)
//	10 chains → 135.6 GFLOPS   (full peak)
//	12 chains → 135.0 GFLOPS
//
// THE LOAD-PORT HYPOTHESIS WAS WRONG, AND DISPROVING IT IS WHY THIS SHAPE IS 3×4. A
// 2×4 kernel was built first, on the theory that dotFMA8 was load-port bound: it cuts
// loads per K-step from 9 to 6 (1.33 FMA/load vs 0.89) at the same 8 accumulators. It
// measured 44.6 GMAC/s against 1×8's 42.8 — 4%, i.e. nothing. Same chain count, a
// third fewer loads, no gain: the loads were never the constraint. The probe numbers
// above say why, and 12 chains is the first shape that clears the 10-chain threshold.
// (That kernel is not kept; at +4% it earned no place, and this note is its residue.)
//
// The load ratio improves as a side effect: 3 a-loads + 4 b-loads per 12 FMAs = 1.71
// FMA/load, against 1×8's 0.89 (its b operands come from memory, so every FMA carries
// its own load). At 2 load ports that allows 3.4 FMA/cycle — comfortably above the 2
// the FMA ports can retire, so loads stop being the constraint too.
//
// Registers, all 16 used: Y0..Y3 row a0, Y4..Y7 row a1, Y8..Y11 row a2, Y12..Y14 the
// three a-vectors, Y15 the rotating b-vector.
//
// BIT-IDENTITY. Each accumulator sees exactly the products dotFMA8 puts in that
// column's accumulator, in the same K order, reduced by the same VEXTRACTF128 +
// VADDPS + two VHADDPS. Row blocking changes which dot products are computed
// together, never the arithmetic within one.
TEXT ·dotFMA3x4(SB), NOSPLIT, $0-72
	MOVQ a0+0(FP), SI
	MOVQ a1+8(FP), BX
	MOVQ a2+16(FP), DX
	MOVQ b0+24(FP), R8
	MOVQ b1+32(FP), R9
	MOVQ b2+40(FP), R10
	MOVQ b3+48(FP), R11
	MOVQ n+56(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

loop8:
	CMPQ CX, $8
	JL   tail4

	VMOVUPS 0(SI), Y12
	VMOVUPS 0(BX), Y13
	VMOVUPS 0(DX), Y14

	VMOVUPS     0(R8), Y15
	VFMADD231PS Y15, Y12, Y0
	VFMADD231PS Y15, Y13, Y4
	VFMADD231PS Y15, Y14, Y8

	VMOVUPS     0(R9), Y15
	VFMADD231PS Y15, Y12, Y1
	VFMADD231PS Y15, Y13, Y5
	VFMADD231PS Y15, Y14, Y9

	VMOVUPS     0(R10), Y15
	VFMADD231PS Y15, Y12, Y2
	VFMADD231PS Y15, Y13, Y6
	VFMADD231PS Y15, Y14, Y10

	VMOVUPS     0(R11), Y15
	VFMADD231PS Y15, Y12, Y3
	VFMADD231PS Y15, Y13, Y7
	VFMADD231PS Y15, Y14, Y11

	ADDQ $32, SI
	ADDQ $32, BX
	ADDQ $32, DX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R11
	SUBQ $8, CX
	JMP  loop8

tail4:
	CMPQ CX, $4
	JL   reduce

	// Trailing 4-group, mirroring dotFMA8_tail4: load into the LOW 128 bits so the
	// accumulators' lanes 4..7 (the loop's partials) survive; an XMM-form FMA would
	// VEX-zero them and drop the entire main-loop result.
	VMOVUPS 0(SI), X12
	VMOVUPS 0(BX), X13
	VMOVUPS 0(DX), X14

	VMOVUPS     0(R8), X15
	VFMADD231PS Y15, Y12, Y0
	VFMADD231PS Y15, Y13, Y4
	VFMADD231PS Y15, Y14, Y8

	VMOVUPS     0(R9), X15
	VFMADD231PS Y15, Y12, Y1
	VFMADD231PS Y15, Y13, Y5
	VFMADD231PS Y15, Y14, Y9

	VMOVUPS     0(R10), X15
	VFMADD231PS Y15, Y12, Y2
	VFMADD231PS Y15, Y13, Y6
	VFMADD231PS Y15, Y14, Y10

	VMOVUPS     0(R11), X15
	VFMADD231PS Y15, Y12, Y3
	VFMADD231PS Y15, Y13, Y7
	VFMADD231PS Y15, Y14, Y11

reduce:
	MOVQ out+64(FP), DI

	VEXTRACTF128 $1, Y0, X15
	VADDPS       X15, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	MOVSS        X0, 0(DI)

	VEXTRACTF128 $1, Y1, X15
	VADDPS       X15, X1, X1
	VHADDPS      X1, X1, X1
	VHADDPS      X1, X1, X1
	MOVSS        X1, 4(DI)

	VEXTRACTF128 $1, Y2, X15
	VADDPS       X15, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	MOVSS        X2, 8(DI)

	VEXTRACTF128 $1, Y3, X15
	VADDPS       X15, X3, X3
	VHADDPS      X3, X3, X3
	VHADDPS      X3, X3, X3
	MOVSS        X3, 12(DI)

	VEXTRACTF128 $1, Y4, X15
	VADDPS       X15, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
	MOVSS        X4, 16(DI)

	VEXTRACTF128 $1, Y5, X15
	VADDPS       X15, X5, X5
	VHADDPS      X5, X5, X5
	VHADDPS      X5, X5, X5
	MOVSS        X5, 20(DI)

	VEXTRACTF128 $1, Y6, X15
	VADDPS       X15, X6, X6
	VHADDPS      X6, X6, X6
	VHADDPS      X6, X6, X6
	MOVSS        X6, 24(DI)

	VEXTRACTF128 $1, Y7, X15
	VADDPS       X15, X7, X7
	VHADDPS      X7, X7, X7
	VHADDPS      X7, X7, X7
	MOVSS        X7, 28(DI)

	VEXTRACTF128 $1, Y8, X15
	VADDPS       X15, X8, X8
	VHADDPS      X8, X8, X8
	VHADDPS      X8, X8, X8
	MOVSS        X8, 32(DI)

	VEXTRACTF128 $1, Y9, X15
	VADDPS       X15, X9, X9
	VHADDPS      X9, X9, X9
	VHADDPS      X9, X9, X9
	MOVSS        X9, 36(DI)

	VEXTRACTF128 $1, Y10, X15
	VADDPS       X15, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, 40(DI)

	VEXTRACTF128 $1, Y11, X15
	VADDPS       X15, X11, X11
	VHADDPS      X11, X11, X11
	VHADDPS      X11, X11, X11
	MOVSS        X11, 44(DI)

	VZEROUPPER
	RET

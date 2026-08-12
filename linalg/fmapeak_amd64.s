#include "textflag.h"

// func fmaPeakAMD64(iters int64)
//
// Measures sustained single-core f32 AVX2+FMA throughput — the amd64 twin of
// fmapeak_arm64.s, and the measured denominator for BenchmarkGEMMPeakFraction.
//
// 14 INDEPENDENT 8-lane accumulators (Y0..Y13) are updated per iteration, with
// Y14/Y15 held at zero as the (read-only) multiplicands so every accumulator stays
// finite: acc += 0*0. No loads, no stores — pure FP-pipe throughput.
//
// WHY 14. Zen 2 issues 2 FMAs per cycle at ~5 cycles of latency, so a dependent
// chain needs at least 2×5 = 10 in-flight accumulators before the pipes, rather
// than the latency, become the limit. 14 clears that with margin and still leaves
// two registers for the multiplicands. Too few accumulators would measure FMA
// LATENCY and report roughly a fifth of the true ceiling — a number that looks
// plausible and is wrong, which is the whole hazard this probe exists to remove.
//
// Each iteration retires 14 FMAs × 8 lanes × 2 flops = 224 flops.
TEXT ·fmaPeakAMD64(SB), NOSPLIT, $0-8
	MOVQ iters+0(FP), AX

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
	VXORPS Y12, Y12, Y12
	VXORPS Y13, Y13, Y13
	VXORPS Y14, Y14, Y14
	VXORPS Y15, Y15, Y15

	TESTQ AX, AX
	JZ    done

loop:
	VFMADD231PS Y14, Y15, Y0
	VFMADD231PS Y14, Y15, Y1
	VFMADD231PS Y14, Y15, Y2
	VFMADD231PS Y14, Y15, Y3
	VFMADD231PS Y14, Y15, Y4
	VFMADD231PS Y14, Y15, Y5
	VFMADD231PS Y14, Y15, Y6
	VFMADD231PS Y14, Y15, Y7
	VFMADD231PS Y14, Y15, Y8
	VFMADD231PS Y14, Y15, Y9
	VFMADD231PS Y14, Y15, Y10
	VFMADD231PS Y14, Y15, Y11
	VFMADD231PS Y14, Y15, Y12
	VFMADD231PS Y14, Y15, Y13
	DECQ AX
	JNZ  loop

done:
	VZEROUPPER
	RET

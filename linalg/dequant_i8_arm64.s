// NEON int8 -> float32 weight widen for arm64 (perf-campaign item 22b).
//
//   dequantI8NEON — dst[i] = float32(q[i]) * scale, for n elements (n a multiple
//                   of 16; the Go wrapper handles any tail).
//
// Per iteration: one 16-byte load, then the signed widen chain int8→int16→int32
// (SXTL/SXTL2), SCVTF int32→float32, FMUL by the broadcast scale, and a 64-byte
// store of 16 floats. SXTL, SCVTF (vector) and FMUL (vector) have no Go arm64
// assembler mnemonic, so they are emitted as raw WORDs (same convention as
// dot_i8dp/dot_w4a8's SDOT/SCVTF/FADDP). All three are base ARMv8-A NEON, so no
// feature detection is needed — unlike SDOT/i8mm.
//
// Bit-identical to the scalar loop: float32(int8) is exact and the scale is one
// f32 multiply in both. TestDequantizeRowsInt8_bitIdentical is the gate.
//
// Encodings (Rn<<5 | Rd unless noted):
//   SXTL  Vd.8H,Vn.8B  = SSHLL,U=0,Q=0,immh=0001 → 0x0F08A400
//   SXTL2 Vd.8H,Vn.16B = SSHLL,U=0,Q=1,immh=0001 → 0x4F08A400
//   SXTL  Vd.4S,Vn.4H  = SSHLL,U=0,Q=0,immh=0010 → 0x0F10A400
//   SXTL2 Vd.4S,Vn.8H  = SSHLL,U=0,Q=1,immh=0010 → 0x4F10A400
//   SCVTF Vd.4S,Vn.4S  = 0x4E21D800
//   FMUL  Vd.4S,Vn.4S,Vm.4S = 0x6E20DC00 | (Rm<<16)   (here Rm=V16 → 0x6E30DC00)

#include "textflag.h"

// func dequantI8NEON(q *int8, dst *float32, n int, scale float32)
TEXT ·dequantI8NEON(SB), NOSPLIT, $0-28
	MOVD  q+0(FP), R0
	MOVD  dst+8(FP), R1
	MOVD  n+16(FP), R2
	MOVWU scale+24(FP), R3
	VDUP  R3, V16.S4          // broadcast the f32 bit pattern of scale to 4 lanes

loop16:
	VLD1.P 16(R0), [V0.B16]   // 16 int8: q[0..15]
	WORD   $0x0F08A401        // SXTL  V1.8H, V0.8B    → int16 q[0..7]
	WORD   $0x4F08A402        // SXTL2 V2.8H, V0.16B   → int16 q[8..15]
	WORD   $0x0F10A423        // SXTL  V3.4S, V1.4H    → int32 q[0..3]
	WORD   $0x4F10A424        // SXTL2 V4.4S, V1.8H    → int32 q[4..7]
	WORD   $0x0F10A445        // SXTL  V5.4S, V2.4H    → int32 q[8..11]
	WORD   $0x4F10A446        // SXTL2 V6.4S, V2.8H    → int32 q[12..15]
	WORD   $0x4E21D863        // SCVTF V3.4S, V3.4S
	WORD   $0x4E21D884        // SCVTF V4.4S, V4.4S
	WORD   $0x4E21D8A5        // SCVTF V5.4S, V5.4S
	WORD   $0x4E21D8C6        // SCVTF V6.4S, V6.4S
	WORD   $0x6E30DC63        // FMUL  V3.4S, V3.4S, V16.4S
	WORD   $0x6E30DC84        // FMUL  V4.4S, V4.4S, V16.4S
	WORD   $0x6E30DCA5        // FMUL  V5.4S, V5.4S, V16.4S
	WORD   $0x6E30DCC6        // FMUL  V6.4S, V6.4S, V16.4S
	VST1.P [V3.S4, V4.S4, V5.S4, V6.S4], 64(R1) // store 16 f32: q[0..15]·scale
	SUBS   $16, R2, R2
	BNE    loop16

	RET

//go:build amd64

package linalg

// AVX-512 VNNI tier for the int8×int8 dot product, ahead of dotI8AVX2 in the
// dotI8 dispatch cascade (quant_i8_amd64.go). Detection is hand-rolled
// CPUID/XGETBV via the same cpuid/xgetbv asm primitives dot_amd64.go's
// detectAVX2 already declares — no x/sys dep, matching the rest of this file
// set.
//
// dotI8AVX2's own doc comment flags the obvious next lever and why it wasn't
// taken there: "VPMADDUBSW would remove the widening entirely, but u8×i8 pair
// sums can exceed int16 and it SATURATES; that route needs range-limited
// codes and belongs with the VNNI work, not here." VPDPBUSD is that work,
// and it sidesteps the hazard rather than working around it: it accumulates
// straight into int32 (no intermediate int16 stage to saturate), at the cost
// of being an asymmetric u8×s8 op, not s8×s8. See dotI8AVX512VNNI below for
// how the sign is recovered exactly.

// dotI8AVX512VNNI returns Σ a[i]*b[i] as int32 over the first n int8 pairs (n
// a multiple of 64 — the caller's contract, mirroring dotI8AVX2's "multiple of
// 16"; quant_i8_amd64.go's dotI8 peels this prefix and routes any remainder to
// the next tier down). Implemented in dot_i8_avx512vnni_amd64.s.
//
// VPDPBUSD computes an UNSIGNED×SIGNED dot (u8×s8 → int32, no saturation), but
// both operands here are signed. The fix is the standard offset trick, done
// entirely in-register: XOR each a-byte with 0x80 (bit-identical to +128 for a
// two's-complement int8, since a+128 ∈ [0,255] already covers the whole
// unsigned range) to get an unsigned view au, then
//
//	Σ au[i]*b[i] = Σ (a[i]+128)*b[i] = Σ a[i]*b[i] + 128·Σ b[i]
//
// so the true dot is the VPDPBUSD(au, b) result minus 128 times Σ b[i]. That
// correction sum is itself one more VPDPBUSD — dot(ones_u8, b) — rather than a
// separate widen-and-add pass, so the whole kernel is two VPDPBUSD per 64
// bytes plus the XOR, no extra reduction shape. Both accumulators are int32
// and integer addition doesn't reassociate, so — like dotI8AVX2 — the result
// is bit-identical to dotI8Scalar regardless of loop/chunk order; only the
// SIMD-vs-scalar boundary (n rounded down to a multiple of 64) can differ
// from a pure scalar sum, and it doesn't, because integer sums never do.
//
//go:noescape
func dotI8AVX512VNNI(a, b *int8, n int) int32

// hasAVX512VNNI is true when the CPU reports AVX512F + AVX512BW + AVX512_VNNI
// (CPUID leaf 7) and the OS has enabled the full ZMM + opmask extended state
// (XCR0 bits 1,2,5,6,7 — SSE, AVX, opmask, ZMM_Hi256, Hi16_ZMM). Computed once
// at init, same pattern as hasAVX2.
var hasAVX512VNNI = detectAVX512VNNI()

func detectAVX512VNNI() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	// OSXSAVE (leaf 1, ECX bit 27) must be set before XGETBV is safe to run —
	// same prerequisite detectAVX2 checks.
	const bitOSXSAVE = 1 << 27
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&bitOSXSAVE == 0 {
		return false
	}
	// Leaf 7, subleaf 0: EBX bit 16 = AVX512F, bit 30 = AVX512BW (byte/word
	// ops — VPDPBUSD's operand granularity); ECX bit 11 = AVX512_VNNI.
	_, ebx7, ecx7, _ := cpuid(7, 0)
	const (
		bitAVX512F    = 1 << 16
		bitAVX512BW   = 1 << 30
		bitAVX512VNNI = 1 << 11
	)
	if ebx7&(bitAVX512F|bitAVX512BW) != (bitAVX512F | bitAVX512BW) {
		return false
	}
	if ecx7&bitAVX512VNNI == 0 {
		return false
	}
	// XCR0 bits 1 (SSE/XMM), 2 (AVX/YMM), 5 (opmask, k0-k7), 6 (ZMM_Hi256,
	// upper half of ZMM0-15), 7 (Hi16_ZMM, all of ZMM16-31) must ALL be set
	// for the OS to preserve full AVX-512 state across a context switch;
	// without that, a ZMM instruction would silently corrupt state.
	xcr0, _ := xgetbv()
	const xcr0AVX512 = 1<<1 | 1<<2 | 1<<5 | 1<<6 | 1<<7
	return xcr0&xcr0AVX512 == xcr0AVX512
}

package embed

import (
	"math"
	"testing"
)

// TestMXFP4Dequant pins the MXFP4 (ggml type 39) block kernel against hand-derived
// values from the OCP MX spec / the gguf reference — asset-free, so it guards the
// kernel in CI without a checkpoint. (goinfer additionally verifies the SAME math
// bit-for-bit against the reference gguf library on a real gpt-oss:20b tensor.)
func TestMXFP4Dequant(t *testing.T) {
	// e8m0_to_fp32_half: x<2 subnormal, else exponent field = x-1.
	for _, c := range []struct {
		x    uint8
		bits uint32
	}{
		{0, 0x00200000},          // 2^-128 (subnormal)
		{1, 0x00400000},          // 2^-127 (subnormal)
		{2, 0x00800000},          // 2^-126 (smallest normal)
		{127, uint32(126) << 23}, // 0.5
		{128, uint32(127) << 23}, // 1.0
		{255, uint32(254) << 23}, // 2^127
	} {
		if got := math.Float32bits(e8m0ToF32Half(c.x)); got != c.bits {
			t.Errorf("e8m0ToF32Half(%d) = %#08x, want %#08x", c.x, got, c.bits)
		}
	}
	if want := [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}; mxfp4KValues != want {
		t.Errorf("mxfp4KValues = %v, want %v", mxfp4KValues, want)
	}

	// One synthetic block: scale byte 128 (d=1.0) ⇒ element = kvalues[nibble]. Byte j
	// packs element j (low nibble) and element j+16 (high nibble).
	blk := make([]byte, 17)
	blk[0] = 128  // d = 1.0
	blk[1] = 0x07 // low=7 (→+12), high=0 (→0)  ⇒ elem0=12, elem16=0
	blk[2] = 0x9F // low=0xF (→-12), high=9 (→-1) ⇒ elem1=-12, elem17=-1
	var out [32]float32
	dequantMXFP4Block(blk, 0, out[:])
	for idx, exp := range map[int]float32{0: 12, 16: 0, 1: -12, 17: -1} {
		if out[idx] != exp {
			t.Errorf("block elem %d = %v, want %v", idx, out[idx], exp)
		}
	}
}

package embed

import "testing"

// TestUint8s_fp8Dtypes pins that the byte accessor admits the fp8 dtypes, and still refuses
// the wide ones.
//
// The distinction it encodes: Uint8s hands back RAW BYTES, which is only safe when there is
// nothing to reinterpret. F8_E4M3/F8_E5M2 are one byte per element — no endianness, and no
// typed accessor aikit could offer, because their scales live in a SEPARATE tensor over 2-D
// blocks (`*.weight_scale_inv`) that only the caller knows how to compose. BF16/F32 are the
// opposite on both counts and must keep failing here.
func TestUint8s_fp8Dtypes(t *testing.T) {
	for _, dt := range []string{"U8", "I8", "BOOL", "F8_E4M3", "F8_E5M2"} {
		tn := Tensor{Name: "w", DType: dt, Shape: []int{2}, raw: []byte{0x38, 0x40}}
		if _, err := tn.Uint8s(); err != nil {
			t.Errorf("%s: Uint8s should be allowed (byte-wide): %v", dt, err)
		}
	}
	for _, dt := range []string{"BF16", "F16", "F32", "I32"} {
		tn := Tensor{Name: "w", DType: dt, Shape: []int{2}, raw: []byte{1, 2, 3, 4}}
		if _, err := tn.Uint8s(); err == nil {
			t.Errorf("%s: Uint8s must refuse a wide dtype — it has a typed accessor and raw bytes would be endianness-dependent", dt)
		}
	}
}

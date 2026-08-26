package embed

import "testing"

// TestTypedLE_dtypeMismatchMessages pins the exact error text of the four typed tensor
// accessors. They shared an identical hand-written body until typedLE extracted it
// (docs/internal/task-code-health.md §4.1); before that extraction nothing asserted the
// message, so a refactor could have silently reworded a library-facing error. These
// strings are what a consumer sees when a checkpoint carries an unexpected dtype — the
// most common real failure on the model-loading path — so they are worth pinning
// independently of the refactor that prompted them.
func TestTypedLE_dtypeMismatchMessages(t *testing.T) {
	// A dtype no accessor accepts, so every case takes the mismatch branch.
	const got = "BF16"
	cases := []struct {
		name string
		call func(Tensor) error
		want string
	}{
		{"Float32s", func(tn Tensor) error { _, err := tn.Float32s(); return err }, `tensor "w": expected F32, got BF16`},
		{"Float64s", func(tn Tensor) error { _, err := tn.Float64s(); return err }, `tensor "w": expected F64, got BF16`},
		{"Int64s", func(tn Tensor) error { _, err := tn.Int64s(); return err }, `tensor "w": expected I64, got BF16`},
		{"Int32s", func(tn Tensor) error { _, err := tn.Int32s(); return err }, `tensor "w": expected I32, got BF16`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.call(Tensor{Name: "w", DType: got})
			if err == nil {
				t.Fatalf("%s on a %s tensor: want an error, got nil", c.name, got)
			}
			if err.Error() != c.want {
				t.Errorf("%s message drifted:\n got: %s\nwant: %s", c.name, err.Error(), c.want)
			}
		})
	}
}

// TestTypedLE_acceptsMatchingDType is the other half: a matching dtype must still reach
// reinterpretLE and decode, not just fail differently. Four little-endian bytes decode
// to one float32 / one int32; the 8-byte cases cover F64 and I64.
func TestTypedLE_acceptsMatchingDType(t *testing.T) {
	f32, err := (Tensor{Name: "w", DType: "F32", raw: []byte{0x00, 0x00, 0x80, 0x3f}}).Float32s()
	if err != nil || len(f32) != 1 || f32[0] != 1 {
		t.Errorf("Float32s: got %v, %v; want [1] and no error", f32, err)
	}
	i32, err := (Tensor{Name: "w", DType: "I32", raw: []byte{0x2a, 0x00, 0x00, 0x00}}).Int32s()
	if err != nil || len(i32) != 1 || i32[0] != 42 {
		t.Errorf("Int32s: got %v, %v; want [42] and no error", i32, err)
	}
	i64, err := (Tensor{Name: "w", DType: "I64", raw: []byte{0x2a, 0, 0, 0, 0, 0, 0, 0}}).Int64s()
	if err != nil || len(i64) != 1 || i64[0] != 42 {
		t.Errorf("Int64s: got %v, %v; want [42] and no error", i64, err)
	}
	f64, err := (Tensor{Name: "w", DType: "F64", raw: []byte{0, 0, 0, 0, 0, 0, 0xf0, 0x3f}}).Float64s()
	if err != nil || len(f64) != 1 || f64[0] != 1 {
		t.Errorf("Float64s: got %v, %v; want [1] and no error", f64, err)
	}
}

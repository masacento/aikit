package encoder

import (
	"io"
	"testing"
)

// Both concrete encoders satisfy io.Closer, so a caller holding the (Hard-tier,
// frozen) Encoder interface can still release resources via a type assertion —
// the audit #19 goal, without widening the interface.
var (
	_ io.Closer = (*Model)(nil)
	_ io.Closer = (*ModelQ8)(nil)
)

// TestEncoder_ioCloser documents the release pattern for a held Encoder.
func TestEncoder_ioCloser(t *testing.T) {
	var enc Encoder = (*ModelQ8)(nil)
	if _, ok := enc.(io.Closer); !ok {
		t.Fatal("ModelQ8 does not satisfy io.Closer via the Encoder interface")
	}
	// A nil-safe Close on ModelQ8 (no mmap held).
	if err := (&ModelQ8{}).Close(); err != nil {
		t.Errorf("ModelQ8.Close() = %v, want nil", err)
	}
}

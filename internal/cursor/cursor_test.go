package cursor

import (
	"errors"
	"fmt"
	"testing"
)

// errTest is a stand-in for a package's own ErrFormat sentinel, so the test can
// confirm a Need failure still satisfies errors.Is against the CALLER's sentinel
// (each of the three real callers wraps a different one, and tests assert
// errors.Is(err, <package>.ErrFormat) — this must keep working through Cursor).
var errTest = errors.New("test: malformed format")

func errTestf(format string, args ...any) error {
	return fmt.Errorf(format+": %w", append(args, errTest)...)
}

func newTestCursor(b []byte) *Cursor {
	return &Cursor{B: b, Context: "test: blob", Errorf: errTestf}
}

func TestCursor_roundTrip(t *testing.T) {
	// u8=0x2A, u16=0x1234, u32=0xDEADBEEF, u64, f32, f64, then 3 raw bytes.
	b := []byte{
		0x2A,
		0x34, 0x12,
		0xEF, 0xBE, 0xAD, 0xDE,
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x80, 0x3F, // 1.0f
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // 1.0
		'x', 'y', 'z',
	}
	c := newTestCursor(b)
	if v := c.U8(); v != 0x2A {
		t.Fatalf("U8() = %#x, want 0x2A", v)
	}
	if v := c.U16(); v != 0x1234 {
		t.Fatalf("U16() = %#x, want 0x1234", v)
	}
	if v := c.U32(); v != 0xDEADBEEF {
		t.Fatalf("U32() = %#x, want 0xDEADBEEF", v)
	}
	if v := c.U64(); v != 1 {
		t.Fatalf("U64() = %d, want 1", v)
	}
	if v := c.F32(); v != 1.0 {
		t.Fatalf("F32() = %v, want 1.0", v)
	}
	if v := c.F64(); v != 1.0 {
		t.Fatalf("F64() = %v, want 1.0", v)
	}
	if got := string(c.Bytes(3)); got != "xyz" {
		t.Fatalf("Bytes(3) = %q, want %q", got, "xyz")
	}
	if c.Err != nil {
		t.Fatalf("Err = %v, want nil", c.Err)
	}
	if c.Remaining() != 0 {
		t.Fatalf("Remaining() = %d, want 0", c.Remaining())
	}
}

func TestCursor_truncatedSetsErrViaCallerSentinel(t *testing.T) {
	c := newTestCursor([]byte{0x01, 0x02})
	if v := c.U32(); v != 0 {
		t.Fatalf("U32() on truncated input = %d, want 0", v)
	}
	if c.Err == nil {
		t.Fatal("Err is nil after a truncated read")
	}
	if !errors.Is(c.Err, errTest) {
		t.Fatalf("Err = %v, does not wrap the caller's own sentinel", c.Err)
	}
}

func TestCursor_firstFailureSticks(t *testing.T) {
	c := newTestCursor([]byte{0x01})
	c.U32() // truncated: sets Err
	first := c.Err
	if v := c.U8(); v != 0 {
		t.Fatalf("U8() after Err was set = %d, want 0 (no further reads)", v)
	}
	if c.Err != first {
		t.Fatalf("Err changed after a second failing call: got %v, want unchanged %v", c.Err, first)
	}
}

func TestCursor_bytesNilOnTruncation(t *testing.T) {
	c := newTestCursor([]byte{0x01, 0x02})
	if got := c.Bytes(5); got != nil {
		t.Fatalf("Bytes(5) on a 2-byte input = %v, want nil", got)
	}
}

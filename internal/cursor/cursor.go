// Package cursor is the bounds-checked little-endian binary reader shared by
// aikit's three persisted-format loaders: ann/hnsw_persist.go's hcur, ann/
// flat_i8_persist.go's fcur, and embed/gguf.go's gcur. All three were
// independently written to the same shape (need/u32/u64/f32/u8, and fcur's
// own doc comment already named this as "the per-format cursor convention,
// alongside HNSW's hcur and gguf's gcur") — copy-pasted, not shared, so
// nothing guaranteed the bounds-check logic stayed identical across edits.
//
// Every read goes through Need(), so a truncated or hostile input sets Err
// and yields zeros instead of panicking — the "error or succeed, never
// crash" contract each of the three loaders documents.
//
// This is deliberately the small common core (Need/U8/U16/U32/U64/F32/F64/
// Remaining/Bytes) and nothing format-specific: allocation-driving length
// checks (hcur's readLen, fcur's nonNeg), config-scalar bounds (hcur's cfg),
// preallocation-hint clamping (gcur's hintLen), and format-specific readers
// (gcur's str, value) stay in each package, built on top of the embedded
// Cursor — they encode a real per-format policy decision (what counts as a
// hostile count for THIS format), not an accident of copy-paste.
package cursor

import (
	"encoding/binary"
	"math"
)

// Cursor is a bounds-checked little-endian reader over a byte slice.
type Cursor struct {
	B   []byte
	Pos int
	Err error

	// Context prefixes Need's truncation error, e.g. "ann: HNSW blob",
	// "ann: FlatI8 blob", "gguf". Required.
	Context string
	// Errorf builds the wrapped format error — the caller's own errFormatf,
	// so a Need failure still satisfies errors.Is(err, <package>.ErrFormat).
	// Required.
	Errorf func(format string, args ...any) error
}

// Need reports whether n more bytes remain, setting Err (via Errorf, so it
// carries the caller's own sentinel) and returning false otherwise. Every
// other read goes through this first, so a truncated or hostile input yields
// zeros instead of panicking, and the first failure sticks — later calls are
// no-ops once Err is set.
func (c *Cursor) Need(n int) bool {
	if c.Err != nil {
		return false
	}
	// n<0 guards a length field that overflowed int in the caller (e.g. a
	// hostile uint64 length ≥ 2^63 converted to int); comparing against the
	// remaining span without adding means c.Pos+n can't itself overflow.
	if n < 0 || n > len(c.B)-c.Pos {
		c.Err = c.Errorf("%s truncated (need %d at %d of %d)", c.Context, n, c.Pos, len(c.B))
		return false
	}
	return true
}

// Remaining is the number of unread bytes — the ceiling on any element count
// or length a caller derives from the input, since every element consumes at
// least one byte.
func (c *Cursor) Remaining() int { return len(c.B) - c.Pos }

// Bytes reads n raw bytes and advances past them, or returns nil if they
// aren't available.
func (c *Cursor) Bytes(n int) []byte {
	if !c.Need(n) {
		return nil
	}
	b := c.B[c.Pos : c.Pos+n]
	c.Pos += n
	return b
}

func (c *Cursor) U8() uint8 {
	if !c.Need(1) {
		return 0
	}
	v := c.B[c.Pos]
	c.Pos++
	return v
}

func (c *Cursor) U16() uint16 {
	if !c.Need(2) {
		return 0
	}
	v := binary.LittleEndian.Uint16(c.B[c.Pos:])
	c.Pos += 2
	return v
}

func (c *Cursor) U32() uint32 {
	if !c.Need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(c.B[c.Pos:])
	c.Pos += 4
	return v
}

func (c *Cursor) U64() uint64 {
	if !c.Need(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(c.B[c.Pos:])
	c.Pos += 8
	return v
}

func (c *Cursor) F32() float32 { return math.Float32frombits(c.U32()) }
func (c *Cursor) F64() float64 { return math.Float64frombits(c.U64()) }

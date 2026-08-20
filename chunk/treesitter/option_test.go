package treesitter

import (
	"fmt"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/chunk"
)

// largeGoSource builds a non-trivial Go file whose parse does enough work
// that a 1µs wall-clock budget reliably trips (many timeout checks), while a
// disabled budget parses it to completion.
func largeGoSource() []byte {
	var b strings.Builder
	b.WriteString("package x\n\n")
	for i := range 400 {
		fmt.Fprintf(&b, "func F%d(a, b int) int {\n\tc := a + b\n\tif c > %d {\n\t\treturn c * 2\n\t}\n\treturn c\n}\n\n", i, i)
	}
	return []byte(b.String())
}

// TestWithParseTimeoutMicros_Wired proves the option changes parse behavior:
// an impossibly tight 1µs budget forces the timeout→line fallback (parseErr
// increments), while a disabled budget (0) parses the same input to an AST
// with no fallback. The default New() keeps the 1s behavior.
func TestWithParseTimeoutMicros_Wired(t *testing.T) {
	src := largeGoSource()

	tight := New(WithParseTimeoutMicros(1)) // 1µs — always trips
	got, err := tight.Chunk(src, "go", chunk.DefaultChunkSize)
	if err != nil {
		t.Fatalf("tight.Chunk: %v", err)
	}
	if got == nil {
		t.Fatal("tight.Chunk returned no chunks")
	}
	if pe := tight.Stats().ParseErr; pe == 0 {
		t.Errorf("a 1µs budget should trip the timeout→fallback (parseErr>0); got parseErr=0")
	}

	off := New(WithParseTimeoutMicros(0)) // disabled — parse to completion
	got2, err := off.Chunk(src, "go", chunk.DefaultChunkSize)
	if err != nil {
		t.Fatalf("off.Chunk: %v", err)
	}
	if got2 == nil {
		t.Fatal("off.Chunk returned no chunks")
	}
	if pe := off.Stats().ParseErr; pe != 0 {
		t.Errorf("a disabled budget must not time out; got parseErr=%d", pe)
	}
	if nr := off.Stats().NilRoot; nr != 0 {
		t.Errorf("disabled budget produced a nil root: %d", nr)
	}
}

// TestNew_DefaultTimeoutUnchanged pins that the zero-value Chunker and New()
// with no options keep the 1s default (neither disabled nor overridden), so
// existing callers — including the registry's init()-registered instance —
// are byte-for-byte unaffected by the new option surface.
func TestNew_DefaultTimeoutUnchanged(t *testing.T) {
	for _, c := range []*Chunker{New(), {}} {
		if c.parseTimeoutDisabled {
			t.Errorf("default Chunker should not disable the timeout")
		}
		if c.parseTimeoutOverride != 0 {
			t.Errorf("default Chunker should not override the timeout; got %d", c.parseTimeoutOverride)
		}
	}
}

// TestParseTimeout_Reports pins the accessor a consumer uses to assert which
// tier it configured (ken's CLI disables it for reproducible builds; the
// server keeps a bounded value).
func TestParseTimeout_Reports(t *testing.T) {
	if m, d := New().ParseTimeout(); d || m != parseTimeoutMicros {
		t.Errorf("New(): got (%d, %v), want (%d, false)", m, d, parseTimeoutMicros)
	}
	if m, d := New(WithParseTimeoutMicros(0)).ParseTimeout(); !d {
		t.Errorf("WithParseTimeoutMicros(0): got disabled=%v micros=%d, want disabled", d, m)
	}
	if m, d := New(WithParseTimeoutMicros(5_000_000)).ParseTimeout(); d || m != 5_000_000 {
		t.Errorf("WithParseTimeoutMicros(5s): got (%d, %v), want (5000000, false)", m, d)
	}
}

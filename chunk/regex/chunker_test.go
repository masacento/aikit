package regex

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/chunk"
)

func TestEmptyAndTinyInput(t *testing.T) {
	if cs := chunkStr(t, "go", 200, ""); cs != nil {
		t.Errorf("empty source: got %d chunks, want nil", len(cs))
	}
	cs := chunkStr(t, "go", 200, "package main\n")
	if len(cs) != 1 || cs[0].StartLine != 1 || cs[0].EndLine != 1 {
		t.Fatalf("one-line source: got %+v", cs)
	}
	assertFidelity(t, "package main\n", cs)
}

func TestNoDefinitions_SizeSplitOnly(t *testing.T) {
	// A "go" file with zero definitions — pure data. No boundaries, so the
	// engine degrades to size-bounded line splitting, still byte-exact.
	var b strings.Builder
	for i := range 60 {
		fmt.Fprintf(&b, "x%d := %d\n", i, i)
	}
	src := b.String()
	cs := chunkStr(t, "go", 100, src)
	assertFidelity(t, src, cs)
	assertMaxSize(t, cs, 100)
	if len(cs) < 2 {
		t.Fatalf("expected size-splitting into ≥2 chunks, got %d", len(cs))
	}
}

func TestOversizedSingleFunction_LineSplitFallback(t *testing.T) {
	// One function whose body alone exceeds chunkSize and contains no
	// nested definitions: there is no boundary to snap to, so it is split
	// at line boundaries (ken's DESIGN.md §2 / deliverable 4 explicit exception).
	var b strings.Builder
	b.WriteString("func Big() {\n")
	for i := range 40 {
		fmt.Fprintf(&b, "\tstep%d()\n", i)
	}
	b.WriteString("}\n")
	src := b.String()

	cs := chunkStr(t, "go", 120, src)
	assertFidelity(t, src, cs)
	assertMaxSize(t, cs, 120)
	if len(cs) < 2 {
		t.Fatalf("oversized function should line-split into ≥2 chunks, got %d", len(cs))
	}
	// First chunk still begins at the func declaration.
	if cs[0].StartLine != 1 {
		t.Errorf("first chunk StartLine=%d, want 1 (the func line)", cs[0].StartLine)
	}
}

func TestUnknownLanguage_Degrades(t *testing.T) {
	src := "module Main where\nmain = putStrLn \"hi\"\n"
	cs := chunkStr(t, "haskell", 200, src) // not a regex-supported language
	assertFidelity(t, src, cs)
}

func TestInterfaceContract(t *testing.T) {
	c := New()
	if c.Name() != "regex" {
		t.Errorf("Name() = %q, want regex", c.Name())
	}
	got := c.SupportedLanguages()
	want := []string{"go", "java", "python", "rust", "typescript"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("SupportedLanguages() = %v, want %v", got, want)
	}
	// init() must have registered us in the chunk registry.
	reg, err := chunk.Get("regex")
	if err != nil {
		t.Fatalf("chunk.Get(regex): %v", err)
	}
	if reg.Name() != "regex" {
		t.Errorf("registered chunker Name() = %q, want regex", reg.Name())
	}
}

// TestAttachBlockAtLine0 is the regression for AUDIT #22: an attach block (license
// header, module doc) that starts at line 0 and runs into the first definition
// must keep the definition with it. The walk's `b-1 >= 1` guard halted at b==1, so
// the boundary landed on a line INSIDE the comment and the definition was split
// from its doc comment (the first chunk held only line 0). `b > 0` reaches line 0.
func TestAttachBlockAtLine0(t *testing.T) {
	// A 2-line comment header at line 0, then a def with a body large enough to
	// force a chunk split (chunkSize 120 splits mid-alpha, not per line). With the
	// fix the header attaches to def alpha (chunk 0 = L1-7, contains "def alpha");
	// with the bug the boundary snaps to line 2 instead of line 0, so chunk 0 is
	// just "# ...line one" and the doc comment is split from its definition.
	var b strings.Builder
	b.WriteString("# license header line one\n# license header line two\ndef alpha():\n")
	for i := range 8 {
		fmt.Fprintf(&b, "    step_%d()\n", i)
	}
	src := b.String()
	cs := chunkStr(t, "python", 120, src)
	assertFidelity(t, src, cs)
	if !strings.Contains(cs[0].Text, "def alpha") {
		t.Errorf("first chunk lost its definition — the line-0 attach block split it off:\n%q", cs[0].Text)
	}
}

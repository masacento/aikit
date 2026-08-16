package treesitter

import (
	"testing"

	"github.com/odvcencio/gotreesitter"
	"github.com/odvcencio/gotreesitter/grammars"
)

// TestSmoke_GoParse confirms gotreesitter is wired up correctly: the
// "go" grammar resolves via DetectLanguageByName, NewParser returns a
// working parser, and the root node exposes named children with byte
// offsets. If this fails, the chunker is unbuildable regardless of any
// algorithm bugs.
func TestSmoke_GoParse(t *testing.T) {
	src := []byte(`package demo

func Add(a, b int) int { return a + b }
func Sub(a, b int) int { return a - b }

type Pair struct { X, Y int }
`)
	entry := grammars.DetectLanguageByName("go")
	if entry == nil {
		t.Fatal("gotreesitter has no \"go\" grammar")
	}
	lang := entry.Language()
	tree, err := gotreesitter.NewParser(lang).Parse(src)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	root := tree.RootNode()
	if got := root.NamedChildCount(); got < 3 {
		t.Errorf("root named children = %d, want ≥3 (package + 2 funcs + type)", got)
	}
	// Each top-level child should have a non-empty byte range inside src.
	for i := range root.NamedChildCount() {
		c := root.NamedChild(i)
		if c == nil {
			t.Errorf("named child %d is nil", i)
			continue
		}
		s, e := c.StartByte(), c.EndByte()
		if e <= s || int(e) > len(src) {
			t.Errorf("child %d (%s) bad range [%d, %d) src len=%d",
				i, c.Type(lang), s, e, len(src))
		}
	}
}

// TestSmoke_SQLParse is the parse smoke for the SQL grammar added for real-world
// popularity (TIOBE #8; SO 2025 58.6%). Like TestSmoke_GoParse it only checks
// that the grammar resolves and produces named children with sane byte ranges —
// exact chunk boundaries are explicitly not pinned (see the package stability
// note), so this is a smoke test, not a golden.
func TestSmoke_SQLParse(t *testing.T) {
	src := []byte(`SELECT id, name FROM users WHERE active = 1 ORDER BY name;

CREATE TABLE accounts (
    id    INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE
);

UPDATE users SET active = 0 WHERE id = 42;
`)
	entry := grammars.DetectLanguageByName("sql")
	if entry == nil {
		t.Fatal("gotreesitter has no \"sql\" grammar")
	}
	lang := entry.Language()
	tree, err := gotreesitter.NewParser(lang).Parse(src)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	root := tree.RootNode()
	if got := root.NamedChildCount(); got < 3 {
		t.Errorf("root named children = %d, want ≥3 (SELECT + CREATE TABLE + UPDATE)", got)
	}
	for i := range root.NamedChildCount() {
		c := root.NamedChild(i)
		if c == nil {
			t.Errorf("named child %d is nil", i)
			continue
		}
		s, e := c.StartByte(), c.EndByte()
		if e <= s || int(e) > len(src) {
			t.Errorf("child %d (%s) bad range [%d, %d) src len=%d",
				i, c.Type(lang), s, e, len(src))
		}
	}
}

// TestSmoke_HTMLParse / CSSParse / SCSSParse / RParse: parse smokes for the
// popularity-driven grammars added alongside SQL. Same bar as TestSmoke_GoParse
// — grammar resolves, named children with sane byte ranges; boundaries not
// pinned. The SCSS snippet deliberately uses SCSS-only constructs ($var/@mixin)
// that the plain css grammar errors on, since routing .scss to the dedicated
// scss grammar (vs the css alias) is the whole point of that entry.

func smokeParse(t *testing.T, grammar string, src []byte, minChildren int) {
	t.Helper()
	entry := grammars.DetectLanguageByName(grammar)
	if entry == nil {
		t.Fatalf("gotreesitter has no %q grammar", grammar)
	}
	lang := entry.Language()
	tree, err := gotreesitter.NewParser(lang).Parse(src)
	if err != nil {
		t.Fatalf("%s parse: %v", grammar, err)
	}
	root := tree.RootNode()
	if got := int(root.NamedChildCount()); got < minChildren {
		t.Errorf("%s root named children = %d, want ≥%d", grammar, got, minChildren)
	}
	for i := range root.NamedChildCount() {
		c := root.NamedChild(i)
		if c == nil {
			t.Errorf("%s named child %d is nil", grammar, i)
			continue
		}
		s, e := c.StartByte(), c.EndByte()
		if e <= s || int(e) > len(src) {
			t.Errorf("%s child %d (%s) bad range [%d, %d) src len=%d",
				grammar, i, c.Type(lang), s, e, len(src))
		}
	}
}

func TestSmoke_HTMLParse(t *testing.T) {
	smokeParse(t, "html", []byte(`<html><head><title>x</title></head><body><div class="c">hi</div></body></html>`), 1)
}

func TestSmoke_CSSParse(t *testing.T) {
	smokeParse(t, "css", []byte(".card { color: red; }\nbody { margin: 0; padding: 0; }\n"), 2)
}

func TestSmoke_SCSSParse(t *testing.T) {
	// $var + @mixin + @include: SCSS-only; the plain css grammar errors on these.
	smokeParse(t, "scss", []byte("$primary: #333;\n@mixin box($p) { padding: $p; }\n.card { @include box(4px); color: $primary; }\n"), 3)
}

func TestSmoke_RParse(t *testing.T) {
	smokeParse(t, "r", []byte("add <- function(a, b) {\n  return(a + b)\n}\nx <- add(1, 2)\n"), 2)
}

package regex

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
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

// TestPrescreen_neverHidesAMatch is the gate for the literal-prefix prescreen
// (lens doc §3.2). The claim is EXACT, not heuristic: a line the prescreen
// rejects provably cannot match, so the chunker's output is unchanged.
//
// It is checked the strong way — for every rule of every registered language,
// against every line of every language fixture in the package, assert that
// skipping the regexp is only ever done when the regexp would have said no.
// Testing it on same-language lines only would miss the case that matters, a
// rule fed a line it was not written for.
func TestPrescreen_neverHidesAMatch(t *testing.T) {
	var lines [][]byte
	for _, src := range allFixtureSources(t) {
		for l := range bytes.SplitSeq(src, []byte("\n")) {
			lines = append(lines, l, bytes.TrimLeft(l, " \t"))
		}
	}
	if len(lines) < 20_000 {
		t.Fatalf("only %d probe lines; the probe set is too small to be a gate", len(lines))
	}

	checked, screened, screenedFB := 0, 0, 0
	for lang, r := range languageRules {
		sets := []struct {
			name string
			res  []*regexp.Regexp
			pre  [][]byte
			fb   []*[256]bool
		}{
			{"defs", r.defs, r.defsPre, r.defsFB},
			{"skip", r.skip, r.skipPre, r.skipFB},
			{"attach", r.attach, r.attachPre, r.attachFB},
		}
		for _, set := range sets {
			if len(set.pre) != len(set.res) || len(set.fb) != len(set.res) {
				t.Fatalf("%s/%s: %d prefixes / %d fb for %d patterns", lang, set.name, len(set.pre), len(set.fb), len(set.res))
			}
			for i, re := range set.res {
				p, s := set.pre[i], set.fb[i]
				for _, line := range lines {
					checked++
					// Mirror anyMatch's screen decision exactly.
					reject := false
					switch {
					case len(p) > 0:
						reject = !bytes.HasPrefix(line, p)
					case s != nil:
						reject = len(line) == 0 || !s[line[0]]
						if reject {
							screenedFB++
						}
					}
					if !reject {
						continue // reaches the regexp
					}
					screened++
					if re.Match(line) {
						t.Fatalf("%s/%s[%d] %q: prescreen rejected a line the pattern MATCHES: %q",
							lang, set.name, i, re.String(), line)
					}
				}
			}
		}
	}
	if screened == 0 {
		t.Fatal("the prescreen never fired — this test proves nothing")
	}
	if screenedFB == 0 {
		t.Fatal("the first-byte-set screen never fired — the FB path is untested")
	}
	t.Logf("%d (pattern, line) pairs checked; prescreen skipped the regexp for %d (%.1f%%), of which %d via first-byte set",
		checked, screened, 100*float64(screened)/float64(checked), screenedFB)
}

// TestPrescreen_prefixesAreWhatWeThink pins the prefixes themselves. If a rule
// is reworded and its literal prefix silently becomes empty, the prescreen stops
// firing for it and nothing else notices — the output stays correct and the
// speed quietly goes away.
func TestPrescreen_prefixesAreWhatWeThink(t *testing.T) {
	got := map[string]int{}
	for lang, r := range languageRules {
		for _, pre := range [][][]byte{r.defsPre, r.skipPre, r.attachPre} {
			for _, p := range pre {
				if len(p) > 0 {
					got[lang]++
				}
			}
		}
	}
	// Measured at the time of the change; a drop means a rule lost its prefix.
	// Measured at the time of the change. These are floors, not targets: a drop
	// means a rule was reworded into a shape whose prefix cannot be extracted.
	want := map[string]int{"go": 7, "java": 5, "python": 3, "rust": 8, "typescript": 6}
	for lang, n := range want {
		if got[lang] < n {
			t.Errorf("%s: %d patterns have a literal prefix, want at least %d — a rule lost its prefix and its prescreen",
				lang, got[lang], n)
		}
	}
	t.Logf("patterns with a usable prefix, by language: %v", got)
}

// allFixtureSources returns probe text for the prescreen gate: the package's
// per-language fixtures plus every .go file in the repository.
//
// The repository is included because the fixtures are a few hundred lines
// between them, which is not enough to be a gate — a prescreen bug that fires on
// one line in ten thousand would pass on them. Real source also supplies the
// shapes nobody writes into a fixture: continuation lines, generics, struct
// tags, strings containing keywords.
func allFixtureSources(t *testing.T) [][]byte {
	t.Helper()
	out := [][]byte{}
	for _, s := range []string{goSrc, tsSrc, pySrc, rustSrc, javaSrc} {
		out = append(out, []byte(s))
	}
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case "benchmarks", "testdata", ".git", ".venv":
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		out = append(out, b)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestAnchoredLiteralPrefix covers the extraction directly, including the guards
// that no rule in this package currently exercises.
//
// Those guards would otherwise be untested code: a mutation removing the
// case-folding check survives the corpus-wide soundness gate, because no rule
// today uses (?i). They are cheap to state and the next rule someone adds may
// need them.
func TestAnchoredLiteralPrefix(t *testing.T) {
	for _, tc := range []struct {
		pattern string
		want    string
		why     string
	}{
		{`^func\b`, "func", "a word boundary after the literal must not block extraction — regexp.LiteralPrefix returns \"\" here"},
		{`^class\s+\w+`, "class", "nor must a following character class"},
		{`^//`, "//", "pure literal"},
		{`^/\*`, "/*", "escaped metacharacter is still a literal"},
		{`^type\b`, "type", ""},
		{`^(export\s+)?class\s+\w+`, "", "an OPTIONAL leading group means no literal is required"},
		{`^(if|for|while)\b`, "", "an alternation has no single literal prefix"},
		{`^[A-Za-z_]\w*`, "", "a character class is not a literal"},
		{`^(?i)func\b`, "", "case-folded: a byte compare is not the same test"},
		{`^日本\b`, "", "non-ASCII: kept out so the prefix stays a byte comparison"},
		{`^`, "", "anchor alone"},
	} {
		got := string(anchoredLiteralPrefix(tc.pattern))
		if got != tc.want {
			t.Errorf("anchoredLiteralPrefix(%q) = %q, want %q — %s", tc.pattern, got, tc.want, tc.why)
		}
		// Whatever it returns must be sound: a match has to start with it.
		if got != "" {
			re := regexp.MustCompile(tc.pattern)
			if loc := re.FindStringIndex(got + "zzz"); loc != nil && loc[0] != 0 {
				t.Errorf("%q: prefix %q does not begin the match", tc.pattern, got)
			}
		}
	}
}

// TestAnchoredLiteralPrefix_rejectsUnanchored checks the panic. The prescreen's
// entire soundness argument is anchoring, so an unanchored rule must stop
// registration loudly rather than silently get no prescreen — the latter reads
// as "this rule is just slow" forever.
func TestAnchoredLiteralPrefix_rejectsUnanchored(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("an unanchored pattern was accepted; the prescreen assumes anchoring")
		}
	}()
	anchoredLiteralPrefix(`func\b`)
}

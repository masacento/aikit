package main

import (
	"regexp"
	"strings"
	"testing"
)

func TestPair_String(t *testing.T) {
	p := Pair{Module: "github.com/townsendmerino/aikit", Version: "v1.4.0"}
	if got, want := p.String(), "github.com/townsendmerino/aikit@v1.4.0"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}

func TestFirstLine(t *testing.T) {
	cases := []struct{ in, want string }{
		{"hello", "hello"},
		{"\n\nhello\nworld", "hello"},
		{"  hello  ", "hello"},
		{"\n \n\t\n", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := firstLine(c.in); got != c.want {
			t.Errorf("firstLine(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestMatchLine(t *testing.T) {
	s := "line one\nERROR: something broke\nline three"
	re := regexp.MustCompile(`^ERROR:`)
	if got, want := matchLine(s, re), "ERROR: something broke"; got != want {
		t.Errorf("matchLine (match found) = %q, want %q", got, want)
	}
	// No match anywhere: falls back to firstLine.
	re2 := regexp.MustCompile(`^NOPE:`)
	if got, want := matchLine(s, re2), "line one"; got != want {
		t.Errorf("matchLine (no match, fallback) = %q, want %q", got, want)
	}
}

func TestBuildError_skipsNoiseLines(t *testing.T) {
	s := strings.Join([]string{
		"go: downloading github.com/foo/bar v1.0.0",
		"go: added github.com/foo/bar v1.0.0",
		"go: found github.com/foo/bar in github.com/foo/bar v1.0.0",
		"# github.com/foo/bar",
		"./main.go:10:2: undefined: Foo",
	}, "\n")
	if got, want := buildError(s), "./main.go:10:2: undefined: Foo"; got != want {
		t.Errorf("buildError = %q, want %q", got, want)
	}
}

func TestBuildError_allNoiseFallsBackToFirstLine(t *testing.T) {
	s := "go: downloading github.com/foo/bar v1.0.0\n# github.com/foo/bar\n"
	// Every line is filtered noise, so buildError falls back to firstLine, which itself
	// skips only BLANK lines — the first (non-blank) noise line comes back.
	want := firstLine(s)
	if got := buildError(s); got != want {
		t.Errorf("buildError (all noise) = %q, want fallback firstLine() = %q", got, want)
	}
}

func TestTrim(t *testing.T) {
	if got, want := trim("  hello  "), "hello"; got != want {
		t.Errorf("trim(padded) = %q, want %q", got, want)
	}
	long := strings.Repeat("x", 200)
	got := trim(long)
	if len(got) != 90 {
		t.Errorf("trim(200 chars) len = %d, want 90", len(got))
	}
	if got != long[:90] {
		t.Error("trim truncated to the wrong prefix")
	}
}

func TestShortLabel(t *testing.T) {
	if got, want := shortLabel(rootPath), "root"; got != want {
		t.Errorf("shortLabel(root) = %q, want %q", got, want)
	}
	if got, want := shortLabel(rootPath+"/gpu/annmetal"), "gpu/annmetal"; got != want {
		t.Errorf("shortLabel(subpath) = %q, want %q", got, want)
	}
}

func TestListOrNone(t *testing.T) {
	if got, want := listOrNone(nil), " none"; got != want {
		t.Errorf("listOrNone(nil) = %q, want %q", got, want)
	}
	if got, want := listOrNone([]string{}), " none"; got != want {
		t.Errorf("listOrNone(empty) = %q, want %q", got, want)
	}
	if got, want := listOrNone([]string{"a", "b"}), " a b"; got != want {
		t.Errorf("listOrNone([a b]) = %q, want %q", got, want)
	}
}

func TestEnvOr(t *testing.T) {
	t.Setenv("CONSUMER_GATE_TEST_VAR", "")
	if got, want := envOr("CONSUMER_GATE_TEST_VAR", "fallback"), "fallback"; got != want {
		t.Errorf("envOr(unset) = %q, want %q", got, want)
	}
	t.Setenv("CONSUMER_GATE_TEST_VAR", "explicit")
	if got, want := envOr("CONSUMER_GATE_TEST_VAR", "fallback"), "explicit"; got != want {
		t.Errorf("envOr(set) = %q, want %q", got, want)
	}
}

func TestEnvInt(t *testing.T) {
	t.Setenv("CONSUMER_GATE_TEST_INT", "")
	if got, want := envInt("CONSUMER_GATE_TEST_INT", 7), 7; got != want {
		t.Errorf("envInt(unset) = %d, want %d", got, want)
	}
	t.Setenv("CONSUMER_GATE_TEST_INT", "42")
	if got, want := envInt("CONSUMER_GATE_TEST_INT", 7), 42; got != want {
		t.Errorf("envInt(set) = %d, want %d", got, want)
	}
	// Malformed: falls back to the default rather than erroring the whole gate.
	t.Setenv("CONSUMER_GATE_TEST_INT", "not-a-number")
	if got, want := envInt("CONSUMER_GATE_TEST_INT", 7), 7; got != want {
		t.Errorf("envInt(malformed) = %d, want fallback %d", got, want)
	}
}

// TestDecidePairs_noArgsUsesLatestForEveryPublished is the default "watch" mode: every
// published module, pinned to @latest.
func TestDecidePairs_noArgsUsesLatestForEveryPublished(t *testing.T) {
	pub := []string{rootPath, rootPath + "/gpu"}
	pairs, code, msg := decidePairs(nil, pub)
	if code != -1 || msg != "" {
		t.Fatalf("code=%d msg=%q, want -1 and empty", code, msg)
	}
	if len(pairs) != 2 {
		t.Fatalf("len(pairs) = %d, want 2", len(pairs))
	}
	for i, p := range pairs {
		if p.Module != pub[i] || p.Version != "latest" {
			t.Errorf("pairs[%d] = %+v, want Module=%s Version=latest", i, p, pub[i])
		}
	}
}

func TestDecidePairs_explicitPairs(t *testing.T) {
	pairs, code, msg := decidePairs([]string{"mod/a@v1.0.0", "mod/b@v2.0.0"}, nil)
	if code != -1 || msg != "" {
		t.Fatalf("code=%d msg=%q, want -1 and empty", code, msg)
	}
	want := []Pair{{"mod/a", "v1.0.0"}, {"mod/b", "v2.0.0"}}
	if len(pairs) != len(want) {
		t.Fatalf("pairs = %v, want %v", pairs, want)
	}
	for i := range want {
		if pairs[i] != want[i] {
			t.Errorf("pairs[%d] = %+v, want %+v", i, pairs[i], want[i])
		}
	}
}

// TestDecidePairs_explicitPairUsesLastAt confirms a module path containing "@" (unlikely
// but not impossible for a pseudo-version like module@v0.0.0-20230101000000-abcdef) splits
// on the LAST "@", not the first.
func TestDecidePairs_explicitPairUsesLastAt(t *testing.T) {
	pairs, code, _ := decidePairs([]string{"mod/a@v0.0.0-20230101000000-abcdef"}, nil)
	if code != -1 {
		t.Fatalf("code = %d, want -1", code)
	}
	if len(pairs) != 1 || pairs[0].Module != "mod/a" || pairs[0].Version != "v0.0.0-20230101000000-abcdef" {
		t.Errorf("pairs = %v, want [{mod/a v0.0.0-20230101000000-abcdef}]", pairs)
	}
}

func TestDecidePairs_malformedArgument(t *testing.T) {
	_, code, msg := decidePairs([]string{"not-a-valid-pair"}, nil)
	if code != 2 {
		t.Errorf("code = %d, want 2 (inconclusive)", code)
	}
	if !strings.Contains(msg, "not-a-valid-pair") || !strings.Contains(msg, "MODULE@VERSION") {
		t.Errorf("msg = %q, want it to name the bad argument and the expected shape", msg)
	}
}

func TestDecidePairs_tagFlagWrongArgCount(t *testing.T) {
	_, code, msg := decidePairs([]string{"--tag"}, nil)
	if code != 2 {
		t.Errorf("code = %d, want 2", code)
	}
	if !strings.Contains(msg, "exactly one ref") {
		t.Errorf("msg = %q, want it to explain the arg-count requirement", msg)
	}
	_, code, _ = decidePairs([]string{"--tag", "v1.0.0", "extra"}, nil)
	if code != 2 {
		t.Errorf("code (too many args) = %d, want 2", code)
	}
}

func TestDecidePairs_tagFlagMapsToPair(t *testing.T) {
	pairs, code, msg := decidePairs([]string{"--tag", "v1.4.0"}, nil)
	if code != -1 || msg != "" {
		t.Fatalf("code=%d msg=%q, want -1 and empty (v1.4.0 is a recognized, published root tag)", code, msg)
	}
	if len(pairs) != 1 || pairs[0].Module != rootPath || pairs[0].Version != "v1.4.0" {
		t.Errorf("pairs = %v, want [{%s v1.4.0}]", pairs, rootPath)
	}
}

func TestDecidePairs_tagFlagUnrecognizedRef(t *testing.T) {
	_, code, msg := decidePairs([]string{"--tag", "not-a-release-tag"}, nil)
	if code != 2 {
		t.Errorf("code = %d, want 2", code)
	}
	if !strings.Contains(msg, "cannot cover it") {
		t.Errorf("msg = %q, want it to explain the tag could not be mapped", msg)
	}
}

// --- mapTagToPair ------------------------------------------------------------------

func TestMapTagToPair(t *testing.T) {
	cases := []struct {
		name    string
		ref     string
		want    Pair
		wantErr bool
	}{
		{"root tag", "v1.4.0", Pair{rootPath, "v1.4.0"}, false},
		{"root tag with refs/tags prefix", "refs/tags/v1.4.0", Pair{rootPath, "v1.4.0"}, false},
		{"gpu series tag", "gpu/v0.28.0", Pair{rootPath + "/gpu", "v0.28.0"}, false},
		{"gpu backend subpath tag", "gpu/annmetal/v0.28.0", Pair{rootPath + "/gpu/annmetal", "v0.28.0"}, false},
		{"treesitter submodule tag", "chunk/treesitter/v1.0.0", Pair{rootPath + "/chunk/treesitter", "v1.0.0"}, false},
		{"not a release tag at all", "not-a-tag", Pair{}, true},
		{"published-looking but not a version", "gpu/notaversion", Pair{}, true},
		{
			// A subpath-shaped ref pointing at a module the classification maps do not
			// list as published (internal, or simply nonexistent) must error, not
			// silently map — this is the "gate cannot skip a module unnoticed" contract.
			"subpath tag for an unpublished module", "tools/v1.0.0", Pair{}, true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := mapTagToPair(c.ref)
			if c.wantErr {
				if err == nil {
					t.Fatalf("mapTagToPair(%q) = %+v, nil — want an error", c.ref, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("mapTagToPair(%q) unexpected error: %v", c.ref, err)
			}
			if got != c.want {
				t.Errorf("mapTagToPair(%q) = %+v, want %+v", c.ref, got, c.want)
			}
		})
	}
}

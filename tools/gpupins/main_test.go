package main

import "testing"

// a realistic backend go.mod: a require block with both aikit series, an indirect block, and
// the two replace directives that must NOT be mistaken for requires.
const sampleGomod = `module github.com/townsendmerino/aikit/gpu/annmetal

go 1.26.6

require (
	github.com/townsendmerino/aikit v1.17.0
	github.com/townsendmerino/aikit/gpu v0.28.0
)

require (
	github.com/ebitengine/purego v0.10.1 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/townsendmerino/aikit => ../../

replace github.com/townsendmerino/aikit/gpu => ../
`

func TestRequires_readsBothSeriesNotReplaces(t *testing.T) {
	root, gpu := requires(sampleGomod)
	if root != "v1.17.0" {
		t.Errorf("root require = %q, want v1.17.0 (a replace line must not be read as the require)", root)
	}
	if gpu != "v0.28.0" {
		t.Errorf("gpu require = %q, want v0.28.0", gpu)
	}
}

func TestPinField_classifies(t *testing.T) {
	cases := []struct {
		got, want, state string
	}{
		{"v1.17.1", "v1.17.1", "ok"},
		{"v1.17.0", "v1.17.1", "DRIFT"},
		{"", "v1.17.1", "MISSING"},
	}
	for _, c := range cases {
		if f := pinField("root", c.got, c.want); f.State != c.state {
			t.Errorf("pinField(got=%q,want=%q).State = %q, want %q", c.got, c.want, f.State, c.state)
		}
	}
}

func TestRewrite_bumpsRootOnlyAndReports(t *testing.T) {
	// gpu already current, root stale — only the root line should change, and the edit
	// list should name exactly that transition.
	out, edits := rewrite(sampleGomod, "v1.17.1", "v0.28.0")
	gotRoot, gotGpu := requires(out)
	if gotRoot != "v1.17.1" {
		t.Errorf("after rewrite root = %q, want v1.17.1", gotRoot)
	}
	if gotGpu != "v0.28.0" {
		t.Errorf("after rewrite gpu = %q, want v0.28.0 (unchanged)", gotGpu)
	}
	if len(edits) != 1 || edits[0] != "root v1.17.0→v1.17.1" {
		t.Errorf("edits = %v, want exactly [root v1.17.0→v1.17.1]", edits)
	}
	// the replace directives must survive untouched.
	if want := "replace github.com/townsendmerino/aikit => ../../"; !contains(out, want) {
		t.Errorf("rewrite dropped a replace directive; output:\n%s", out)
	}
}

func TestRewrite_noopWhenCurrent(t *testing.T) {
	out, edits := rewrite(sampleGomod, "v1.17.0", "v0.28.0")
	if len(edits) != 0 {
		t.Errorf("edits = %v, want none when already current", edits)
	}
	if out != sampleGomod {
		t.Errorf("rewrite changed bytes on a no-op:\n%s", out)
	}
}

func TestRewrite_bumpsBoth(t *testing.T) {
	out, edits := rewrite(sampleGomod, "v1.18.0", "v0.29.0")
	gotRoot, gotGpu := requires(out)
	if gotRoot != "v1.18.0" || gotGpu != "v0.29.0" {
		t.Errorf("after rewrite root=%q gpu=%q, want v1.18.0 / v0.29.0", gotRoot, gotGpu)
	}
	if len(edits) != 2 {
		t.Errorf("edits = %v, want two (root and gpu)", edits)
	}
}

func contains(hay, needle string) bool {
	for i := 0; i+len(needle) <= len(hay); i++ {
		if hay[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

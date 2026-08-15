package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeGoMod(t *testing.T, dir, modulePath string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %q: %v", dir, err)
	}
	body := "module " + modulePath + "\n\ngo 1.21\n"
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(body), 0o644); err != nil {
		t.Fatalf("write go.mod in %q: %v", dir, err)
	}
}

func TestModulePath(t *testing.T) {
	dir := t.TempDir()
	writeGoMod(t, dir, "example.com/foo")
	got, err := modulePath(filepath.Join(dir, "go.mod"))
	if err != nil {
		t.Fatalf("modulePath error = %v", err)
	}
	if got != "example.com/foo" {
		t.Errorf("modulePath = %q, want %q", got, "example.com/foo")
	}
}

func TestModulePath_noModuleLine(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("go 1.21\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	if _, err := modulePath(filepath.Join(dir, "go.mod")); err == nil {
		t.Error("modulePath on a go.mod with no module line returned nil error")
	}
}

func TestModulePath_missingFile(t *testing.T) {
	if _, err := modulePath(filepath.Join(t.TempDir(), "nope", "go.mod")); err == nil {
		t.Error("modulePath on a missing file returned nil error")
	}
}

func TestEnumerate(t *testing.T) {
	root := t.TempDir()
	writeGoMod(t, root, "example.com/root")
	writeGoMod(t, filepath.Join(root, "sub", "a"), "example.com/root/sub/a")
	// Inside .git — must be skipped, the same as a real checkout's own .git/modules noise.
	writeGoMod(t, filepath.Join(root, ".git", "weird"), "example.com/should-not-be-seen")

	mods, err := enumerate(root)
	if err != nil {
		t.Fatalf("enumerate error = %v", err)
	}
	want := map[string]bool{"example.com/root": true, "example.com/root/sub/a": true}
	if len(mods) != len(want) {
		t.Fatalf("enumerate = %v, want exactly %v", mods, want)
	}
	for _, m := range mods {
		if !want[m] {
			t.Errorf("enumerate returned unexpected module %q (want only %v)", m, want)
		}
		if strings.Contains(m, "should-not-be-seen") {
			t.Errorf("enumerate descended into .git: found %q", m)
		}
	}
}

// TestClassifyTree_realMaps exercises classifyTree against the REAL published/internal
// maps this package ships — the actual classification data the gate runs with in
// production, not a synthetic stand-in — using a temp tree assembled from real entries of
// each map.
func TestClassifyTree_realMaps(t *testing.T) {
	root := t.TempDir()
	writeGoMod(t, root, rootPath)
	writeGoMod(t, filepath.Join(root, "gpu"), rootPath+"/gpu")
	writeGoMod(t, filepath.Join(root, "tools"), rootPath+"/tools") // internal, must not appear in pub

	pub, total, err := classifyTree(root)
	if err != nil {
		t.Fatalf("classifyTree error = %v", err)
	}
	if total != 3 {
		t.Errorf("total = %d, want 3", total)
	}
	wantPub := []string{rootPath, rootPath + "/gpu"} // sorted; tools/ is internal, excluded
	if len(pub) != len(wantPub) {
		t.Fatalf("pub = %v, want %v", pub, wantPub)
	}
	for i, w := range wantPub {
		if pub[i] != w {
			t.Errorf("pub[%d] = %q, want %q", i, pub[i], w)
		}
	}
}

// TestClassifyTree_unclassifiedModuleFails is the integrity check this package's whole
// design rests on: a go.mod whose module path is in NEITHER map must fail the gate, not
// silently pass through — this is what stops a newly added module from skipping the check
// unnoticed.
func TestClassifyTree_unclassifiedModuleFails(t *testing.T) {
	root := t.TempDir()
	writeGoMod(t, root, rootPath)
	writeGoMod(t, filepath.Join(root, "somewhere", "new"), "github.com/townsendmerino/aikit/somewhere/new")

	_, _, err := classifyTree(root)
	if err == nil {
		t.Fatal("classifyTree with an unclassified module returned nil error")
	}
	if !strings.Contains(err.Error(), "unclassified") {
		t.Errorf("error = %q, want it to name the module as unclassified", err.Error())
	}
}

// TestClassifyTree_zeroModulesFails pins "never an empty green": enumerating zero go.mod
// files must fail, not vacuously pass as "0/0 published, all fine".
func TestClassifyTree_zeroModulesFails(t *testing.T) {
	root := t.TempDir() // no go.mod anywhere
	_, total, err := classifyTree(root)
	if err == nil {
		t.Fatal("classifyTree over a tree with zero go.mod files returned nil error")
	}
	if total != 0 {
		t.Errorf("total = %d, want 0", total)
	}
}

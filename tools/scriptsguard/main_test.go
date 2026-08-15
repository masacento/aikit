package main

import (
	"os"
	"path/filepath"
	"testing"
)

// chdir changes to dir for the test's duration and restores the original cwd on cleanup —
// run() resolves its target tree via gpumod.RepoRoot(), which reads os.Getwd(), so
// exercising it against a fake tree means actually being in it.
func chdir(t *testing.T, dir string) {
	t.Helper()
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("os.Chdir(%q): %v", dir, err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(orig); err != nil {
			t.Fatalf("restoring cwd to %q: %v", orig, err)
		}
	})
}

// fakeRepo builds a minimal repo root (a ".git" marker directory, for gpumod.RepoRoot's
// walk-up) with the given scripts/-relative files created (empty content; scriptsguard
// classifies purely by path).
func fakeRepo(t *testing.T, files ...string) string {
	t.Helper()
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	for _, rel := range files {
		p := filepath.Join(root, "scripts", rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatalf("mkdir for %q: %v", rel, err)
		}
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatalf("write %q: %v", rel, err)
		}
	}
	return root
}

func TestRun_ok_onlyOracleAndFixtures(t *testing.T) {
	root := fakeRepo(t, "oracle/pin_bge.py", "oracle/pin_gte.py", "fixtures/prep_beir.py")
	chdir(t, root)
	if got := run(); got != 0 {
		t.Errorf("run() = %d, want 0 (oracle + fixtures only, no stray)", got)
	}
}

func TestRun_fail_strayPy(t *testing.T) {
	root := fakeRepo(t, "oracle/pin_bge.py", "loose_script.py")
	chdir(t, root)
	if got := run(); got != 1 {
		t.Errorf("run() = %d, want 1 (a .py directly under scripts/ is residue)", got)
	}
}

func TestRun_fail_strayShAnywhere(t *testing.T) {
	root := fakeRepo(t, "oracle/pin_bge.py", "oracle/build.sh")
	chdir(t, root)
	if got := run(); got != 1 {
		t.Errorf("run() = %d, want 1 (.sh anywhere under scripts/ is a deciding-shell regression)", got)
	}
}

func TestRun_cannotEvaluate_noScriptsDir(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	// No scripts/ directory at all — the denominator is zero, must not read as a pass.
	chdir(t, root)
	if got := run(); got != 2 {
		t.Errorf("run() = %d, want 2 (CANNOT-EVALUATE: no .py found at all)", got)
	}
}

func TestRun_cannotEvaluate_emptyScriptsDir(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "scripts"), 0o755); err != nil {
		t.Fatalf("mkdir scripts: %v", err)
	}
	chdir(t, root)
	if got := run(); got != 2 {
		t.Errorf("run() = %d, want 2 (CANNOT-EVALUATE: scripts/ exists but has zero .py)", got)
	}
}

// TestRun_nestedFixturesPathIsNotOracle guards the prefix-match logic itself: a path like
// "oracle_backup/x.py" must NOT be classified as oracle/ just because it shares a string
// prefix — the check is path-separator-anchored (HasPrefix(rel, "oracle"+Separator)).
func TestRun_nestedFixturesPathIsNotOracle(t *testing.T) {
	root := fakeRepo(t, "oracle/pin_bge.py", "oracle_backup/sneaky.py")
	chdir(t, root)
	if got := run(); got != 1 {
		t.Errorf("run() = %d, want 1 — \"oracle_backup/\" must not be treated as oracle/", got)
	}
}

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// coreSteps is what preflight KNOWS ci.yml's core (root-module) job runs, in order. This
// list is the TRIPWIRE: tools/ is stdlib-only so preflight cannot parse the YAML and derive
// the mirror, but it can read the step names as text and fail when they change — so the next
// check CI's core job gains cannot silently go un-mirrored. Update this list AND
// buildChecks/implementedAs/notImplemented together, on purpose — TestCoreStepsAccountedFor,
// below, enforces that every name here lands in exactly one of the other two.
var coreSteps = []string{
	"gofmt",
	"build",
	"vet",
	"golangci-lint",
	"no cgo deps in core graph",
	"build with cgo disabled",
	"test (race)",
	"test (aikit_checks)",
	"fuzz (smoke)",
}

// implementedAs maps each ci.yml core-job step name preflight mirrors to the Check.Name
// buildChecks runs it under — identical except "test (race)", which preflight runs UNRACED
// (too slow for a pre-push gate; see main.go's doc comment) under the name "go test".
var implementedAs = map[string]string{
	"gofmt":                     "gofmt",
	"build":                     "build",
	"vet":                       "vet",
	"golangci-lint":             "golangci-lint",
	"no cgo deps in core graph": "no cgo deps in core graph",
	"build with cgo disabled":   "build (cgo-free)",
	"test (race)":               "go test",
	"test (aikit_checks)":       "test (aikit_checks)",
}

// notImplemented is every ci.yml core-job step name preflight deliberately does not run at
// all, with why. This is the enforcement point for the finding that prompted it: a prose
// exclusion policy in main.go's doc comment is not checked against reality, so a step could
// go unmirrored with no reason that actually fit the policy (as "no cgo deps in core graph"
// and "test (aikit_checks)" both did, despite needing neither network nor a GPU). A name in
// neither this map nor implementedAs now fails TestCoreStepsAccountedFor instead of just
// being absent from a comment nobody re-checks.
var notImplemented = map[string]string{
	"fuzz (smoke)": "real wall-clock cost (8 targets x 15s x up to 2 retries) — too slow for a pre-push gate",
}

func TestCICoreStepsUnchanged(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	got, err := coreJobStepNames(filepath.Join(root, ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatalf("reading ci.yml core steps: %v", err)
	}
	if strings.Join(got, "|") != strings.Join(coreSteps, "|") {
		t.Fatalf("ci.yml core-job steps changed — the preflight mirror may be stale.\n"+
			"  got:  %v\n  want: %v\n"+
			"Reconcile: update coreSteps in this file AND decide whether tools/preflight should now run the new/changed check.",
			got, coreSteps)
	}
}

// TestCoreStepsAccountedFor is the check TestCICoreStepsUnchanged does NOT do: it verifies
// every ci.yml core-job step name is either actually implemented by buildChecks (via
// implementedAs) or explicitly, individually excluded (via notImplemented) — never both,
// never neither. buildChecks itself does no I/O (a Check's Run closure only executes inside
// gate.RunAll), so this runs fast without a real build/lint/test pass.
func TestCoreStepsAccountedFor(t *testing.T) {
	root, err := gpumod.RepoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	implemented := map[string]bool{}
	for _, c := range buildChecks(root, false) {
		implemented[c.Name] = true
	}
	for _, name := range coreSteps {
		wantName, isImplemented := implementedAs[name]
		_, isExcluded := notImplemented[name]
		switch {
		case isImplemented && isExcluded:
			t.Errorf("ci.yml core step %q is listed in BOTH implementedAs and notImplemented — fix the bookkeeping", name)
		case !isImplemented && !isExcluded:
			t.Errorf("ci.yml core step %q is in neither implementedAs nor notImplemented — a silent gap "+
				"(either implement it in buildChecks or add a reason to notImplemented)", name)
		case isImplemented && !implemented[wantName]:
			t.Errorf("ci.yml core step %q claims to run as preflight check %q (implementedAs), "+
				"but buildChecks has no such check", name, wantName)
		}
	}
}

// coreJobStepNames reads ci.yml as TEXT and returns the `- name:` step names inside the
// `core:` job, in order, stopping at the next 2-space-indented job header.
func coreJobStepNames(ymlPath string) ([]string, error) {
	b, err := os.ReadFile(ymlPath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(b), "\n")
	start := -1
	for i, ln := range lines {
		if strings.TrimRight(ln, " ") == "  core:" {
			start = i
			break
		}
	}
	if start < 0 {
		return nil, fmt.Errorf("no `  core:` job found in %s", ymlPath)
	}
	var names []string
	for _, ln := range lines[start+1:] {
		if len(ln) > 2 && ln[0] == ' ' && ln[1] == ' ' && isJobNameStart(ln[2]) {
			break // a sibling job header at 2-space indent — end of the core block
		}
		t := strings.TrimSpace(ln)
		if strings.HasPrefix(t, "- name:") {
			names = append(names, strings.TrimSpace(t[len("- name:"):]))
		}
	}
	return names, nil
}

func isJobNameStart(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

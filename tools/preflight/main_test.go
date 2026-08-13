package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

// coreSteps is what preflight KNOWS ci.yml's core (root-module) job runs, in order. preflight
// mirrors the subset that needs no network, no GPU, and not the -race triple-time; the rest
// it deliberately leaves to CI. This list is the TRIPWIRE: tools/ is stdlib-only so preflight
// cannot parse the YAML and derive the mirror, but it can read the step names as text and
// fail when they change — so the next check CI's core job gains cannot silently go
// un-mirrored. Update this list AND main.go's check set together, on purpose.
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

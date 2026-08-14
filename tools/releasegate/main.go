// Command releasegate validates that the repo is ready to release vX.Y.Z — the root-module
// release gate, ported from scripts/release-gate.sh onto tools/gate. release.yml re-runs it
// on the tag before publishing, so it is the last check between a commit and a Go module
// version that is immutable once the proxy fetches it.
//
// FOUR CHECKS:
//  1. CHANGELOG.md has a `## [X.Y.Z]` section AND a `[X.Y.Z]:` compare link — a release
//     deliverable; its failure messages name exactly what is missing.
//  2. golangci-lint clean against .golangci.yml, build-pinned via `go run @v2.11.4` (the
//     same the gpu gate, preflight, and CI resolve to) so a local gate reports identically.
//  3. apidiff shows NO incompatible change in any Hard-tier package vs the previous tag —
//     the 1.0 compatibility bar — with documented-Experimental symbols/members exempted.
//  4. The core module pulls no external dependency beyond golang.org/x/text and
//     golang.org/x/sys (darwin-only madvise) — the cgo-free, dependency-light invariant.
//
// EXTERNAL TOOLS ARE INVOKED, NOT DEPENDED ON. golangci-lint and apidiff are run via
// `go run pkg@ver`; nothing enters tools/go.mod, and either being unbuildable is
// INCONCLUSIVE — could not judge — never a pass. Previous-tag resolution takes the highest
// v* by version, excluding the one being released, so the retracted v1.8.0 is never picked.
//
// Usage:  go run -C tools ./releasegate <version, e.g. 1.4.0 (no leading v)>
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

const rootMod = "github.com/townsendmerino/aikit"

// hardPkgs are the Hard-tier packages under the 1.0 compatibility guarantee (README).
var hardPkgs = []string{"topk", "ann", "bm25", "fuse", "embed", "encoder", "chunk"}

// experimentalSyms: symbols documented Experimental that LIVE INSIDE a Hard-tier package —
// their leading symbol name is the whole match, so an apidiff line led by one is exempt.
var experimentalSyms = map[string][]string{
	"encoder": {"Backend", "RegisterBackend", "NewBackend", "LoadQ8", "ModelQ8", "WeightsQ8", "LayerWeightsQ8", "LoadBERT", "BERT", "LoadSPLADE", "SPLADE", "LoadCrossEncoder", "CrossEncoder"},
	"ann":     {"FlatBinary", "NewFlatBinary", "NewFlatBinaryOverquery", "DefaultOverquery"},
	"embed":   {"LoadMmap"},
}

// experimentalMembers: Experimental MEMBERS on Hard-tier types — matched as whole apidiff
// symbol paths so listing one member never exempts the rest of its type.
var experimentalMembers = map[string][]string{
	"embed": {"(*SafetensorsFile).ReleaseTensors", "Tensor.SubF32"},
}

const apidiffPkg = "golang.org/x/exp/cmd/apidiff@latest"

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	if len(args) < 1 || strings.TrimSpace(args[0]) == "" {
		fmt.Fprintln(os.Stderr, "usage: releasegate <version, e.g. 1.4.0 (no leading v)>")
		return 2
	}
	ver := args[0]
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("release-gate: INCONCLUSIVE — cannot locate repo root: " + err.Error())
		return 2
	}

	checks := []gate.Check{
		{Name: "changelog", Run: func() gate.Cell { return checkChangelog(root, ver) }},
		{Name: "golangci-lint", Run: func() gate.Cell { return checkLint(root) }},
		{Name: "apidiff", Run: func() gate.Cell { return checkAPIDiff(root, ver) }},
		{Name: "core-deps", Run: func() gate.Cell { return checkCoreDeps(root) }},
	}
	cells := gate.RunAll(checks)
	rep := gate.ReconcileWith(cells, gate.FailWins)

	switch rep.Outcome {
	case gate.Fail:
		fmt.Printf("release-gate: v%s FAILED ✗\n", ver)
		return 1
	case gate.Inconclusive:
		fmt.Printf("release-gate: v%s INCONCLUSIVE ✗ — a required external tool was unavailable\n", ver)
		return 2
	}
	fmt.Printf("release-gate: v%s OK ✓\n", ver)
	return 0
}

// (1) CHANGELOG — the messages name exactly what is missing (a release deliverable).
func checkChangelog(root, ver string) gate.Cell {
	data, err := os.ReadFile(filepath.Join(root, "CHANGELOG.md"))
	if err != nil {
		return failMsg("changelog", "cannot read CHANGELOG.md: "+err.Error())
	}
	s := string(data)
	q := regexp.QuoteMeta(ver)
	var missing []string
	if !regexp.MustCompile(`(?m)^## \[` + q + `\]`).MatchString(s) {
		missing = append(missing, fmt.Sprintf("CHANGELOG.md has no '## [%s]' section", ver))
	}
	if !regexp.MustCompile(`(?m)^\[` + q + `\]: `).MatchString(s) {
		missing = append(missing, fmt.Sprintf("CHANGELOG.md has no '[%s]:' compare link", ver))
	}
	if len(missing) > 0 {
		for _, m := range missing {
			fmt.Println("::error::release-gate: " + m)
		}
		return failMsg("changelog", strings.Join(missing, "; "))
	}
	fmt.Println("release-gate: CHANGELOG section + compare link present")
	return okCell("changelog")
}

// (2) golangci-lint — pinned build, on the whole root module.
func checkLint(root string) gate.Cell {
	fmt.Println("release-gate: golangci-lint (.golangci.yml)")
	if _, rc := runIn(root, "go", "run", gpumod.GolangciLint, "version"); rc != 0 {
		return inconMsg("golangci-lint", "could not build the pinned golangci-lint")
	}
	// CANARY: trust a "clean" only after the linter flags its fixture (see tools/canary).
	cout, _ := runIn(filepath.Join(root, canary.FixturesDir), "go", "run", gpumod.GolangciLint, "run", "--build-tags", "canaryfixture")
	if res := canary.CheckGolangci(cout); !res.Fired {
		return inconMsg("golangci-lint", "CANNOT-EVALUATE — "+res.Reason)
	}
	out, rc := runIn(root, "go", "run", gpumod.GolangciLint, "run", "./...")
	if rc != 0 {
		fmt.Println("::error::release-gate: golangci-lint reported issues")
		fmt.Println(strings.TrimRight(out, "\n"))
		return failMsg("golangci-lint", "issues reported")
	}
	return okCell("golangci-lint")
}

// (3) apidiff — Hard-tier compatibility vs the previous tag.
func checkAPIDiff(root, ver string) gate.Cell {
	prev := prevTag(root, ver)
	if prev == "" {
		fmt.Println("release-gate: no previous tag — skipping apidiff")
		return okCell("apidiff")
	}
	fmt.Printf("release-gate: apidiff Hard tier %s → current tree\n", prev)
	wt, err := os.MkdirTemp("", "releasegate-wt")
	if err != nil {
		return inconMsg("apidiff", "mktemp: "+err.Error())
	}
	if _, rc := runIn(root, "git", "worktree", "add", "-q", "-f", wt, prev); rc != 0 {
		return inconMsg("apidiff", "git worktree add "+prev+" failed")
	}
	defer func() {
		runIn(root, "git", "worktree", "remove", "--force", wt)
		runIn(root, "git", "worktree", "prune")
	}()

	var broken []string
	for _, p := range hardPkgs {
		base := filepath.Join(os.TempDir(), "gate-"+p+".api")
		// apidiff exits non-zero only on a TOOL/LOAD error (it cannot build itself, or
		// cannot load the package) — reporting API changes is exit 0. So a non-zero here is
		// "could not judge", INCONCLUSIVE, never mistaken for a diff. -w writes the previous
		// tag's API from the worktree; -incompatible compares the current tree against it.
		if base_out, rc := runIn(wt, "go", "run", apidiffPkg, "-w", base, rootMod+"/"+p); rc != 0 {
			return inconMsg("apidiff", "apidiff could not record the "+prev+" baseline for "+p+": "+firstLine(base_out))
		}
		inc, rc := runIn(root, "go", "run", apidiffPkg, "-incompatible", base, rootMod+"/"+p)
		if rc != 0 {
			return inconMsg("apidiff", "apidiff could not compare "+p+" against "+prev+": "+firstLine(inc))
		}
		inc = filterExperimental(inc, p)
		if strings.TrimSpace(inc) != "" {
			fmt.Printf("::error::release-gate: apidiff: incompatible change in Hard-tier '%s' vs %s:\n%s\n", p, prev, strings.TrimRight(inc, "\n"))
			broken = append(broken, p)
		}
	}
	if len(broken) > 0 {
		return failMsg("apidiff", "Hard-tier breakage vs "+prev+" in: "+strings.Join(broken, " "))
	}
	return okCell("apidiff")
}

// (4) core dependency invariant.
func checkCoreDeps(root string) gate.Cell {
	out, _ := runIn(root, "go", "list", "-deps", "-f", "{{if not .Standard}}{{.ImportPath}}{{end}}", "./...")
	var ext []string
	allowed := regexp.MustCompile(`^golang\.org/x/(text|sys)(/|$)`)
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, rootMod) || allowed.MatchString(ln) {
			continue
		}
		ext = append(ext, ln)
	}
	if len(ext) > 0 {
		fmt.Println("::error::release-gate: core module pulls unexpected external dependency(ies): " + strings.Join(ext, " "))
		return failMsg("core-deps", "unexpected external deps: "+strings.Join(ext, " "))
	}
	return okCell("core-deps")
}

// prevTag is the highest v* tag by version, excluding the one being released — so a retracted
// old tag (v1.8.0) is never selected for a current-era release.
func prevTag(root, ver string) string {
	out, _ := runIn(root, "git", "tag", "--list", "v*", "--sort=-v:refname")
	for _, ln := range strings.Split(out, "\n") {
		t := strings.TrimSpace(ln)
		if t != "" && t != "v"+ver {
			return t
		}
	}
	return ""
}

// filterExperimental drops apidiff incompatible lines whose subject is documented
// Experimental for this package — the leading-symbol cases and the whole-member cases.
func filterExperimental(inc, pkg string) string {
	lines := strings.Split(inc, "\n")
	syms := experimentalSyms[pkg]
	var symRe *regexp.Regexp
	if len(syms) > 0 {
		quoted := make([]string, len(syms))
		for i, s := range syms {
			quoted[i] = regexp.QuoteMeta(s)
		}
		symRe = regexp.MustCompile(`^- \(?\*?(` + strings.Join(quoted, "|") + `)[).: ]`)
	}
	members := experimentalMembers[pkg]
	var kept []string
	for _, ln := range lines {
		if symRe != nil && symRe.MatchString(ln) {
			continue
		}
		drop := false
		for _, m := range members {
			if strings.HasPrefix(ln, "- "+m+":") {
				drop = true
				break
			}
		}
		if !drop {
			kept = append(kept, ln)
		}
	}
	return strings.Join(kept, "\n")
}

// runIn runs a command in dir with the ambient environment (release-gate, like its shell,
// does not override GOWORK — on the Linux/CI target there is no go.work). Returns combined
// output and exit code.
func runIn(dir, name string, args ...string) (string, int) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return string(out), ee.ExitCode()
	}
	return string(out), -1
}

func okCell(name string) gate.Cell { return gate.Cell{Name: name, Outcome: gate.OK} }
func failMsg(name, msg string) gate.Cell {
	return gate.Cell{Name: name, Outcome: gate.Fail, Fields: []gate.Field{{Key: "msg", State: msg}}}
}
func inconMsg(name, msg string) gate.Cell {
	fmt.Println("release-gate: INCONCLUSIVE — " + name + ": " + msg)
	return gate.Cell{Name: name, Outcome: gate.Inconclusive, Fields: []gate.Field{{Key: "msg", State: msg}}}
}

func firstLine(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			return strings.TrimSpace(ln)
		}
	}
	return ""
}

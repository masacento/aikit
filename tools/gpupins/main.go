// Command gpupins is the HOME of one fact: this repo publishes two independently-versioned
// module series — the root `github.com/townsendmerino/aikit` (tags `vX.Y.Z`) and the GPU
// module `github.com/townsendmerino/aikit/gpu` (tags `gpu/vX.Y.Z`) — and the eight backend
// modules under gpu/ must `require` BOTH by an exact, current version. A go.mod `require` is
// static text that MVS reads; it cannot be derived at build time. So the invariant is gated
// here instead: each of the eight names the latest root tag and the latest gpu tag, drift is
// a FAIL, and `--fix` rewrites the requires to the current tags so no human restates them.
//
// WHY THIS EXISTS. The two-series fact has caused a defect four times — three downstream in
// goinfer, once here: the root series advanced to v1.17.1 while all eight backends still
// pinned aikit v1.17.0, because RELEASING.md step 3 asks a human to hand-edit eight go.mod
// files and one release didn't. A stale `require` is not merely cosmetic: the eight are
// consumer-facing modules whose only way to name the root/gpu surface they compile against
// is this line, and a wrong guess surfaces as a compile error inside someone else's
// dependency (RELEASING.md "Backend submodules" documents exactly that failure). The fifth
// instance now has a place to look and a check that goes red before a consumer hits it.
//
// THE INVARIANT, precisely: for each of gpu/{anncuda,annmetal,enccuda,encmetal,qwencuda,
// qwenmetal,visioncuda,visionmetal}/go.mod,
//   - `require github.com/townsendmerino/aikit`      == the latest `vX.Y.Z` tag, and
//   - `require github.com/townsendmerino/aikit/gpu`  == the latest `gpu/vX.Y.Z` tag.
//
// The `replace` directives are kept — they are for local development and Go ignores them from
// a dependency's go.mod, so they never mask the require for a consumer.
//
// A LEGITIMATE, BRIEF RED. Between tagging a new root (or gpu) version and running
// `--fix` + re-tagging the eight (RELEASING.md step 3, now one command), the eight lag by
// design and this gate is red on main. That red is the point — it says "finish the release",
// not "a bug shipped" — and the direct-push workflow (no branch protection, see RELEASING.md)
// makes a transient red on main the right nudge rather than a blocked merge.
//
// OUT OF SCOPE, on purpose: chunk/treesitter (frozen at v1.0.0 for the v1.0 line), the
// examples/ and benchmarks/ modules (in-tree, `replace`-backed, not published), and the
// `go 1.26.6` toolchain directive — a different axis with its own single source (go.mod).
//
// Usage:
//
//	go run -C tools ./gpupins          # check; drift is a FAIL
//	go run -C tools ./gpupins --fix    # rewrite the eight requires to the latest tags
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
)

const (
	rootMod = "github.com/townsendmerino/aikit"
	gpuMod  = "github.com/townsendmerino/aikit/gpu"
)

// backends are the eight GPU backend modules that restate both series. Enumerated, not
// discovered, on purpose: this is a fixed set the invariant is written about, and a new
// backend that should join it is a deliberate edit here — the same reason gpumod.ModuleDirs
// walks a known tree rather than trusting whatever go.mod happens to exist.
var backends = []string{
	"gpu/anncuda", "gpu/annmetal",
	"gpu/enccuda", "gpu/encmetal",
	"gpu/qwencuda", "gpu/qwenmetal",
	"gpu/visioncuda", "gpu/visionmetal",
}

var (
	reRootTag = regexp.MustCompile(`^v\d+\.\d+\.\d+$`)
	reGpuTag  = regexp.MustCompile(`^gpu/v\d+\.\d+\.\d+$`)
)

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fix := len(args) > 0 && args[0] == "--fix"

	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("gpupins: INCONCLUSIVE — cannot locate repo root: " + err.Error())
		return 2
	}

	// The two targets the eight must name. A tag we cannot resolve is INCONCLUSIVE — the
	// gate could not judge (and --fix has nothing to write) — never a silent pass.
	wantRoot := latestTag(root, "v*", reRootTag)
	wantGpuTag := latestTag(root, "gpu/v*", reGpuTag)
	if wantRoot == "" || wantGpuTag == "" {
		fmt.Println("gpupins: INCONCLUSIVE — could not resolve the latest tag(s) from git " +
			"(need `git tag`; CI must checkout with fetch-depth: 0). " +
			"root=" + orNone(wantRoot) + " gpu=" + orNone(wantGpuTag))
		return 2
	}
	wantGpu := strings.TrimPrefix(wantGpuTag, "gpu/") // the require names the version, not the tag path

	p := gpumod.Provenance(root)
	fmt.Printf("aikit gpu backend pins — %s%s — %s\n", p.Commit, p.Dirty, p.Date)
	fmt.Printf("targets: %s %s   |   %s %s\n\n", rootMod, wantRoot, gpuMod, wantGpu)

	if fix {
		return runFix(root, wantRoot, wantGpu)
	}

	checks := make([]gate.Check, 0, len(backends))
	for _, m := range backends {
		checks = append(checks, gate.Check{Name: m, Run: func() gate.Cell { return checkModule(root, m, wantRoot, wantGpu) }})
	}
	cells := gate.RunAll(checks)

	for _, c := range cells {
		fmt.Printf("  %-22s %s\n", c.Name, statusLine(c))
	}
	rep := gate.ReconcileWith(cells, gate.FailWins)

	fmt.Println()
	switch rep.Outcome {
	case gate.Fail:
		fmt.Printf("gpupins: %d/%d backend(s) drift from the current tags — run `go run -C tools ./gpupins --fix`.\n", rep.Fail, rep.Total)
		fmt.Println(gate.Verdict(gate.Fail, fmt.Sprintf("%d/%d pin the latest root+gpu tags", rep.Pass, rep.Total)))
		return 1
	case gate.Inconclusive:
		fmt.Println(gate.Verdict(gate.Inconclusive, "a backend go.mod could not be read"))
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("all %d backends pin %s + gpu %s", rep.Total, wantRoot, wantGpu)))
	return 0
}

// checkModule reads one backend's go.mod and compares its two aikit requires to the targets.
func checkModule(root, mod, wantRoot, wantGpu string) gate.Cell {
	path := filepath.Join(root, mod, "go.mod")
	data, err := os.ReadFile(path)
	if err != nil {
		return gate.Cell{Name: mod, Outcome: gate.Inconclusive, Fields: []gate.Field{{Key: "read", State: "n/a", Detail: err.Error()}}}
	}
	gotRoot, gotGpu := requires(string(data))

	c := gate.Cell{Name: mod, Outcome: gate.OK}
	c.Fields = append(c.Fields, pinField("root", gotRoot, wantRoot))
	c.Fields = append(c.Fields, pinField("gpu", gotGpu, wantGpu))
	for _, f := range c.Fields {
		if f.State != "ok" {
			c.Outcome = gate.Fail
		}
	}
	return c
}

// pinField reports one require as ok / missing / a `got != want` drift.
func pinField(key, got, want string) gate.Field {
	switch {
	case got == "":
		return gate.Field{Key: key, State: "MISSING", Detail: "no require " + want}
	case got != want:
		return gate.Field{Key: key, State: "DRIFT", Detail: got + " != " + want}
	default:
		return gate.Field{Key: key, State: "ok"}
	}
}

// requires extracts the versions of the two aikit modules from go.mod text. It reads the
// require lines directly rather than pulling in golang.org/x/mod/modfile — the tools stay
// stdlib-only, and the lines are unambiguous once trimmed. The `replace` lines begin with
// "replace" (not the bare module path), so they are ignored; the aikit/gpu path is checked
// before the aikit path because the former is a prefix-extension of the latter.
func requires(gomod string) (root, gpu string) {
	for ln := range strings.SplitSeq(gomod, "\n") {
		t := strings.TrimSpace(ln)
		switch {
		case strings.HasPrefix(t, gpuMod+" "):
			gpu = versionField(t, gpuMod)
		case strings.HasPrefix(t, rootMod+" "):
			root = versionField(t, rootMod)
		}
	}
	return root, gpu
}

// versionField returns the version token following modPath on a trimmed require line.
func versionField(trimmed, modPath string) string {
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, modPath))
	if i := strings.IndexAny(rest, " \t"); i >= 0 {
		rest = rest[:i] // drop a trailing // comment or anything after the version
	}
	return rest
}

// runFix rewrites the two aikit requires in each backend's go.mod to the targets, preserving
// indentation and any trailing comment, and reports what it changed. This IS RELEASING.md
// step 3: the hand-edit of eight files becomes one command that reads the tags and writes
// them, so the version is derived from git rather than restated from memory.
func runFix(root, wantRoot, wantGpu string) int {
	changed, failed := 0, 0
	for _, mod := range backends {
		path := filepath.Join(root, mod, "go.mod")
		data, err := os.ReadFile(path)
		if err != nil {
			fmt.Printf("  %-22s could not read: %v\n", mod, err)
			failed++
			continue
		}
		out, edits := rewrite(string(data), wantRoot, wantGpu)
		if len(edits) == 0 {
			fmt.Printf("  %-22s already current\n", mod)
			continue
		}
		if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
			fmt.Printf("  %-22s could not write: %v\n", mod, err)
			failed++
			continue
		}
		fmt.Printf("  %-22s %s\n", mod, strings.Join(edits, "; "))
		changed++
	}
	fmt.Println()
	if failed > 0 {
		fmt.Println(gate.Verdict(gate.Inconclusive, fmt.Sprintf("%d rewritten, %d could not be written", changed, failed)))
		return 2
	}
	fmt.Println(gate.Verdict(gate.OK, fmt.Sprintf("%d backend(s) rewritten to %s + gpu %s", changed, wantRoot, wantGpu)))
	return 0
}

// rewrite returns go.mod text with the two aikit require versions set to the targets, and a
// list of human-readable edits made. Lines are matched by trimmed prefix so indentation and
// trailing comments survive; only the version token is replaced.
func rewrite(gomod, wantRoot, wantGpu string) (string, []string) {
	lines := strings.Split(gomod, "\n")
	var edits []string
	for i, ln := range lines {
		indent := ln[:len(ln)-len(strings.TrimLeft(ln, " \t"))]
		t := strings.TrimSpace(ln)
		switch {
		case strings.HasPrefix(t, gpuMod+" "):
			if got := versionField(t, gpuMod); got != wantGpu {
				lines[i] = setVersion(indent, gpuMod, t, wantGpu)
				edits = append(edits, "gpu "+got+"→"+wantGpu)
			}
		case strings.HasPrefix(t, rootMod+" "):
			if got := versionField(t, rootMod); got != wantRoot {
				lines[i] = setVersion(indent, rootMod, t, wantRoot)
				edits = append(edits, "root "+got+"→"+wantRoot)
			}
		}
	}
	return strings.Join(lines, "\n"), edits
}

// setVersion rebuilds a require line with a new version, keeping indentation and any tokens
// after the version (e.g. a `// indirect` comment — not present on these two, but preserved).
func setVersion(indent, modPath, trimmed, ver string) string {
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, modPath))
	tail := ""
	if i := strings.IndexAny(rest, " \t"); i >= 0 {
		tail = strings.TrimSpace(rest[i:])
	}
	line := indent + modPath + " " + ver
	if tail != "" {
		line += " " + tail
	}
	return line
}

// latestTag returns the highest tag matching re, using git's version sort. The glob is a
// prefilter (git's fnmatch lets `*` cross `/`, so `gpu/v*` also matches `gpu/<name>/v…`);
// re is the real filter that keeps only the module-root series.
func latestTag(root, glob string, re *regexp.Regexp) string {
	out, _ := gpumod.Exec(root, "", nil, "git", "tag", "--list", glob, "--sort=-v:refname")
	for ln := range strings.SplitSeq(out, "\n") {
		t := strings.TrimSpace(ln)
		if re.MatchString(t) {
			return t
		}
	}
	return ""
}

func statusLine(c gate.Cell) string {
	parts := make([]string, 0, len(c.Fields))
	for _, f := range c.Fields {
		parts = append(parts, f.String())
	}
	word := "ok"
	if c.Outcome != gate.OK {
		word = string(c.Outcome)
	}
	return fmt.Sprintf("%-4s  %s", word, strings.Join(parts, "  "))
}

func orNone(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}

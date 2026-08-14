// Command preflight runs what CI's core (root-module) job runs, locally, before pushing —
// the pre-push mirror of CI, ported from scripts/preflight.sh onto tools/gate.
//
// WHY IT EXISTS. Using CI as a first linter rather than a last check costs a full round trip
// per mistake (a revert once left an asm kernel with no caller; golangci-lint reports that
// in three seconds, but it was found from CI, after pushing, twice — 2026-08-12). So this is
// the same checks in the same order as ci.yml's core job, minus the parts that need the
// network, a GPU, or the -race triple-time: gofmt → build → vet → golangci-lint → cgo-free
// build → go test.
//
// ENVIRONMENT PARITY IS PART OF EACH CHECK. Reproducing a command without its environment is
// not reproducing the check. Every go step is pinned GOWORK=off explicitly on its own
// exec.Cmd — a developer's go.work must not decide what this reports — and the cgo-free build
// adds CGO_ENABLED=0; the env each check ran under is printed next to it. This CORRECTS the
// shell, which set GOWORK=off once as a global export (correct in effect, but invisible and
// not per-command) and used whatever golangci-lint was on PATH — a build that drifts by the
// Go it was compiled with (see gpumod.GolangciLint). This runs the linter build-pinned via
// `go run @v2.11.4`, the same one CI's goinstall v2.11.4 and the gpu gate resolve to.
//
// THE RELATIONSHIP TO ci.yml IS GUARDED. This check list mirrors the core job; a silent
// second copy drifts. tools/ is stdlib-only (no YAML parser), so a full derivation is out of
// reach — but main_test.go's tripwire reads ci.yml as text and fails if the core job's step
// names change, so the next check CI gains cannot silently go un-mirrored.
//
// Usage:
//
//	go run -C tools ./preflight            # everything
//	go run -C tools ./preflight --fast     # skip the test run (fmt/vet/lint/build only)
package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/tools/canary"
	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/gpumod"
	"github.com/townsendmerino/aikit/tools/skips"
)

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fast := len(args) > 0 && args[0] == "--fast"
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("PREFLIGHT: INCONCLUSIVE — cannot locate repo root: " + err.Error())
		return 2
	}
	p := gpumod.Provenance(root)
	fmt.Printf("aikit preflight — %s%s — root module\n\n", p.Commit, p.Dirty)

	checks := []gate.Check{
		{Name: "gofmt", Run: func() gate.Cell { return checkGofmt(root) }},
		{Name: "build", Run: func() gate.Cell { return goStep(root, "build", nil, "build", "./...") }},
		{Name: "vet", Run: func() gate.Cell { return goStep(root, "vet", nil, "vet", "./...") }},
		{Name: "golangci-lint", Run: func() gate.Cell { return checkLint(root) }},
		{Name: "build (cgo-free)", Run: func() gate.Cell { return goStep(root, "build (cgo-free)", []string{"CGO_ENABLED=0"}, "build", "./...") }},
	}
	if !fast {
		checks = append(checks, gate.Check{Name: "go test", Run: func() gate.Cell { return goTest(root) }})
	} else {
		fmt.Println("  (--fast: skipping go test; CI runs it with -race)")
	}
	cells := gate.RunAll(checks)

	for _, c := range cells {
		fmt.Printf("  %-18s %-26s %s\n", c.Name, "["+field(c, "env")+"]", statusWord(c.Outcome))
		for _, f := range c.Fields {
			if f.Key == "line" {
				fmt.Printf("      %s\n", f.State)
			}
		}
	}
	rep := gate.ReconcileWith(cells, gate.FailWins)

	fmt.Println()
	switch rep.Outcome {
	case gate.Fail:
		fmt.Printf("PREFLIGHT: %d check(s) failed — fix before pushing.\n", rep.Fail)
		return 1
	case gate.Inconclusive:
		fmt.Printf("PREFLIGHT: INCONCLUSIVE — %d check(s) could not be run (external tool unavailable).\n", rep.Incon)
		return 2
	}
	fmt.Println("PREFLIGHT: clean. CI still runs -race, the cgo-deps guard, aikit_checks, fuzz, the gpu jobs and vulncheck.")
	return 0
}

// checkGofmt: gofmt reports by printing filenames and still exits 0, so EMPTINESS is the
// check. .venv is not Go source.
func checkGofmt(root string) gate.Cell {
	out, _ := runIn(root, nil, "gofmt", "-l", ".")
	var bad []string
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln != "" && !strings.HasPrefix(ln, ".venv/") {
			bad = append(bad, ln)
		}
	}
	if len(bad) > 0 {
		c := cell("gofmt", gate.Fail, "none")
		for _, b := range bad {
			c.Fields = append(c.Fields, gate.Field{Key: "line", State: b})
		}
		return c
	}
	return cell("gofmt", gate.OK, "none")
}

// goStep runs a `go` subcommand with GOWORK=off (+ any extra env), recording the env it ran
// under. A non-zero exit is a FAIL with the first lines of output.
func goStep(root, name string, extraEnv []string, args ...string) gate.Cell {
	env := append([]string{"GOWORK=off"}, extraEnv...)
	out, rc := runIn(root, env, "go", args...)
	if rc != 0 {
		c := cell(name, gate.Fail, strings.Join(env, " "))
		c.Fields = append(c.Fields, headLines(out, 12)...)
		return c
	}
	return cell(name, gate.OK, strings.Join(env, " "))
}

// goTest runs `go test ./...` through the skip census (AK3), so a green here reports the skip
// tally by reason as a denominator rather than folding skips into a bare pass. Output is
// captured (preflight stays concise); on failure the failing lines are surfaced, and either
// way the census summary is printed under the row.
func goTest(root string) gate.Cell {
	var buf bytes.Buffer
	res, _ := skips.Run(root, []string{"GOWORK=off"}, []string{"./..."}, &buf)
	outcome := gate.OK
	if res.ExitCode != 0 {
		outcome = gate.Fail
	}
	c := cell("go test", outcome, "GOWORK=off")
	if outcome == gate.Fail {
		for _, ln := range strings.Split(buf.String(), "\n") {
			if strings.HasPrefix(ln, "---") || strings.HasPrefix(ln, "FAIL") || strings.HasPrefix(ln, "panic") {
				c.Fields = append(c.Fields, gate.Field{Key: "line", State: strings.TrimRight(ln, "\r")})
				if countLines(c) >= 8 {
					break
				}
			}
		}
	}
	for _, ln := range strings.Split("skip census — "+res.Summary(), "\n") {
		c.Fields = append(c.Fields, gate.Field{Key: "line", State: ln})
	}
	return c
}

// checkLint runs the pinned golangci-lint on the ROOT module. A preflight `version` run
// separates "the linter could not be built/run" (INCONCLUSIVE — an external tool the check
// depends on) from "the linter found issues" (FAIL). Build-pinned via `go run @v2.11.4`.
func checkLint(root string) gate.Cell {
	if _, rc := runIn(root, []string{"GOWORK=off"}, "go", "run", gpumod.GolangciLint, "version"); rc != 0 {
		return cell("golangci-lint", gate.Inconclusive, "GOWORK=off")
	}
	// CANARY: a "clean" is trusted only after the linter flags its fixture (see tools/canary).
	cout, _ := runIn(filepath.Join(root, canary.FixturesDir), []string{"GOWORK=off"}, "go", "run", gpumod.GolangciLint, "run", "--build-tags", "canaryfixture")
	if res := canary.CheckGolangci(cout); !res.Fired {
		c := cell("golangci-lint", gate.Inconclusive, "GOWORK=off")
		c.Fields = append(c.Fields, gate.Field{Key: "line", State: "CANNOT-EVALUATE — " + res.Reason})
		return c
	}
	out, rc := runIn(root, []string{"GOWORK=off"}, "go", "run", gpumod.GolangciLint, "run")
	if rc != 0 {
		c := cell("golangci-lint", gate.Fail, "GOWORK=off")
		c.Fields = append(c.Fields, headLines(out, 12)...)
		return c
	}
	return cell("golangci-lint", gate.OK, "GOWORK=off")
}

// runIn runs a command in root with os.Environ plus the explicit overrides — the pins are
// set per-command, not left to an inherited shell export.
func runIn(root string, extraEnv []string, name string, args ...string) (string, int) {
	cmd := exec.Command(name, args...)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), extraEnv...)
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return string(out), ee.ExitCode()
	}
	return string(out), -1
}

func cell(name string, o gate.Outcome, env string) gate.Cell {
	return gate.Cell{Name: name, Outcome: o, Fields: []gate.Field{{Key: "env", State: env}}}
}

func field(c gate.Cell, key string) string {
	for _, f := range c.Fields {
		if f.Key == key {
			return f.State
		}
	}
	return ""
}

func statusWord(o gate.Outcome) string {
	switch o {
	case gate.OK:
		return "ok"
	case gate.Fail:
		return "FAIL"
	case gate.Inconclusive:
		return "INCONCLUSIVE (tool unavailable)"
	default:
		return string(o)
	}
}

func headLines(out string, n int) []gate.Field {
	var fs []gate.Field
	for _, ln := range strings.Split(out, "\n") {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		fs = append(fs, gate.Field{Key: "line", State: strings.TrimRight(ln, "\r")})
		if len(fs) >= n {
			break
		}
	}
	return fs
}

func countLines(c gate.Cell) int {
	n := 0
	for _, f := range c.Fields {
		if f.Key == "line" {
			n++
		}
	}
	return n
}

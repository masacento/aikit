// Package canary turns "the analyzer reported nothing" into a decidable question. A tool that
// could not run — a missing binary, a cross-GOOS `go run` that built a linter it cannot exec,
// a wrong config path, a skipped directory — emits zero findings, which is byte-identical to a
// real pass. So before a lint result is trusted, the analyzer is pointed at a checked-in
// fixture that contains a deliberate violation of exactly what it checks; only if that
// violation is flagged does a clean result on the real code mean anything. If the canary does
// not fire, the caller reports CANNOT-EVALUATE and names the analyzer, never "clean".
//
// The fixture violations live under the `canaryfixture` build tag (see fixtures/), invisible
// to every normal build/test/lint; the canary invocation adds -tags canaryfixture.
package canary

import "strings"

// FixturesDir is the fixtures package, relative to the repo root. The canary invocation runs
// golangci-lint here with -tags canaryfixture.
const FixturesDir = "tools/canary/fixtures"

// VulnFixtureDir is the separate-module fixture (its own go.mod) the govulncheck canary scans;
// it reaches a symbol carrying CanaryAdvisory. tools/vulncheck skips it, so the deliberately
// vulnerable dependency never enters the shipped scan.
const VulnFixtureDir = "tools/canary/vulnfixture"

// CanaryAdvisory is the stable advisory the vulnfixture reaches (golang.org/x/text v0.3.5
// language.Parse). If it is ever withdrawn the canary stops firing → cannot-evaluate, which is
// the safe direction.
const CanaryAdvisory = "GO-2021-0113"

// Result is the outcome of a canary. Fired=true means the analyzer flagged its fixture and
// so is really running; a clean result elsewhere can be trusted. Fired=false is
// CANNOT-EVALUATE, with Reason naming why (no output, or output that lacks the fixture hit).
type Result struct {
	Fired  bool
	Reason string
}

// CheckGolangci inspects the output of a golangci-lint run that the CALLER performed against
// the fixtures dir with -tags canaryfixture — using the SAME invocation mechanism it uses for
// its real lint, so that whatever would stop the real run (exec-format-error, missing binary)
// stops this one too. It fires only if errcheck flagged the fixture's unchecked os.Remove.
func CheckGolangci(out string) Result {
	// The fixture is fixtures/errcheck_canary.go with a bare `os.Remove(...)`; errcheck names
	// both the file and the function on the finding line.
	if strings.Contains(out, "errcheck_canary.go") && strings.Contains(out, "os.Remove") {
		return Result{Fired: true}
	}
	reason := "golangci-lint did not flag the errcheck canary"
	if strings.TrimSpace(out) == "" {
		reason += " (no output — the linter produced nothing, e.g. it could not build/exec here)"
	} else if fl := firstLine(out); fl != "" {
		reason += " (first output line: " + fl + ")"
	}
	return Result{Fired: false, Reason: reason}
}

// CheckGovulncheck inspects the output of a govulncheck run the CALLER performed in the
// vulnfixture module, using the SAME binary and environment as the shipped scan. It fires only
// if the known advisory is reported. govulncheck that could not run, has no vuln DB, or scanned
// an empty/unreached graph reports nothing here — the tool's own "could not scan" catches an
// ERRORING scan, not a scan that examined nothing and exited 0 with "No vulnerabilities found".
func CheckGovulncheck(out string) Result {
	if strings.Contains(out, CanaryAdvisory) {
		return Result{Fired: true}
	}
	reason := "govulncheck did not report the canary advisory " + CanaryAdvisory
	if strings.TrimSpace(out) == "" {
		reason += " (no output — the scanner produced nothing)"
	} else if strings.Contains(out, "No vulnerabilities found") {
		reason += " (got \"No vulnerabilities found\" — the scanner examined nothing reachable)"
	} else if fl := firstLine(out); fl != "" {
		reason += " (first output line: " + fl + ")"
	}
	return Result{Fired: false, Reason: reason}
}

// ApidiffFixtureOld and ApidiffFixtureNew are the two package import paths whose exported
// surfaces differ incompatibly (F's parameter type). The apidiff canary writes old's API and
// compares new against it; apidiff MUST report the break.
const (
	ApidiffFixtureOld = "github.com/townsendmerino/aikit/tools/canary/apidifffixture/old"
	ApidiffFixtureNew = "github.com/townsendmerino/aikit/tools/canary/apidifffixture/new"
)

// CheckApidiff inspects the output of an apidiff comparison the CALLER ran between the two
// fixtures, using the SAME `go run apidiff` invocation as the real check. It fires only if the
// known incompatible change is reported. An apidiff that could not build/exec, or that compared
// an absent/empty baseline (the shape that reports "no incompatible changes" over nothing),
// does not report it → cannot-evaluate.
func CheckApidiff(out string) Result {
	if strings.Contains(out, "F: changed") {
		return Result{Fired: true}
	}
	reason := "apidiff did not report the canary's known incompatible change"
	if strings.TrimSpace(out) == "" {
		reason += " (no output — apidiff compared nothing, e.g. it could not build/exec or had no baseline)"
	} else if fl := firstLine(out); fl != "" {
		reason += " (first output line: " + fl + ")"
	}
	return Result{Fired: false, Reason: reason}
}

func firstLine(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			return strings.TrimSpace(ln)
		}
	}
	return ""
}

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

func firstLine(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			return strings.TrimSpace(ln)
		}
	}
	return ""
}

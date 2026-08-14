package canary

import (
	"strings"
	"testing"
)

func TestCheckGolangci_firesOnFixtureHit(t *testing.T) {
	out := "tools/canary/fixtures/errcheck_canary.go:16:2: Error return value of `os.Remove` is not checked (errcheck)\n"
	if r := CheckGolangci(out); !r.Fired {
		t.Fatalf("expected Fired on a real fixture hit; got %+v", r)
	}
}

func TestCheckGolangci_cannotEvaluateOnEmpty(t *testing.T) {
	// The exact shape of both incidents: the linter emitted nothing (could not exec).
	r := CheckGolangci("")
	if r.Fired {
		t.Fatal("empty output must be cannot-evaluate, not fired")
	}
	if !strings.Contains(r.Reason, "no output") {
		t.Errorf("reason should name the empty output; got %q", r.Reason)
	}
}

func TestCheckGolangci_cannotEvaluateWhenFixtureAbsent(t *testing.T) {
	// The linter ran and reported something, but NOT the canary — it did not scan the fixture
	// (skipped dir, wrong target). Still cannot-evaluate.
	out := "some/other/file.go:1:1: Error return value of `x.Y` is not checked (errcheck)\n"
	if r := CheckGolangci(out); r.Fired {
		t.Fatal("a report that misses the fixture must not count as fired")
	}
}

func TestCheckGovulncheck_firesOnAdvisory(t *testing.T) {
	out := "Vulnerability #1: " + CanaryAdvisory + "\n      #1: vuln.go:18:23: vulnfixture.CanaryVuln calls language.Parse\n"
	if r := CheckGovulncheck(out); !r.Fired {
		t.Fatalf("expected Fired on the advisory; got %+v", r)
	}
}

func TestCheckGovulncheck_cannotEvaluateOnNoVulns(t *testing.T) {
	// The scanner ran and exited 0 with "No vulnerabilities found" — but over the fixture that
	// means it reached nothing. This is the case per-module UNSCANNED=FAIL cannot catch.
	r := CheckGovulncheck("No vulnerabilities found.\n")
	if r.Fired {
		t.Fatal("\"No vulnerabilities found\" on the vuln fixture must be cannot-evaluate")
	}
	if !strings.Contains(r.Reason, "examined nothing") {
		t.Errorf("reason should name the empty scan; got %q", r.Reason)
	}
}

func TestCheckGovulncheck_cannotEvaluateOnEmpty(t *testing.T) {
	if r := CheckGovulncheck(""); r.Fired {
		t.Fatal("empty output must be cannot-evaluate")
	}
}

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

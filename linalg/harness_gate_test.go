package linalg

import (
	"os"
	"testing"
)

// harnessOnly gates a MEASUREMENT HARNESS — a Test that exists to produce numbers on a
// settled box, not to assert correctness. The tests in this class drive testing.Benchmark
// internally with multi-repeat sweeps, so their runtime is self-calibrating: each inner
// benchmark scales its iteration count until it has spent ~1s of WALL CLOCK. On a quiet
// machine that is bounded and useful; on a shared, virtualized CI runner it is not bounded
// by anything the code controls, and a contended runner stretches the same work by 10x.
//
// That is not hypothetical. TestW4A8IssueWidthProbe ran the amd64 linalg package in 56.6s
// and 55.1s on two consecutive CI runs and then hit `go test`'s 600s package timeout on the
// third — identical code, no change but the runner. Before this gate there were thirteen such
// tests and exactly ONE of them (TestFMAIssueProbe) was excluded, via a `-skip` regex in one
// CI job; the other three jobs that run this package (root -race, the arm64 simd leg,
// preflight) had no exclusion at all and were sitting at 228-276s of the same 600s budget.
//
// Gating at the source rather than in the workflow is deliberate: a `-skip` regex protects
// only the job it is written in and has to be remembered again for every new harness, and it
// is invisible to tools/skips — the census exists so a skip lands in a counted, named bucket
// instead of folding into a bare green. "set AIKIT_HARNESS=1" classifies as "env opt-in"
// there (skips.go's reasonRules), alongside AIKIT_GPU_BENCH, with no change needed to the
// classifier.
//
// Compilation is still gated everywhere: `go vet ./...` and every job's build step type-check
// these files whether or not the tests run, which is the failure that actually bit at 9724289
// (a harness referencing arm64-only symbols with no build tag). Skipping the RUN loses the
// numbers, which were never trustworthy on CI hardware anyway; it does not lose the compile.
//
// To run them: AIKIT_HARNESS=1 go test ./linalg/ -run TestW4A8OpsPerByte -v
func harnessOnly(t *testing.T) {
	t.Helper()
	if os.Getenv("AIKIT_HARNESS") != "1" {
		t.Skip("measurement harness — set AIKIT_HARNESS=1 to run it on a settled box")
	}
}

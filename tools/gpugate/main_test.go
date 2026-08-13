package main

import (
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
)

// Synthetic `go test -v` fragments — the shapes the classifiers must read.
const (
	outPass = "=== RUN   TestX\n--- PASS: TestX (0.01s)\nPASS\nok  \tpkg\t0.02s\n"
	outSkip = "=== RUN   TestX\n--- SKIP: TestX (0.00s)\n    x_test.go:10: NVRTC not found\nPASS\nok  \tpkg\t0.01s\n"
	outNone = "testing: warning: no tests to run\nPASS\nok  \tpkg\t0.00s\n"
	outFail = "=== RUN   TestX\n    x_test.go:42: bytes differ\n--- FAIL: TestX (0.01s)\nFAIL\n"
)

// A skip is never a pass — but the two kernel classifiers resolve it differently, which is
// the whole point of separating them: fma assertions have no legitimate skip (FAIL), while
// ptx-repro's skip means NVRTC was unfindable (INCONCLUSIVE — could not judge). This test
// pins both, deterministically, on any box — it needs no NVRTC and no Linux runner, so the
// NVRTC→INCONCLUSIVE contract is demonstrated here rather than only on the NVIDIA leg.
func TestKernelClassifiers(t *testing.T) {
	cases := []struct {
		name    string
		got     gate.Outcome
		want    gate.Outcome
		comment string
	}{
		{"fma pass", classifyKernel("fma-lint", "TestX", outPass, 0).Outcome, gate.OK, "a real pass is OK"},
		{"fma fail", classifyKernel("fma-lint", "TestX", outFail, 1).Outcome, gate.Fail, "a failing test is FAIL"},
		{"fma no-match", classifyKernel("fma-lint", "TestX", outNone, 0).Outcome, gate.Fail, "a vanished assertion is FAIL"},
		{"fma skip", classifyKernel("fma-lint", "TestX", outSkip, 0).Outcome, gate.Fail, "a skip is not a pass — FAIL for fma"},
		{"ptx pass", classifyPTX(outPass, 0).Outcome, gate.OK, "reproducibility verified is OK"},
		{"ptx fail", classifyPTX(outFail, 1).Outcome, gate.Fail, "byte drift is FAIL"},
		{"ptx no-match", classifyPTX(outNone, 0).Outcome, gate.Fail, "a vanished assertion is FAIL"},
		{"ptx skip", classifyPTX(outSkip, 0).Outcome, gate.Inconclusive, "NVRTC unfindable is INCONCLUSIVE, never PASS"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %s, want %s (%s)", c.name, c.got, c.want, c.comment)
		}
	}
}

// Package fixtures holds deliberate analyzer violations used as CANARIES. A lint/analyzer
// run is trusted only after it flags the fixture for exactly what that analyzer checks; if
// the canary is not flagged, the tool did not actually run and the run's result is
// cannot-evaluate, not clean.
//
// Why this exists: on 2026-08-13 the same mechanism produced a false "clean" twice, hours
// apart — `go run golangci-lint` under a mismatched GOOS cross-compiled the linter itself,
// which then could not exec (exec-format-error) and emitted nothing, and a `go run` errcheck
// census did the same. A "0 findings" from a tool that never ran is indistinguishable from a
// real pass unless something is KNOWN to be flaggable. The canary is that known thing.
//
// The violations live behind the `canaryfixture` build tag, so they are invisible to every
// normal build, test, and lint (this package is empty without the tag — doc.go only). The
// canary run adds `-tags canaryfixture` to reveal them. NEVER "fix" a fixture: the violation
// firing is the whole point, the same rule as verifying a forcing mechanism fires before
// trusting its null.
package fixtures

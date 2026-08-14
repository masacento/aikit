# Analyzers whose "clean" a gate trusts — and their canaries

A tool that examines nothing and exits 0 is byte-identical to a tool that examined everything
and found nothing. On 2026-08-13 that ambiguity produced a false "clean" twice in one day (a
`go run golangci-lint` cross-compiled the linter for the wrong platform → exec-format-error →
no findings → looked clean). Building a guard per incident is how the second happened. This
list is the alternative: every analyzer whose absence-of-findings a gate treats as a pass,
what it does when it runs but examines nothing, and whether that is distinguishable from clean.

The rule: where "runs but examines nothing" is indistinguishable from clean, the gate runs a
CANARY first — a checked-in fixture with a deliberate violation of exactly what the analyzer
checks, using the SAME invocation as the real run. If the canary is not flagged, the result is
**CANNOT-EVALUATE**, naming the analyzer — never clean. Canary logic lives in `tools/canary`.

## The census

| analyzer | gate(s) | the trusted null | runs-but-examines-nothing | distinguishable from clean? | guard |
|---|---|---|---|---|---|
| **golangci-lint** (errcheck, staticcheck, govet, ineffassign, unused) | gpugate, preflight, releasegate | "0 issues" | missing binary, cross-GOOS `go run` exec-format-error, no config, skipped dir → 0 findings | **No** | **CANARY** — `fixtures/errcheck_canary.go` (a bare `os.Remove`), run with `-tags canaryfixture` |
| **govulncheck** | vulncheck | "No vulnerabilities found" | scans a graph it never reaches (empty/unbuilt) and exits 0 with "No vulnerabilities found" | **No** — the tool's `UNSCANNED=FAIL` catches an ERRORING scan, not a scan that examined nothing | **CANARY** — `vulnfixture/` (separate module, pins x/text v0.3.5, reaches GO-2021-0113) |
| **apidiff** | releasegate | "no incompatible changes" | no previous tag → skip→OK; absent/empty baseline surface → empty diff→OK | **No** | **CANARY** — `apidifffixture/{old,new}` (F's signature changes int→string) |
| **`go list -deps`** (core-deps invariant) | releasegate | "no unexpected external deps" | a failed `go list` emits little → filters to no deps → OK | **No** (rc was ignored) | **rc-guard** — non-zero `go list` is now CANNOT-EVALUATE (cheaper than a canary; the exit code is the positive control) |
| **go vet** | gpugate, preflight | no findings | no packages → no findings | low | none — core `go` subcommand: no separate binary to fail-to-exec, no baseline; every call is `go vet ./...` (never a path list — verified) |
| **gofmt -l** | gpugate, preflight | empty output | no files → empty | low | none — core tool; every call is `gofmt -l .` (recursive from root, never a path list — verified). Residual: stderr is discarded, so a failed gofmt reads empty→clean; a mis-formatted fixture canary is possible if this ever bites |
| **go build** (cgo-free) | gpugate, preflight | rc == 0 | empty package builds trivially | n/a — a build is not a null; failures are loud (rc checked). Every call is `go build ./...` (never a path list — verified) | none needed |
| **kernel tests** (fma-lint, fma-histogram, ptx-repro) | gpugate | test passes | test skips (examines nothing) | **Yes** — already guarded: a skip → FAIL / INCONCLUSIVE ("a skip is not a pass") | pre-existing skip guard |
| **`go test`** (root suite) | preflight, CI | package `ok` | every test skips → `ok` | partially | AK3 skip census surfaces the skip tally by reason next to the verdict |

## What is warranted, and what was built

Three analyzers had the indistinguishable false-clean shape and now carry canaries:
golangci-lint (the one that bit us), **govulncheck**, and **apidiff**. `go list -deps` had the
shape but its exit code is a sufficient positive control, so it got an rc-guard rather than a
canary. go vet / gofmt / go build are core `go` subcommands with no separate binary or baseline
to be silently empty. Their one examines-nothing path would be an explicit path list that
stops matching (goinfer's gofmt had exactly this — a hand-maintained `SCAN_DIRS`); every aikit
invocation instead takes `./...` (vet, build) or `.` (gofmt, recursive from root), verified by
grep, so there is no list to go stale. That — not merely "low risk" — is why they need no
guard. The kernel tests and
the root `go test` already distinguish "examined nothing" (a skip) from a pass.

## The invariant for adding an analyzer

Before a new gate trusts a new analyzer's clean: ask the middle column. If "runs but examines
nothing" is indistinguishable from clean, it needs a canary (a fixture it must flag) or a
built-in positive control (an exit code, a skip-is-not-a-pass rule) BEFORE the null is
believed. Never fix a fixture — its firing is the precondition for the run meaning anything.

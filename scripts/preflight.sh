#!/usr/bin/env bash
# preflight.sh — run what CI's root-module job runs, locally, before pushing.
#
# WHY THIS EXISTS. On 2026-08-12 a revert left an assembly kernel with no caller.
# `golangci-lint` reports that in about three seconds; I found out from CI, after
# pushing, twice. The linter was already installed on the box at the exact version CI
# pins. Using CI as a first linter rather than a last check costs a full round trip per
# mistake, and on that day it also collided with two unrelated CDN failures in the same
# step, which made a real finding look like more flakiness.
#
# So this is deliberately the same checks in the same order as .github/workflows/ci.yml's
# `core` job, minus the parts that need the network or a GPU:
#
#   gofmt -> build -> vet -> golangci-lint -> cgo-free build -> go test
#
# It does NOT run -race (CI does; it roughly triples the test time) and does not touch
# the nine gpu modules — those have `go run -C tools ./gpugate` and `./gpudevice`.
#
# Usage:
#   scripts/preflight.sh              # everything below
#   scripts/preflight.sh --fast       # skip the test run (fmt/vet/lint/build only)
#
# Install as a pre-push hook (not versioned, so it is opt-in per clone):
#   ln -sf ../../scripts/preflight.sh .git/hooks/pre-push
set -u

# Resolve the repo root via git, NOT relative to this script. As a pre-push hook this
# runs from a symlink at .git/hooks/pre-push, where "$(dirname "$0")/.." is `.git` — so a
# BASH_SOURCE-relative root made the first hook invocation lint an empty directory and
# report "no go files to analyze" as three failures. Caught by the hook, on itself.
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$ROOT" ] && ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# The gpu modules carry `replace` directives and a developer may have a go.work; neither
# should decide what this reports. Same reasoning as scripts/gpu_gate.sh.
export GOWORK=off

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

FAILED=0
step() { # step <name> <command...>
	printf '  %-22s ' "$1"
	shift
	out="$("$@" 2>&1)"
	if [ $? -ne 0 ] || [ -n "${STEP_EXPECT_EMPTY:-}" ] && [ -n "$out" ]; then
		echo "FAIL"
		printf '%s\n' "$out" | head -12 | sed 's/^/      /'
		FAILED=$((FAILED + 1))
		return 1
	fi
	echo "ok"
	return 0
}

echo "aikit preflight — $(git rev-parse --short HEAD 2>/dev/null || echo '?') — root module"
echo

# gofmt reports by printing filenames and still exits 0, so emptiness is the check.
printf '  %-22s ' gofmt
unformatted="$(gofmt -l . 2>/dev/null | grep -v '^\.venv/' || true)"
if [ -n "$unformatted" ]; then
	echo "FAIL"
	printf '%s\n' "$unformatted" | sed 's/^/      /'
	FAILED=$((FAILED + 1))
else
	echo "ok"
fi

step build go build ./...
step vet go vet ./...

# golangci-lint: prefer whatever is on PATH, else the usual GOPATH location. If neither
# exists, say so LOUDLY rather than skipping quietly — a missing linter reporting "ok"
# is the failure mode this whole script is a reaction to.
printf '  %-22s ' golangci-lint
GCL="$(command -v golangci-lint 2>/dev/null || true)"
[ -z "$GCL" ] && [ -x "$HOME/go/bin/golangci-lint" ] && GCL="$HOME/go/bin/golangci-lint"
if [ -z "$GCL" ]; then
	echo "MISSING — install it, do not skip it"
	echo "      go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.11.4"
	echo "      (CI pins v2.11.4; a different version reports differently)"
	FAILED=$((FAILED + 1))
else
	out="$("$GCL" run 2>&1)"
	if [ $? -ne 0 ]; then
		echo "FAIL"
		printf '%s\n' "$out" | head -12 | sed 's/^/      /'
		FAILED=$((FAILED + 1))
	else
		echo "ok ($("$GCL" version 2>&1 | head -1 | grep -o 'version [^ ]*' || echo 'version ?'))"
	fi
fi

printf '  %-22s ' "build (CGO_ENABLED=0)"
if CGO_ENABLED=0 go build ./... >/dev/null 2>&1; then echo "ok"; else
	echo "FAIL"
	FAILED=$((FAILED + 1))
fi

if [ "$FAST" -eq 0 ]; then
	printf '  %-22s ' "go test"
	out="$(go test ./... 2>&1)"
	if [ $? -ne 0 ]; then
		echo "FAIL"
		printf '%s\n' "$out" | grep -E '^(---|FAIL|panic)' | head -8 | sed 's/^/      /'
		FAILED=$((FAILED + 1))
	else
		echo "ok"
	fi
else
	echo "  go test                skipped (--fast); CI runs it with -race"
fi

echo
if [ "$FAILED" -gt 0 ]; then
	echo "PREFLIGHT: $FAILED check(s) failed — fix before pushing."
	exit 1
fi
echo "PREFLIGHT: clean. CI still runs -race, the gpu jobs and vulncheck."
exit 0

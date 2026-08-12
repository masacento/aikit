#!/usr/bin/env bash
# gpu_gate.sh — the pre-tag gate for the `gpu/` module tree. Run it, paste the verdict
# into the tag message, then tag.
#
# WHY THIS IS THE GATE AND NOT CI. Until 2026-08 no CI job touched any of the nine
# gpu/* modules; the `gpu-kernels` job added alongside this script is the first, and it
# covers only the checks that need no GPU. The device tests (Metal parity, CUDA numerics)
# still cannot run in CI — Metal is darwin-only, CUDA needs an NVIDIA card, and no runner
# has either. So the release ritual is the enforcement. Branch protection was considered
# and declined: required checks effectively force PR-only merges, which is real friction
# against a threat model aikit does not have (no outside contributors), and it relocates
# the actual risk — a maintainer not running the check — rather than solving it. A gate
# that is part of tagging is a step someone can visibly fail to complete.
#
# HONESTY RULES, inherited from goinfer/scripts/gpu_gate.sh because they were learned the
# hard way there:
#   - A SKIP IS NOT A PASS. `go test` prints "ok" for a package whose tests all skipped.
#     The PTX reproducibility check therefore runs with AIKIT_REQUIRE_PTX_REPRO=1, which
#     turns "NVRTC not installed" from a silent skip into a failure. Green here means
#     tested.
#   - ...EXCEPT WHERE THE TOOL CANNOT EXIST, WHICH IS NOT THE SAME THING. NVIDIA ships no
#     NVRTC for Apple Silicon — there is no darwin-arm64 wheel, only a 0.0.1.dev5
#     placeholder — so on darwin that rule turned "this platform cannot run this check"
#     into FAIL, and the gate could never be green on a Mac. "Absent because nobody
#     installed it" and "absent because it does not exist for this OS" deserve different
#     answers: the first is a failure, the second is n/a. Conflating them either blocks
#     the release or, worse, invites someone to delete the strictness that makes the first
#     case honest.
#   - GROUPS ARE DECLARED UP FRONT AND RECONCILED AT THE END. A tally computed only from
#     what emitted can never notice what did not emit: a group that dies mid-block simply
#     vanishes and the gate reports PASS having tested nothing. Any declared group that
#     produces no verdict is itself a FAIL.
#
# What it does NOT do: run the device tests. Those need a GPU and are the hand-run half of
# the ritual in RELEASING.md. This script is the half that is mechanical, so that the
# hand-run half is the only thing depending on a human remembering.
#
# Usage:  scripts/gpu_gate.sh
# Env:    NVRTC_LIB / CUDA_INC   override NVRTC discovery (see gpu/build_ptx.sh)
set -u

# GOWORK=off IS LOAD-BEARING, not a workaround.
#
# go.work is gitignored (.gitignore) and therefore per-machine: this box has none, the
# Mac has one listing only the *metal modules. With the workspace active, `go list ./...`
# inside a module the workspace omits fails with "directory prefix . does not contain
# modules listed in go.work" — a workspace error raised BEFORE build tags are considered,
# so a not-applicable module is indistinguishable from a broken one. That is how the Mac
# got FAIL for the four *cuda modules on 2026-08-12.
#
# Adding them to go.work would fix that symptom and make this gate WORSE. A workspace
# deliberately overrides module resolution with local directories, and what this gate
# exists to check is each module AS IT WILL BE PUBLISHED. A workspace would have hidden
# the `require aikit/gpu v0.0.0` defect completely — in-repo everything built, while no
# consumer could resolve it at all (RELEASING.md, "Backend submodules"). A release gate
# that reads the developer's local overrides is measuring the wrong thing.
#
# So: resolve every module standalone, through its own go.mod, on every machine.
export GOWORK=off

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
DIRTY=""
[ -n "$(git status --porcelain 2>/dev/null)" ] && DIRTY=" +dirty"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BOX="$(uname -s)/$(uname -m)"

# Declared groups (see the reconciliation rule above). Every one must emit a verdict.
#
# SPACE-DELIMITED STRINGS, NOT ARRAYS, AND THAT IS DELIBERATE. This used `declare -A`,
# which is bash 4.0+; macOS ships bash 3.2.57 (the last GPLv2 release) and has no other
# bash by default. The gate therefore died on `declare: -A: invalid option` before
# emitting a single verdict — on a Mac, which is the machine the RELEASING ritual most
# needs it on. It went unnoticed from the day it was written because it had only ever run
# on Linux. Keep every construct here POSIX-portable; `sh scripts/gpu_gate.sh` should
# work, and is the cheap way to check.
EXPECTED="build fmt-vet lint fma-lint fma-histogram ptx-repro"
NEXPECTED="$(set -- $EXPECTED; echo $#)"
SEEN=" "
NA=""
NNA=0
FAILED=0
OS="$(uname -s)"

verdict() { # verdict <group> <PASS|FAIL> <detail>
	SEEN="$SEEN$1 "
	printf '  %-14s %-4s %s\n' "$1" "$2" "$3"
	[ "$2" = "FAIL" ] && FAILED=$((FAILED + 1))
	return 0
}

notapplicable() { # notapplicable <group> <why>
	SEEN="$SEEN$1 "
	NA="$NA $1"
	NNA=$((NNA + 1))
	printf '  %-14s %-4s %s\n' "$1" "n/a" "$2"
	return 0
}

echo "aikit gpu gate — $COMMIT$DIRTY — $BOX — $DATE"
echo

# (1) Every gpu module builds cgo-free. Nine separate modules; a break in one is invisible
#     from the others, and none of them is in the root module's `go build ./...`.
mods="$(find gpu -name go.mod -exec dirname {} \; | sort)"
nmod=0
bad=""
for m in $mods; do
	nmod=$((nmod + 1))
	if ! (cd "$m" && CGO_ENABLED=0 go build ./... >/dev/null 2>&1); then bad="$bad $m"; fi
done
if [ -n "$bad" ]; then
	verdict build FAIL "cgo-free build failed:$bad"
else
	verdict build PASS "$nmod modules build with CGO_ENABLED=0"
fi

# (2b) golangci-lint across ALL NINE modules. The root module's CI job lints only the
#      root module — `./...` stops at module boundaries — so before this the nine gpu
#      modules had gofmt and vet and nothing that finds an unused function. That is
#      exactly how a reverted kernel sat with no caller in linalg on 2026-08-12; had it
#      been in gpu/anncuda instead, nothing would have said a word.
#
#      A MISSING LINTER IS A FAILURE, NOT A SKIP. Unlike NVRTC on darwin (which cannot
#      exist and is reported n/a), golangci-lint installs everywhere, so "not installed"
#      means unchecked and must read as red.
lint_gpu_modules() {
	gcl="$(command -v golangci-lint 2>/dev/null || true)"
	[ -z "$gcl" ] && [ -x "$HOME/go/bin/golangci-lint" ] && gcl="$HOME/go/bin/golangci-lint"
	if [ -z "$gcl" ]; then
		verdict lint FAIL "golangci-lint not installed — install it, do not skip it (CI pins v2.11.4)"
		return
	fi
	bad=""
	n=0
	for m in $mods; do
		# Modules with no buildable packages on this OS have nothing to lint.
		pkgs="$(cd "$m" && CGO_ENABLED=0 go list ./... 2>/dev/null)"
		[ $? -eq 0 ] && [ -z "$pkgs" ] && continue
		n=$((n + 1))
		if ! (cd "$m" && CGO_ENABLED=0 "$gcl" run >/dev/null 2>&1); then bad="$bad $m"; fi
	done
	if [ -n "$bad" ]; then
		verdict lint FAIL "golangci-lint issues in:$bad"
	else
		verdict lint PASS "$n applicable modules clean"
	fi
}

# (2) gofmt + vet on the gpu module itself.
unformatted="$(cd gpu && gofmt -l . 2>/dev/null)"
vetout="$(cd gpu && CGO_ENABLED=0 go vet ./... 2>&1)"
if [ -n "$unformatted" ] || [ -n "$vetout" ]; then
	verdict fmt-vet FAIL "gofmt:[$unformatted] vet:[$(echo "$vetout" | head -1)]"
else
	verdict fmt-vet PASS "gofmt clean, go vet clean"
fi

# (3)(4)(5) The three kernel assertions. Run separately so one failure cannot mask another
#     and so each reports its own verdict — the reconciliation rule needs per-group output.
run_test() { # run_test <group> <-run pattern> <extra env assignments…>
	local group="$1" pattern="$2"
	shift 2
	local out rc
	out="$(cd gpu && env ${1+"$@"} CGO_ENABLED=0 go test -count=1 -run "$pattern" -v ./... 2>&1)"
	rc=$?
	local nrun npass nskip
	nrun="$(printf '%s\n' "$out" | grep -c '^=== RUN')"
	npass="$(printf '%s\n' "$out" | grep -c '^--- PASS\|^    --- PASS')"
	nskip="$(printf '%s\n' "$out" | grep -c '^--- SKIP\|^    --- SKIP')"
	if [ "$rc" -ne 0 ]; then
		verdict "$group" FAIL "$(printf '%s\n' "$out" | grep -m1 -E '^\s+.*_test\.go:[0-9]+' | sed 's/^[[:space:]]*//' | cut -c1-90)"
	elif [ "$nrun" -eq 0 ]; then
		verdict "$group" FAIL "no tests matched '$pattern' — the assertion is gone or renamed"
	elif [ "$nskip" -gt 0 ]; then
		verdict "$group" FAIL "$nskip skipped — a skip is not a pass"
	else
		verdict "$group" PASS "$npass passed ($nrun run)"
	fi
}

run_test fma-lint      'TestKernelFMALint'
run_test fma-histogram 'TestGemvQuantFMAHistogram'
lint_gpu_modules

# NVRTC required: without the flag this test skips when NVRTC is absent, and a skipping
# gate check is a passing one.
# NVRTC exists for Linux and Windows, never for Apple Silicon, so this group is not
# merely unconfigured on darwin — it is unrunnable there. Reported as n/a rather than
# FAIL, and excluded from the tally, so that a Mac's verdict says what it actually
# covered instead of blocking on a check that machine can never perform.
if [ "$OS" = "Darwin" ]; then
	notapplicable ptx-repro "no NVRTC for darwin (NVIDIA ships none for Apple Silicon)"
else
	run_test ptx-repro     'TestPTXReproducible'       AIKIT_REQUIRE_PTX_REPRO=1
fi

# Reconcile declared groups against emitted verdicts (goinfer audit G-01).
missing=""
for g in $EXPECTED; do
	case "$SEEN" in
	*" $g "*) ;;
	*) missing="$missing $g" ;;
	esac
done
if [ -n "$missing" ]; then
	printf '  %-14s %-4s %s\n' reconcile FAIL "declared groups produced no verdict:$missing"
	FAILED=$((FAILED + 1))
fi

NAPPLIC=$((NEXPECTED - NNA))

echo
if [ -n "$NA" ]; then
	echo "NOT APPLICABLE on this platform —$NA"
	echo "      This verdict does not cover them; a box that can run them must."
	echo
fi
if [ "$FAILED" -gt 0 ]; then
	echo "VERDICT: FAIL — $FAILED of $NAPPLIC applicable groups red at $COMMIT$DIRTY ($BOX, $DATE)"
	exit 1
fi
# Every group is green. A dirty tree is not a failure of the CHECKS — it is a failure of
# PROVENANCE: the verdict names a commit, and an uncommitted edit means the verdict does
# not describe what that commit contains. Distinguish the two rather than printing
# "FAIL — 0 groups red", which reads as a bug in the gate.
if [ -n "$DIRTY" ]; then
	echo "VERDICT: INCONCLUSIVE — $NAPPLIC/$NAPPLIC applicable groups green, but the working tree is DIRTY."
	echo "The verdict names $COMMIT and the tree is not $COMMIT. Commit, then re-run before tagging."
	exit 1
fi
echo "VERDICT: PASS — $NAPPLIC/$NEXPECTED applicable groups green at $COMMIT ($BOX, $DATE)"
if [ "$NNA" -gt 0 ]; then
	echo "         $NNA of $NEXPECTED not applicable here —$NA"
fi
echo "Paste that line into the tag message. Device tests are the hand-run half — see RELEASING.md."
exit 0

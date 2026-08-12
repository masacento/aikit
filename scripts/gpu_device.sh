#!/usr/bin/env bash
# gpu_device.sh — the DEVICE half of the pre-tag ritual: run the gpu/* test suites on a
# machine that actually has the GPU, and print a verdict for the tag message.
#
# WHY THIS EXISTS. RELEASING.md used to say `cd gpu && CGO_ENABLED=0 go test ./...`, and
# `./...` STOPS AT MODULE BOUNDARIES — so that command tested one of the nine gpu modules
# and silently skipped the other eight. Found on 2026-08-12 by the Mac, running the step
# as written and noticing gpu/annmetal had not been covered.
#
# The eight it missed are the same eight that have never been tagged (RELEASING.md,
# "Backend submodules"). That is not a coincidence: both are what you get when a tree of
# sibling modules is treated as one module by habit. A ritual whose command under-tests by
# 8/9 is worse than no ritual, because it produces the feeling of having checked.
#
# The mechanical, no-GPU half is scripts/gpu_gate.sh. This is the half that needs hardware,
# so it reports what it actually ran on and refuses to imply more.
#
# HONESTY RULES, same as the gate:
#   - A SKIP IS NOT A PASS. `go test` prints "ok" for a package whose tests all skipped,
#     and every device test here skips cleanly when its hardware is absent — which is
#     exactly the failure mode this script must not launder. It therefore reports the
#     BACKEND it saw (metal / cuda / none) and marks modules whose tests all skipped as
#     SKIPPED, never as passed.
#   - NOT APPLICABLE IS NOT A FAILURE, AND NOT A PASS EITHER. The four *metal modules have
#     no buildable Go files on Linux and the four *cuda modules none on darwin; `go test`
#     exits 1 with "matched no packages" for those, which is neither red nor green. They
#     are reported as n/a and excluded from the tally, which is what makes the arithmetic
#     honest: NO SINGLE MACHINE CAN COVER ALL NINE. A gpu/vX.Y.Z tag needs a verdict line
#     from a Mac AND one from an NVIDIA box, and this script's output shows at a glance
#     which half you are holding.
#   - The verdict names a commit, so a dirty tree is INCONCLUSIVE rather than PASS.
#
# Usage:  scripts/gpu_device.sh
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
OS="$(uname -s)"
BOX="$(uname -n) $OS/$(uname -m)"

# Which backend can actually run here. Reported in the verdict so a green line cannot be
# mistaken for coverage of the other platform.
BACKEND="none"
if [ "$OS" = "Darwin" ]; then
	BACKEND="metal"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
	BACKEND="cuda"
fi

echo "aikit gpu device tests — $COMMIT$DIRTY — $BOX — backend:$BACKEND — $DATE"
echo

mods="$(find gpu -name go.mod -exec dirname {} \; | sort)"
nmod=0
napplic=0
failed=""
skipped=""
nap=""
for m in $mods; do
	nmod=$((nmod + 1))
	# -v so skips are visible: a module whose every test skipped still prints "ok",
	# which is the single most important thing this script must not launder.
	# Applicability is decided by `go list`, NOT by matching text in `go test` output.
	# An earlier version grepped for "matched no packages" and got it wrong on darwin,
	# where the same situation surfaced as `[setup failed]` — parsing English from a
	# toolchain is a bug waiting for a different platform to find it.
	#
	# `go list ./...` distinguishes the two cases that matter:
	#   exit 0, no output  -> nothing buildable here. The other platform's module: n/a.
	#   exit != 0          -> the packages exist but could not be LOADED (a module or
	#                         go.sum problem). That is a real failure and is reported as
	#                         one, with the loader's own message, rather than excused as
	#                         a platform difference.
	pkgs="$(cd "$m" && CGO_ENABLED=0 go list ./... 2>/dev/null)"
	lrc=$?
	if [ $lrc -eq 0 ] && [ -z "$pkgs" ]; then
		nap="$nap $m"
		printf '  %-24s n/a  (no buildable packages on this platform)\n' "$m"
		continue
	fi
	if [ $lrc -ne 0 ]; then
		napplic=$((napplic + 1))
		failed="$failed $m"
		printf '  %-24s FAIL (cannot load packages)\n' "$m"
		(cd "$m" && CGO_ENABLED=0 go list ./... 2>&1 >/dev/null) | head -4 | sed 's/^/      /'
		continue
	fi
	out="$(cd "$m" && CGO_ENABLED=0 go test -v ./... 2>&1)"
	rc=$?
	napplic=$((napplic + 1))
	if [ $rc -ne 0 ]; then
		failed="$failed $m"
		printf '  %-24s FAIL\n' "$m"
		echo "$out" | grep -E '^(--- FAIL|FAIL|panic)' | head -5 | sed 's/^/      /'
		continue
	fi
	ran="$(echo "$out" | grep -c '^--- PASS')"
	skip="$(echo "$out" | grep -c '^--- SKIP')"
	if [ "$ran" -eq 0 ]; then
		skipped="$skipped $m"
		printf '  %-24s SKIPPED (0 passed, %s skipped)\n' "$m" "$skip"
	else
		printf '  %-24s ok (%s passed, %s skipped)\n' "$m" "$ran" "$skip"
	fi
done

echo
if [ -n "$failed" ]; then
	echo "VERDICT: FAIL — device tests red in:$failed (at $COMMIT$DIRTY, $BOX, backend:$BACKEND, $DATE)"
	exit 1
fi
if [ -n "$skipped" ]; then
	echo "NOTE: ran but every test skipped —$skipped"
	echo
fi
if [ -n "$nap" ]; then
	echo "NOT APPLICABLE on this platform —$nap"
	echo "      Those need a run on the other OS. This verdict does not cover them."
	echo
fi
if [ "$BACKEND" = "none" ]; then
	echo "VERDICT: INCONCLUSIVE — $napplic applicable modules ran green, but NO GPU BACKEND was detected here,"
	echo "so every device test skipped. This says the code compiles and the CPU paths agree;"
	echo "it says nothing about Metal or CUDA. Run it on the hardware before tagging."
	exit 2
fi
if [ -n "$DIRTY" ]; then
	echo "VERDICT: INCONCLUSIVE — $napplic/$napplic applicable modules green on backend:$BACKEND, but the working tree is DIRTY."
	echo "The verdict names $COMMIT and the tree is not $COMMIT. Commit, then re-run before tagging."
	exit 2
fi
echo "VERDICT: PASS — $napplic/$napplic applicable gpu modules green at $COMMIT ($BOX, backend:$BACKEND, $DATE)"
echo "         $((nmod - napplic)) of $nmod not applicable on this platform."
echo "Paste that line into the tag message. It covers ONLY backend:$BACKEND — a gpu/ tag"
echo "needs the other platform's verdict line too."

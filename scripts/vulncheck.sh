#!/usr/bin/env bash
# vulncheck.sh — the vulnerability statement for every module in this repo.
#
# WHY THIS IS A DELIVERABLE, NOT HYGIENE. aikit's pitch is a single static binary you
# scp somewhere and run offline (examples/embedded-corpus: ~1.8 MB, //go:embed's its
# own model and corpus, no network). A binary like that is deployed by someone who
# cannot patch it in place and may not revisit it for a long time. "What is in it, and
# is any of it known-vulnerable" is therefore part of the deployment claim the README
# makes — the same way the cgo-free and dependency-light invariants are, both of which
# already have gates (scripts/release-gate.sh).
#
# govulncheck is the right tool for that claim specifically because it is
# REACHABILITY-FILTERED: it reports a vulnerability only when a symbol your build
# actually reaches is affected, not merely because a vulnerable version appears in the
# module graph. A "no vulnerabilities found" from it is a statement about this binary,
# not about the dependency list. That is what makes it low-noise enough to gate on.
#
# THIRTEEN MODULES, AND SIX OF THEM COULD NOT BE SCANNED AT ALL until 2026-08-11: their
# go.mod files were stale ("updates to go.mod needed"), and four more matched zero
# packages on Linux because they are darwin-only. A scanner that silently covers half a
# repo is worse than none, because the clean report implies the other half. Hence:
#
#   - EVERY module is enumerated from `find -name go.mod`, never a list.
#   - A module that cannot be SCANNED is a FAILURE, distinct from a module that scans
#     clean. "Could not load packages" must never read as "no vulnerabilities".
#   - The four *metal modules are scanned under GOOS=darwin, since their sources are
#     entirely behind //go:build darwin and would otherwise contribute zero packages
#     and a green tick.
#
# Usage:  scripts/vulncheck.sh
# Env:    GOVULNCHECK   path to the binary (default: govulncheck on PATH, else
#                       $(go env GOPATH)/bin/govulncheck)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GVC="${GOVULNCHECK:-}"
if [ -z "$GVC" ]; then
	if command -v govulncheck >/dev/null 2>&1; then GVC="govulncheck"
	elif [ -x "$(go env GOPATH)/bin/govulncheck" ]; then GVC="$(go env GOPATH)/bin/govulncheck"
	else
		echo "vulncheck: govulncheck not found — go install golang.org/x/vuln/cmd/govulncheck@latest" >&2
		exit 1
	fi
fi

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
DIRTY=""
[ -n "$(git status --porcelain 2>/dev/null)" ] && DIRTY=" +dirty"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "aikit vulnerability scan — $COMMIT$DIRTY — $DATE"
echo "scanner: $($GVC -version 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')"
echo

VULN=0     # modules reporting a reachable vulnerability
UNSCANNED=0 # modules that could not be scanned at all
CLEAN=0

for m in $(find . -name go.mod -not -path "./.venv/*" -exec dirname {} \; | sort); do
	# darwin-only modules contribute zero packages on Linux; scanning them under the
	# host GOOS would produce a vacuous pass.
	env_pfx=()
	case "$m" in
	*metal) env_pfx=(env GOOS=darwin GOARCH=arm64) ;;
	esac

	out="$("${env_pfx[@]}" sh -c "cd '$m' && '$GVC' ./... 2>&1")"
	rc=$?

	if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q "No vulnerabilities found"; then
		printf '  %-30s CLEAN\n' "$m"
		CLEAN=$((CLEAN + 1))
	elif printf '%s' "$out" | grep -qE "Vulnerability #[0-9]+:"; then
		n="$(printf '%s' "$out" | grep -cE "Vulnerability #[0-9]+:")"
		printf '  %-30s VULNERABLE (%s reachable)\n' "$m" "$n"
		printf '%s\n' "$out" | grep -E "Vulnerability #[0-9]+:|Found in:|Fixed in:" | sed 's/^/      /'
		VULN=$((VULN + 1))
	else
		# Not clean and no findings parsed: the scan itself failed. This is the case
		# that must not be mistaken for a pass.
		printf '  %-30s UNSCANNED — %s\n' "$m" "$(printf '%s' "$out" | grep -v '^$' | head -1 | cut -c1-70)"
		UNSCANNED=$((UNSCANNED + 1))
	fi
done

total=$((CLEAN + VULN + UNSCANNED))
echo
if [ "$VULN" -eq 0 ] && [ "$UNSCANNED" -eq 0 ]; then
	echo "STATEMENT: no reachable vulnerabilities in $CLEAN/$total modules at $COMMIT$DIRTY ($DATE)"
	exit 0
fi
[ "$UNSCANNED" -gt 0 ] && echo "  $UNSCANNED module(s) could not be scanned — that is not a clean result."
[ "$VULN" -gt 0 ] && echo "  $VULN module(s) have reachable vulnerabilities."
echo "STATEMENT: INCOMPLETE — $CLEAN clean, $VULN vulnerable, $UNSCANNED unscanned, of $total at $COMMIT$DIRTY"
exit 1

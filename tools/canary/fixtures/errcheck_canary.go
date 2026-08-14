//go:build canaryfixture

package fixtures

import "os"

// CanaryErrcheck deliberately ignores a returned error. errcheck (which golangci-lint runs
// under .golangci.yml) MUST flag this line: "Error return value of `os.Remove` is not
// checked". A golangci-lint invocation that does not report it did not really run errcheck
// here — the linter is missing, could not exec (cross-GOOS build), loaded no config, or
// skipped this directory — and any "clean" it reports elsewhere means nothing.
//
// Do not add error handling. This violation is a positive control, not a bug.
func CanaryErrcheck() {
	os.Remove("/canary-does-not-exist") // INTENTIONAL unchecked error — the canary errcheck must flag this
}

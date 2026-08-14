// Package vulnfixture reaches a KNOWN-vulnerable symbol so govulncheck must report it: it
// calls golang.org/x/text v0.3.5's language.Parse, which carries advisory GO-2021-0113
// (unbounded recursion in BiDi/language parsing, fixed in v0.3.7). The govulncheck canary runs
// the scanner HERE and asserts GO-2021-0113 is reported; if it is not, govulncheck examined
// nothing (missing binary, no vuln DB, an empty graph) and its "No vulnerabilities found" on
// the shipped modules means nothing.
//
// This module is required by nothing and tools/vulncheck skips it, so the pinned vulnerable
// dependency never enters the shipped scan. Do not bump x/text here — the vulnerability is the
// point. If GO-2021-0113 is ever withdrawn the canary stops firing and reports
// cannot-evaluate, which is the safe direction.
package vulnfixture

import "golang.org/x/text/language"

// CanaryVuln reaches language.Parse, the symbol GO-2021-0113 makes govulncheck flag.
func CanaryVuln(s string) (language.Tag, error) {
	return language.Parse(s)
}

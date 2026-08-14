// A CANARY fixture module (see ../canary.go), NOT shipped and required by nothing. It pins a
// dependency carrying a known, stable advisory so the govulncheck canary has something it
// MUST report; tools/vulncheck skips this directory, so the intentionally-vulnerable dep
// never enters the shipped scan.
module github.com/townsendmerino/aikit/tools/canary/vulnfixture

go 1.26.6

require golang.org/x/text v0.3.5

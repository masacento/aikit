// Package api is the OLD side of the apidiff canary: its exported surface differs
// incompatibly from ../new, so apidiff MUST report a break. Never reconcile the two.
package api

// F is the canary symbol. In old it takes an int; in new, a string — an incompatible change.
func F(x int) {}

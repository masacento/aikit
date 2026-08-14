// Package api is the NEW side of the apidiff canary: F's signature is incompatible with
// ../old, so apidiff MUST report it. Never reconcile the two.
package api

// F is the canary symbol. Its parameter type changed from int (old) to string — a break.
func F(x string) {}

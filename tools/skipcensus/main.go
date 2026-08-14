// Command skipcensus runs the root module's tests and prints the SKIP tally by reason next to
// the verdict, so the automated path stops folding skips into a bare green. It is a thin
// wrapper over the skips package: it forwards its arguments to `go test -json`, re-emits the
// normal test log, and appends the census. The exit code is `go test`'s own — this reports,
// it does not gate (the two hand-run gpu gates are where a skip becomes a failure).
//
// CI's `test (race)` step calls it as `go run -C tools ./skipcensus -race ./...` in place of a
// bare `go test -race ./...`; the census then rides on the run CI already pays for.
//
// Usage:  go run -C tools ./skipcensus [go test args…]   (default: ./...)
package main

import (
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/tools/gpumod"
	"github.com/townsendmerino/aikit/tools/skips"
)

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Fprintln(os.Stderr, "skipcensus: cannot locate repo root: "+err.Error())
		return 2
	}
	if len(args) == 0 {
		args = []string{"./..."}
	}
	res, err := skips.Run(root, nil, args, os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, "skipcensus: "+err.Error())
		if res.ExitCode == 0 {
			return 1
		}
	}
	fmt.Println()
	fmt.Println("skip census — " + res.Summary())
	return res.ExitCode
}

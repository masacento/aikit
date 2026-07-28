// Command benchreport is the `bench report` step of docs/BENCH-gpu.md: it reads one or more
// records.jsonl files (this run, plus a second machine's file when co-located) and writes the
// GENERATED results doc to stdout. The results doc is never hand-edited.
//
//	go run ./bench/cmd/benchreport records.jsonl [nvidia-records.jsonl ...] > docs/BENCH-gpu-results.md
//
// Merging multiple machines' files is how the one normalized cross-platform table is built:
// Metal exists only on Apple Silicon and CUDA only on NVIDIA, so they never co-reside — each
// box emits its own file and this step joins them on (workload × shape × precision).
package main

import (
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/bench"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: benchreport records.jsonl [more.jsonl ...] > docs/BENCH-gpu-results.md")
		os.Exit(2)
	}
	var all []bench.Record
	for _, path := range os.Args[1:] {
		recs, err := bench.LoadRecords(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "benchreport: %s: %v\n", path, err)
			os.Exit(1)
		}
		all = append(all, recs...)
	}
	fmt.Print(bench.Report(all))
}

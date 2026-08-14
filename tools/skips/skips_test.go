package skips

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestCategory_reasonsFromCensus(t *testing.T) {
	cases := []struct {
		reason, cat string
	}{
		{"no bge-small model at testdata/ — fetch + run scripts/oracle/pin_bge.py", "missing asset"},
		{"testdata/model/ not present; see testdata/README.md", "missing asset"},
		{"no Metal device: Metal is not available", "no GPU device"},
		{"no CUDA backend registered (no GPU?)", "no GPU device"},
		{"EnableResident: no Metal device? ", "no GPU device"},
		{"periodic GPU pass — set AIKIT_GPU_BENCH=1 to run", "env opt-in"},
		{"GOINFER_NOCOPY_PROBE=1 not set", "env opt-in"},
		{"CPU lacks AVX2/FMA; dotFMA asm path not exercised", "cpu feature"},
		{"CPU has no DotProd (ASIMDDP); SDOT kernel not exercisable", "cpu feature"},
		{"speedup measurement runs ~50s; skipped under -short", "short mode"},
		{"NVRTC not found", "missing external tool"},
		{"python3 not found", "missing external tool"},
		{"the two kernels happen to agree bitwise on this input", "other"},
		{"fixture has no fullatt blocks — the windowed/full switch is untested", "other"},
	}
	for _, c := range cases {
		if got := category(c.reason); got != c.cat {
			t.Errorf("category(%q) = %q, want %q", c.reason, got, c.cat)
		}
	}
}

// a canned `go test -json` stream: one pass, two skips (different reasons), one fail.
const jsonStream = `{"Action":"run","Package":"p","Test":"TestPass"}
{"Action":"output","Package":"p","Test":"TestPass","Output":"=== RUN   TestPass\n"}
{"Action":"pass","Package":"p","Test":"TestPass","Elapsed":0}
{"Action":"run","Package":"p","Test":"TestSkipAsset"}
{"Action":"output","Package":"p","Test":"TestSkipAsset","Output":"=== RUN   TestSkipAsset\n"}
{"Action":"output","Package":"p","Test":"TestSkipAsset","Output":"    bge_test.go:19: no bge-small model at testdata/\n"}
{"Action":"output","Package":"p","Test":"TestSkipAsset","Output":"--- SKIP: TestSkipAsset (0.00s)\n"}
{"Action":"skip","Package":"p","Test":"TestSkipAsset","Elapsed":0}
{"Action":"run","Package":"p","Test":"TestSkipGPU"}
{"Action":"output","Package":"p","Test":"TestSkipGPU","Output":"    smoke_test.go:23: no Metal device: unavailable\n"}
{"Action":"skip","Package":"p","Test":"TestSkipGPU","Elapsed":0}
{"Action":"run","Package":"p","Test":"TestFail"}
{"Action":"output","Package":"p","Test":"TestFail","Output":"    x_test.go:5: boom\n"}
{"Action":"fail","Package":"p","Test":"TestFail","Elapsed":0}
{"Action":"fail","Package":"p","Elapsed":0}
`

func TestParse_tallyAndReemit(t *testing.T) {
	// drive the parser directly by reusing its inner loop through a fake stream: Run execs
	// `go`, so here we exercise the classification+reason extraction the same way Run does.
	res := Result{ByReason: map[string]int{}, Examples: map[string]string{}}
	var out bytes.Buffer
	buf := map[string][]string{}
	for _, line := range strings.Split(strings.TrimRight(jsonStream, "\n"), "\n") {
		var ev testEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("unmarshal %q: %v", line, err)
		}
		key := ev.Package + "\x00" + ev.Test
		switch ev.Action {
		case "output":
			out.WriteString(ev.Output)
			if ev.Test != "" {
				buf[key] = append(buf[key], ev.Output)
			}
		case "skip":
			res.Skipped++
			cat := category(reasonFrom(buf[key]))
			res.ByReason[cat]++
			if _, ok := res.Examples[cat]; !ok {
				res.Examples[cat] = reasonFrom(buf[key])
			}
		case "pass":
			if ev.Test != "" {
				res.Passed++
			}
		case "fail":
			if ev.Test != "" {
				res.Failed++
			}
		}
	}
	if res.Passed != 1 || res.Skipped != 2 || res.Failed != 1 {
		t.Fatalf("tally = %d passed, %d skipped, %d failed; want 1/2/1", res.Passed, res.Skipped, res.Failed)
	}
	if res.ByReason["missing asset"] != 1 || res.ByReason["no GPU device"] != 1 {
		t.Errorf("by-reason = %v; want one missing-asset + one no-GPU-device", res.ByReason)
	}
	// the reconstructed log must contain the original RUN line verbatim.
	if !strings.Contains(out.String(), "=== RUN   TestPass") {
		t.Errorf("re-emitted log lost the RUN line:\n%s", out.String())
	}
	sum := res.Summary()
	if !strings.Contains(sum, "1 passed, 2 skipped, 1 FAILED") {
		t.Errorf("summary head = %q", sum)
	}
}

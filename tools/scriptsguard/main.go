// Command pyguard backs a precise claim about aikit's Python with a derivable rule rather than a
// hand-kept list: every .py under scripts/ lives in exactly one of two declared categories, and a
// stray .py directly under scripts/ (the residue the zero-Python campaign removed) fails CI.
//
//	scripts/oracle/   — reference generators that reach external ecosystems Go has no equivalent
//	                    for: PyTorch/HF goldens the Go tests assert against, and llama.cpp's ggml
//	                    grid tables (the tables ARE the reference; a Go port would transcribe the
//	                    constants the generator exists to avoid transcribing). Irreducible.
//	scripts/fixtures/ — kept on cost grounds, reducible in principle: prep_beir.py fetches an HF
//	                    parquet dataset for a manual benchmark behind no gate; porting it would add
//	                    a parquet dependency to a repo that watches its graph, for no gate.
//	                    Reviewable — a later reader can re-decide rather than inherit it forever.
//
// The rule is one-per-directory and derived by path, so nothing is hand-enumerated (B7): adding a
// pin_ script to oracle/ needs no edit here. Scope is scripts/ — the campaign's residue lived
// there; the gpu/ CUDA build toolchain (build_ptx.sh + nvrtc_compile.py, NVRTC via a C library Go
// has no build-time binding for, shared verbatim with goinfer) is separate build infrastructure,
// out of scope by that boundary.
//
// The DENOMINATOR is printed so a pass and a scan-of-nothing differ (that ambiguity has cost real
// time). Usage: go run -C tools ./pyguard
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/tools/gpumod"
)

func main() { os.Exit(run()) }

func run() int {
	root, err := gpumod.RepoRoot()
	if err != nil {
		fmt.Println("pyguard: CANNOT-EVALUATE — cannot locate repo root: " + err.Error())
		return 2
	}
	scripts := filepath.Join(root, "scripts")
	var oracle, fixtures, stray []string
	_ = filepath.WalkDir(scripts, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(d.Name(), ".py") {
			return nil
		}
		rel, _ := filepath.Rel(scripts, p) // e.g. "oracle/pin_bge.py"
		switch {
		case strings.HasPrefix(rel, "oracle"+string(filepath.Separator)):
			oracle = append(oracle, rel)
		case strings.HasPrefix(rel, "fixtures"+string(filepath.Separator)):
			fixtures = append(fixtures, rel)
		default:
			stray = append(stray, "scripts/"+rel)
		}
		return nil
	})
	n := len(oracle) + len(fixtures) + len(stray)

	// Denominator first — a scan that found nothing must not read as a pass.
	fmt.Printf("pyguard: %d .py under scripts/ — %d under scripts/oracle/, %d under scripts/fixtures/, %d elsewhere\n",
		n, len(oracle), len(fixtures), len(stray))
	if n == 0 {
		fmt.Println("pyguard: CANNOT-EVALUATE — found no .py under scripts/ at all (did the tree move, or is scripts/ absent?)")
		return 2
	}
	if len(stray) > 0 {
		sort.Strings(stray)
		for _, s := range stray {
			fmt.Println("  STRAY: " + s)
		}
		fmt.Printf("pyguard: FAIL — %d .py under scripts/ outside scripts/{oracle,fixtures}/. "+
			"Every scripts/ Python must be an external-reference generator (oracle/) or a declared "+
			"fixture (fixtures/); if it is neither, it is residue — port it, or move it to goinfer if "+
			"it is goinfer's.\n", len(stray))
		return 1
	}
	fmt.Println("pyguard: OK — aikit has no Python under scripts/ except (a) reference generators " +
		"reaching external ecosystems Go has no equivalent for (PyTorch/HF goldens, llama.cpp ggml " +
		"tables) and (b) one fixture fetcher kept because porting it would add a parquet dependency " +
		"for a manual benchmark.")
	return 0
}

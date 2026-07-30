package bm25

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// intern_test.go gates Build's key interning (task-perf-memoization §1b). Score parity is
// already covered by scoring_identity_test.go / index_test.go (interning changes only the
// key BACKING, never its bytes). Here we prove the two properties interning adds: (1) the
// retained postings/df keys do NOT alias the caller's token strings — so (2) a built Index
// no longer pins the source texts / per-call tokenizer arenas its keys used to hold alive.

func realSourceTexts(t testing.TB) []string {
	t.Helper()
	var texts []string
	_ = filepath.WalkDir("..", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "testdata", "node_modules", ".venv":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") {
			return nil
		}
		if b, err := os.ReadFile(p); err == nil {
			texts = append(texts, string(b))
		}
		return nil
	})
	return texts
}

func strPtr(s string) uintptr { return uintptr(unsafe.Pointer(unsafe.StringData(s))) }

// TestBuildInternsKeys proves Build's postings/df keys are clones, not views into the
// tokens it was handed — the property that unpins the Index.
func TestBuildInternsKeys(t *testing.T) {
	text := "func FuncName_helper does func things with Helper and FuncName_helper"
	toks := Tokenize(text)
	if len(toks) == 0 {
		t.Fatal("no tokens")
	}
	// Record the backing pointer of each input token occurrence.
	inputPtrs := map[string][]uintptr{}
	for _, tk := range toks {
		inputPtrs[tk] = append(inputPtrs[tk], strPtr(tk))
	}

	ix := Build([][]string{toks})

	for term := range ix.terms {
		kp := strPtr(term)
		for _, ip := range inputPtrs[term] {
			if kp == ip {
				t.Fatalf("Index key %q aliases an input token (backing %#x) — not interned", term, kp)
			}
		}
		// Also must not point into the source text.
		if kp >= strPtr(text) && kp < strPtr(text)+uintptr(len(text)) {
			t.Fatalf("Index key %q aliases the source text — not interned", term)
		}
	}
}

// TestBuildUnpinsSources proves the un-pinning by a differential that is robust to
// whole-heap noise: with the Index held live, dropping the source texts + token slices
// must actually RECLAIM their bytes. If the Index's keys still aliased the sources (the
// pre-intern behavior), the texts could not be freed and almost nothing would be
// reclaimed. We assert the reclaimed amount is a large fraction of the source corpus.
func TestBuildUnpinsSources(t *testing.T) {
	texts := realSourceTexts(t)
	if len(texts) < 50 {
		t.Skipf("only %d source docs", len(texts))
	}
	var srcBytes int
	for _, tx := range texts {
		srcBytes += len(tx)
	}
	docs := make([][]string, len(texts))
	for i, tx := range texts {
		docs[i] = Tokenize(tx)
	}
	ix := Build(docs)

	readHeap := func() uint64 {
		runtime.GC()
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapAlloc
	}

	// H0: Index + corpus (docs + source texts) all live.
	h0 := readHeap()
	// Drop everything a caller would drop after Build; keep only the Index.
	// Clearing the ELEMENTS is what releases the strings — nilling the slice
	// variables themselves would be ineffectual (Go's liveness analysis already
	// treats them as dead past their last use, and staticcheck says so).
	for i := range docs {
		docs[i] = nil
	}
	for i := range texts {
		texts[i] = ""
	}
	// H1: Index only. The corpus must be gone — the Index must not pin it.
	h1 := readHeap()
	runtime.KeepAlive(ix)

	reclaimed := int64(h0) - int64(h1)
	t.Logf("source corpus=%.2f MB; heap with corpus=%.2f MB, Index-only=%.2f MB, reclaimed=%.2f MB",
		float64(srcBytes)/(1<<20), float64(h0)/(1<<20), float64(h1)/(1<<20), float64(reclaimed)/(1<<20))
	// The source texts alone are several MB; if the Index pinned them via aliased keys,
	// reclaimed would be near zero. Require at least half the source bytes back.
	if reclaimed < int64(srcBytes)/2 {
		t.Errorf("only reclaimed %.2f MB of a %.2f MB corpus — Index is still pinning its key backing",
			float64(reclaimed)/(1<<20), float64(srcBytes)/(1<<20))
	}
}

package embed

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"
)

// wordpiece_memo_test.go gates the wordPiece memo (task-perf-memoization §1): the cached path
// must return byte-identical ids to the uncached wordPieceCompute over the WHOLE real corpus (a
// memo bug hides in a rare word, not a common one), and it measures the real repetition rate +
// cold-vs-memo speedup against the REAL vocab (the task doc's number used a synthesized vocab).

const memoModel = "../testdata/minilm-model"

var wordRE = regexp.MustCompile(`[A-Za-z]+`)

// realWords walks the repo's own Go sources and lowercases every identifier fragment — a real,
// repetition-heavy corpus (code is dense in shared identifiers), matching the task doc's method.
func realWords(t testing.TB) []string {
	t.Helper()
	var words []string
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
		b, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		for _, w := range wordRE.FindAllString(string(b), -1) {
			words = append(words, strings.ToLower(w))
		}
		return nil
	})
	return words
}

func loadWPTokenizer(t testing.TB) *Tokenizer {
	t.Helper()
	tjson := memoModel + "/tokenizer.json"
	if _, err := os.Stat(tjson); err != nil {
		t.Skipf("no tokenizer at %s", tjson)
	}
	tok, err := LoadTokenizer(tjson)
	if err != nil {
		t.Skipf("LoadTokenizer: %v", err)
	}
	if tok.wp == nil {
		t.Skip("not a WordPiece tokenizer (no memo to test)")
	}
	return tok
}

func eqIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestWordPieceMemo_equiv is the differential fuzz + the repetition-rate report.
func TestWordPieceMemo_equiv(t *testing.T) {
	tok := loadWPTokenizer(t)
	words := realWords(t)
	if len(words) < 1000 {
		t.Skipf("only %d words scanned", len(words))
	}

	// Differential parity over EVERY distinct word: cached wordPiece ≡ uncached compute.
	seen := map[string]bool{}
	var totalRunes, uniqueRunes int
	for _, w := range words {
		if !seen[w] {
			seen[w] = true
			uniqueRunes += utf8.RuneCountInString(w)
			cached := tok.wordPiece(w)
			uncached := tok.wordPieceCompute([]rune(w))
			if !eqIDs(cached, uncached) {
				t.Fatalf("MEMO DIVERGENCE on %q: cached %v != uncached %v", w, cached, uncached)
			}
		}
		totalRunes += utf8.RuneCountInString(w)
	}
	// ref is wordPiece WITHOUT the cache — the short-circuits plus the uncached compute.
	ref := func(w string) []int32 {
		if utf8.RuneCountInString(w) > tok.maxCharsPerWord {
			return []int32{tok.unkID}
		}
		r := []rune(w)
		if len(r) == 0 {
			return nil
		}
		return tok.wordPieceCompute(r)
	}
	// Adversarial edges: empty, over-cap (→ unk), unicode.
	for _, w := range []string{"", strings.Repeat("x", tok.maxCharsPerWord+5), "café", "naïve", "日本語"} {
		if !eqIDs(tok.wordPiece(w), ref(w)) {
			t.Fatalf("edge-case divergence on %q", w)
		}
	}

	total, uniq := len(words), len(seen)
	t.Logf("REAL vocab (minilm): %d total words → %d unique = %.1f%% of wordPiece calls are repeats",
		total, uniq, 100*float64(total-uniq)/float64(total))
	t.Logf("mean word length: %.2f over all, %.2f over unique", float64(totalRunes)/float64(total), float64(uniqueRunes)/float64(uniq))
}

// TestWordPieceMemo_concurrent hammers the sharded cache from many goroutines (as EncodeBatch
// does) — under -race this proves the get/put path is race-free, and it re-checks parity so a
// torn write can't slip through. Uses a fresh tokenizer so the cache starts cold and races on
// the first-write path, not just reads.
func TestWordPieceMemo_concurrent(t *testing.T) {
	tok := loadWPTokenizer(t)
	words := realWords(t)
	if len(words) < 1000 {
		t.Skip("corpus too small")
	}
	// distinct-word reference (uncached), computed single-threaded up front
	ref := map[string][]int32{}
	for _, w := range words {
		if _, ok := ref[w]; !ok {
			ref[w] = tok.wordPieceCompute([]rune(w))
		}
	}
	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan string, workers)
	for g := range workers {
		wg.Add(1)
		go func(off int) {
			defer wg.Done()
			for i := range words {
				w := words[(i+off*97)%len(words)]
				if !eqIDs(tok.wordPiece(w), ref[w]) {
					errs <- w
					return
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	if w, bad := <-errs; bad {
		t.Fatalf("concurrent memo divergence on %q", w)
	}
}

// BenchmarkWordPiece_Cold recomputes every word (no cache) — the pre-memo cost.
func BenchmarkWordPiece_Cold(b *testing.B) {
	tok := loadWPTokenizer(b)
	words := realWords(b)
	if len(words) < 1000 {
		b.Skip("corpus too small")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for _, w := range words {
			_ = tok.wordPieceCompute([]rune(w))
		}
	}
}

// BenchmarkWordPiece_Memo goes through the cache (warm after the first pass) — the steady state
// of bulk corpus embedding, where ~98% of calls hit.
func BenchmarkWordPiece_Memo(b *testing.B) {
	tok := loadWPTokenizer(b)
	words := realWords(b)
	if len(words) < 1000 {
		b.Skip("corpus too small")
	}
	for _, w := range words { // warm the cache
		_ = tok.wordPiece(w)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for _, w := range words {
			_ = tok.wordPiece(w)
		}
	}
}

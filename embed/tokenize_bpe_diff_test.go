package embed

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// bpeStringRef is the ORIGINAL string-based bpe(), kept here as the reference the
// span-based bpeInto (lens §3.3) must match symbol-for-symbol.
func bpeStringRef(b *bpeBackend, mapped string) []string {
	symbols := make([]string, 0, len(mapped))
	for _, r := range mapped {
		symbols = append(symbols, string(r))
	}
	if len(symbols) < 2 {
		return symbols
	}
	for {
		bestRank := int(^uint(0) >> 1)
		bestI := -1
		for i := 0; i < len(symbols)-1; i++ {
			if r, ok := b.rank[[2]string{symbols[i], symbols[i+1]}]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		a, c := symbols[bestI], symbols[bestI+1]
		merged := symbols[:0:0]
		for i := 0; i < len(symbols); {
			if i < len(symbols)-1 && symbols[i] == a && symbols[i+1] == c {
				merged = append(merged, a+c)
				i += 2
			} else {
				merged = append(merged, symbols[i])
				i++
			}
		}
		symbols = merged
	}
	return symbols
}

// refEncodeSegment is encodeSegment via the string reference.
func (b *bpeBackend) refEncodeSegment(text string) []int32 {
	var ids []int32
	for _, piece := range b.preTokenize(text) {
		var sb strings.Builder
		sb.Grow(len(piece))
		for i := 0; i < len(piece); i++ {
			sb.WriteRune(b.byte2rune[piece[i]])
		}
		for _, sym := range bpeStringRef(b, sb.String()) {
			id, ok := b.vocab[sym]
			if !ok {
				id = b.unkID
			}
			ids = append(ids, id)
		}
	}
	return ids
}

// TestBPESpan_matchesStringRef: the span-based bpeInto must produce ids identical
// to the original string-based algorithm over the whole real corpus (a merge-order
// bug hides in a rare word, not a common one).
func TestBPESpan_matchesStringRef(t *testing.T) {
	const tj = "../testdata/granite-en/tokenizer.json"
	if _, err := os.Stat(tj); err != nil {
		t.Skip("no granite tokenizer")
	}
	tok, err := LoadTokenizer(tj)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if tok.bpe == nil {
		t.Skip("not a BPE tokenizer")
	}
	b := tok.bpe

	var texts []string
	_ = filepath.WalkDir("..", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "testdata", ".venv", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(p, ".go") {
			if bs, e := os.ReadFile(p); e == nil {
				texts = append(texts, string(bs))
			}
		}
		return nil
	})
	if len(texts) < 50 {
		t.Skipf("only %d texts", len(texts))
	}

	for _, text := range texts {
		got := b.encodeSegment(text) // span-based
		want := b.refEncodeSegment(text)
		if len(got) != len(want) {
			t.Fatalf("len %d != %d", len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("id[%d] %d != %d", i, got[i], want[i])
			}
		}
	}
	t.Logf("span-based BPE bit-identical to the string reference over %d files", len(texts))
}

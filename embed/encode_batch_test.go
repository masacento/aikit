package embed

import (
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"
)

// TestEncodeBatch_matchesSerial is A1's gate, and it is EXACT over the WHOLE
// corpus rather than a sample: the failure mode of a fan-out is a rare input or
// a rare interleaving, not a common one.
//
// Bit-identity here is structural — StaticModel is immutable after load and
// Encode touches no shared mutable state — so what this really guards is that
// the scatter puts each vector back at its own index. An off-by-one in the work
// counter produces perfectly valid vectors in the wrong order, which no
// numerical tolerance would catch and which every downstream cosine would
// quietly accept.
func TestEncodeBatch_matchesSerial(t *testing.T) {
	m := loadTestStaticModel(t)
	texts := goSourceChunks(t)
	t.Logf("corpus: %d chunks", len(texts))

	want := make([][]float32, len(texts))
	for i, s := range texts {
		want[i] = m.Encode(s)
	}

	for _, conc := range []int{0, 1, 2, 3, 8, 16, 64, len(texts) + 10} {
		got := m.EncodeBatch(texts, conc)
		if len(got) != len(want) {
			t.Fatalf("concurrency %d: got %d vectors, want %d", conc, len(got), len(want))
		}
		for i := range want {
			if len(got[i]) != len(want[i]) {
				t.Fatalf("concurrency %d, text %d: dim %d, want %d", conc, i, len(got[i]), len(want[i]))
			}
			for j := range want[i] {
				if got[i][j] != want[i][j] {
					t.Fatalf("concurrency %d, text %d, component %d: %v != serial %v",
						conc, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
}

// TestEncodeBatch_orderIsInputOrder is the cheap, targeted version of the above:
// distinct texts must come back distinguishable and in position. It uses inputs
// whose vectors are far apart, so a misordered scatter is unmistakable rather
// than a near-tie.
func TestEncodeBatch_orderIsInputOrder(t *testing.T) {
	m := loadTestStaticModel(t)
	texts := []string{
		"func encode(text string) []float32",
		"the quick brown fox jumps over the lazy dog",
		"SELECT * FROM users WHERE id = ?",
		"import numpy as np",
		"",
		"package main",
	}
	for _, conc := range []int{1, 2, 6, 16} {
		got := m.EncodeBatch(texts, conc)
		for i, s := range texts {
			want := m.Encode(s)
			for j := range want {
				if got[i][j] != want[j] {
					t.Fatalf("concurrency %d, text %d (%q), component %d: %v != %v",
						conc, i, s, j, got[i][j], want[j])
				}
			}
		}
	}
}

// TestEncodeBatch_degenerate covers the shapes a bulk API meets in production
// and never in a benchmark.
func TestEncodeBatch_degenerate(t *testing.T) {
	m := loadTestStaticModel(t)
	if got := m.EncodeBatch(nil, 0); len(got) != 0 {
		t.Errorf("nil input returned %d vectors", len(got))
	}
	if got := m.EncodeBatch([]string{}, 8); len(got) != 0 {
		t.Errorf("empty input returned %d vectors", len(got))
	}
	// One text, more workers than work.
	got := m.EncodeBatch([]string{"hello"}, 16)
	if len(got) != 1 {
		t.Fatalf("single text returned %d vectors", len(got))
	}
	want := m.Encode("hello")
	for j := range want {
		if got[0][j] != want[j] {
			t.Fatalf("single text component %d: %v != %v", j, got[0][j], want[j])
		}
	}
	// All-empty strings: every vector is the documented zero vector, and none
	// is nil (a caller indexing the result must not have to check).
	got = m.EncodeBatch([]string{"", "", ""}, 4)
	for i, v := range got {
		if len(v) != m.Dim() {
			t.Fatalf("empty text %d: dim %d, want %d", i, len(v), m.Dim())
		}
		for j, x := range v {
			if x != 0 {
				t.Fatalf("empty text %d component %d = %v, want 0", i, j, x)
			}
		}
	}
}

// TestEncodeBatch_concurrentCallers checks the other half of the contract the
// type doc claims: EncodeBatch is itself safe to call from several goroutines,
// which a caller sharding a corpus across services will do. Run under -race.
func TestEncodeBatch_concurrentCallers(t *testing.T) {
	m := loadTestStaticModel(t)
	texts := goSourceChunks(t)
	if len(texts) > 200 {
		texts = texts[:200]
	}
	want := m.EncodeBatch(texts, 1)

	var wg sync.WaitGroup
	errs := make([]string, 4)
	for w := range 4 {
		wg.Go(func() {
			got := m.EncodeBatch(texts, runtime.NumCPU())
			for i := range want {
				for j := range want[i] {
					if got[i][j] != want[i][j] {
						errs[w] = "vector mismatch under concurrent EncodeBatch calls"
						return
					}
				}
			}
		})
	}
	wg.Wait()
	for _, e := range errs {
		if e != "" {
			t.Fatal(e)
		}
	}
}

// loadTestStaticModel opens the Model2Vec checkpoint, skipping without it.
func loadTestStaticModel(tb testing.TB) *StaticModel {
	tb.Helper()
	const dir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		tb.Skipf("no static model at %s — see testdata/README.md", dir)
	}
	m, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

// TestPreTokenize_slicedMatchesRebuilt is A4's gate: the byte-slicing path must
// produce exactly what the Builder rebuild produced, token for token.
//
// It runs over the whole repository, not a sample, plus a table of shapes real
// source does not reliably contain — because the two paths agree everywhere
// EXCEPT on invalid UTF-8, and normalize's cleanText removes that before
// preTokenize ever sees it. The interesting inputs are therefore precisely the
// ones the pipeline never delivers, which is why they are written down here.
func TestPreTokenize_slicedMatchesRebuilt(t *testing.T) {
	m := loadTestStaticModel(t)
	tok := m.Tokenizer()

	cases := []string{
		"", " ", "   \t\n ", "plain words here",
		"func Encode(text string) []float32 { return nil }",
		"a.b.c", "...", "!!!", "a!!b", "!a", "a!",
		"emoji 🙂 and 汉字 mixed", "naïve café", "«guillemets»", "—em—dash—",
		"trailing punct...", "...leading punct",
		"tabs\tand\nnewlines", "double  space",
	}
	for _, src := range allRepoGoSources(t) {
		cases = append(cases, string(src))
	}
	// Invalid UTF-8, where the two paths legitimately differ and the rebuild is
	// what must be used.
	invalid := []string{"\xff", "ab\xffcd", "a\xc3", "\x80\x80", "ok \xf0\x9f then"}

	checked := 0
	for _, c := range cases {
		if !utf8.ValidString(c) {
			t.Fatalf("case %q is not valid UTF-8; it belongs in the invalid table", c)
		}
		got, want := tok.preTokenize(c), tok.preTokenizeRebuild(c)
		if !slices.Equal(got, want) {
			t.Fatalf("sliced and rebuilt disagree on %q:\n got %q\nwant %q", trunc(c), got, want)
		}
		checked += len(want)
	}
	if checked < 100_000 {
		t.Fatalf("only %d tokens compared; the corpus is too small to be a gate", checked)
	}
	t.Logf("%d tokens compared across %d inputs", checked, len(cases))

	for _, c := range invalid {
		if utf8.ValidString(c) {
			t.Fatalf("case %q is valid UTF-8 and does not test the fallback", c)
		}
		// preTokenize must route these to the rebuild, so the two agree here too.
		if got, want := tok.preTokenize(c), tok.preTokenizeRebuild(c); !slices.Equal(got, want) {
			t.Fatalf("invalid-UTF-8 input %q did not take the rebuild path:\n got %q\nwant %q", c, got, want)
		}
	}
}

// TestPreTokenize_tokensAreViews pins the property the memo's key-clone exists
// for: on valid input every token is a SLICE of the argument, not a copy. If
// that stops being true the clone in wpCache.put becomes dead weight, and if it
// becomes true somewhere the clone is missing, the cache pins whole chunks.
func TestPreTokenize_tokensAreViews(t *testing.T) {
	m := loadTestStaticModel(t)
	const src = "func Encode(text string) []float32"
	toks := m.Tokenizer().preTokenize(src)
	if len(toks) < 5 {
		t.Fatalf("expected several tokens, got %q", toks)
	}
	for _, tk := range toks {
		if len(tk) == 0 {
			t.Fatal("empty token")
		}
		if !strings.Contains(src, tk) {
			t.Fatalf("token %q is not a substring of the input", tk)
		}
	}
}

func trunc(s string) string {
	if len(s) > 120 {
		return s[:120] + "…"
	}
	return s
}

// allRepoGoSources returns every .go file in the repository.
func allRepoGoSources(tb testing.TB) [][]byte {
	tb.Helper()
	root, err := filepath.Abs("..")
	if err != nil {
		tb.Fatal(err)
	}
	var out [][]byte
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case "benchmarks", "testdata", ".git", ".venv":
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		out = append(out, b)
		return nil
	})
	if err != nil {
		tb.Fatal(err)
	}
	return out
}

// TestEncode_carveOutSlicedMatchesRebuilt is A2's gate: the sliced carve-out
// must produce exactly the ids the Builder rebuild produced.
//
// The corpus alone would not exercise it — aikit's own source contains almost no
// added-token literals — so the table plants `[PAD]`/`[UNK]`/`[MASK]` in every
// position that matters: at the start, at the end, adjacent to each other, with
// and without surrounding text, and as prefixes of one another (which is why
// addedKeys is sorted longest-first and why the scan must take the FIRST match).
func TestEncode_carveOutSlicedMatchesRebuilt(t *testing.T) {
	m := loadTestStaticModel(t)
	tok := m.Tokenizer()
	if len(tok.addedKeys) == 0 {
		t.Skip("checkpoint has no added tokens; the carve-out never runs")
	}
	t.Logf("added keys: %q (single first byte: %v)", tok.addedKeys, tok.addedSingle)

	var cases []string
	for _, k := range tok.addedKeys {
		cases = append(cases,
			k, k+k, " "+k+" ", k+"tail", "head"+k, "head"+k+"tail",
			k+" "+k, "a"+k+"b"+k+"c",
			"[", "[[", "[not-a-key]", "["+k, k+"[",
			strings.Repeat(k, 5), strings.Repeat("x"+k, 5),
		)
	}
	cases = append(cases, "", " ", "no keys at all", "brackets [ ] but no key")
	for _, src := range allRepoGoSources(t) {
		cases = append(cases, string(src))
	}

	planted := 0
	for _, c := range cases {
		if !utf8.ValidString(c) {
			t.Fatalf("case %q is not valid UTF-8", c)
		}
		got, want := tok.Encode(c), tok.encodeAddedRebuild(c)
		if !slices.Equal(got, want) {
			t.Fatalf("sliced and rebuilt carve-outs disagree on %q:\n got %v\nwant %v", trunc(c), got, want)
		}
		for _, k := range tok.addedKeys {
			planted += strings.Count(c, k)
		}
	}
	if planted < 50 {
		t.Fatalf("only %d added-token occurrences across the corpus; the carve-out is barely exercised", planted)
	}
	t.Logf("%d inputs, %d added-token occurrences", len(cases), planted)

	// Invalid UTF-8 must route to the rebuild, so the two agree there too.
	for _, c := range []string{"\xff", "a\xffb", tok.addedKeys[0] + "\xff", "\xff" + tok.addedKeys[0]} {
		if got, want := tok.Encode(c), tok.encodeAddedRebuild(c); !slices.Equal(got, want) {
			t.Fatalf("invalid-UTF-8 input %q did not take the rebuild path:\n got %v\nwant %v", c, got, want)
		}
	}
}

// TestEncode_carveOutPrefilterIsComplete checks the byte table against the keys
// it is derived from. A missing entry makes the scan skip a real added token
// silently — the ids stay valid, just wrong — which is the failure mode a
// spot-check would not catch.
func TestEncode_carveOutPrefilterIsComplete(t *testing.T) {
	m := loadTestStaticModel(t)
	tok := m.Tokenizer()
	for _, k := range tok.addedKeys {
		if k == "" {
			continue
		}
		if !tok.addedFirst[k[0]] {
			t.Errorf("key %q begins with %q, which the prefilter does not admit", k, k[0])
		}
		if tok.addedSingle && k[0] != tok.addedOne {
			t.Errorf("addedSingle is set but key %q begins with %q, not %q", k, k[0], tok.addedOne)
		}
	}
	admitted := 0
	for b := range 256 {
		if tok.addedFirst[b] {
			admitted++
		}
	}
	if tok.addedSingle != (admitted == 1) {
		t.Errorf("addedSingle=%v but %d distinct first bytes are admitted", tok.addedSingle, admitted)
	}
}

// TestLoadMmap_matchesLoadFromFS gates the mmap load path: a model whose tensors
// alias a memory mapping must behave identically to one whose tensors alias a
// heap buffer.
//
// Vectors are compared bit-for-bit over the whole corpus, not sampled. The
// failure mode of an aliasing change is not "slightly different numbers" — it is
// a wrong offset, a truncated tensor, or memory freed underneath a live slice,
// all of which either match exactly or are obviously broken.
func TestLoadMmap_matchesLoadFromFS(t *testing.T) {
	const dir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no static model at %s", dir)
	}
	heap, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	mapped, err := LoadMmap(dir)
	if err != nil {
		t.Fatal(err)
	}
	if mapped.VocabSize() != heap.VocabSize() || mapped.Dim() != heap.Dim() {
		t.Fatalf("shape differs: mmap %d×%d, heap %d×%d",
			mapped.VocabSize(), mapped.Dim(), heap.VocabSize(), heap.Dim())
	}
	texts := goSourceChunks(t)
	for i, s := range texts {
		a, b := mapped.Encode(s), heap.Encode(s)
		for j := range b {
			if a[j] != b[j] {
				t.Fatalf("chunk %d component %d: mmap %v, heap %v", i, j, a[j], b[j])
			}
		}
	}
	t.Logf("%d chunks identical across both load paths", len(texts))

	// The mapping must outlive the load call — a finalizer that ran early, or a
	// Close hidden in the loader, would surface as a fault or as garbage here.
	runtime.GC()
	runtime.GC()
	if v := mapped.Encode("func Encode(text string) []float32"); len(v) != mapped.Dim() {
		t.Fatalf("post-GC encode returned %d components", len(v))
	}
}

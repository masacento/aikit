package embed

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// BenchmarkEncodeSplit decomposes StaticModel.Encode into its two halves —
// Tokenizer.Encode and encodeIDs (gather + weighted pool + L2) — over aikit's
// own source tree. Step 0c of docs/internal/task-perf-handoff-linux.md.
//
// It lives in package embed rather than alongside the W1/W2 workload benchmarks
// because encodeIDs is unexported. That is the point: the alternative is to
// measure Encode and Tokenizer.Encode and call the difference "pooling", which
// silently attributes every mismeasurement and all of the slice allocation in
// Encode's seam to whichever half you were less careful about. Here both halves
// are measured directly and `whole` checks that they add up.
//
// The split decides how much of the campaign's tokenizer work is worth doing:
// items A2, A3 and A4 all live inside Tokenizer.Encode, and their end-to-end
// value scales linearly with its share.
func BenchmarkEncodeSplit(b *testing.B) {
	const dir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		b.Skipf("no static model at %s — see testdata/README.md", dir)
	}
	m, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		b.Fatal(err)
	}
	texts := goSourceChunks(b)

	// Pre-tokenize so the pooling half measures only pooling.
	idSets := make([][]int32, len(texts))
	var totalIDs int
	for i, t := range texts {
		idSets[i] = m.tokenizer.Encode(t)
		totalIDs += len(idSets[i])
	}
	var totalBytes int
	for _, t := range texts {
		totalBytes += len(t)
	}
	b.Logf("corpus: %d chunks, %d bytes, %d WordPiece ids, vocab %d, dim %d",
		len(texts), totalBytes, totalIDs, m.VocabSize(), m.Dim())

	b.Run("tokenize", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for _, t := range texts {
				sinkIDs = m.tokenizer.Encode(t)
			}
		}
	})
	b.Run("pool", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for _, ids := range idSets {
				sinkEmb = m.encodeIDs(ids)
			}
		}
	})
	b.Run("whole", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for _, t := range texts {
				sinkEmb = m.Encode(t)
			}
		}
	})
}

// goSourceChunks slices aikit's own .go files into ~1500-byte pieces on line
// boundaries — the same corpus and chunk size the workload benchmarks use,
// reproduced here without importing chunk (which would be an import cycle
// through nothing, but also would put a second package's behaviour inside a
// measurement of this one).
func goSourceChunks(tb testing.TB) []string {
	tb.Helper()
	root, err := filepath.Abs("..")
	if err != nil {
		tb.Fatal(err)
	}
	const chunkSize = 1500
	var out []string
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
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		var cur strings.Builder
		for line := range strings.SplitSeq(string(src), "\n") {
			if cur.Len()+len(line)+1 > chunkSize && cur.Len() > 0 {
				out = append(out, cur.String())
				cur.Reset()
			}
			cur.WriteString(line)
			cur.WriteByte('\n')
		}
		if cur.Len() > 0 {
			out = append(out, cur.String())
		}
		return nil
	})
	if err != nil {
		tb.Fatal(err)
	}
	if len(out) == 0 {
		tb.Fatal("no .go files found; the corpus walk is broken")
	}
	return out
}

var (
	sinkIDs []int32
	sinkEmb []float32
)

package embed

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
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

// BenchmarkEncodeBatch sweeps the fan-out width over the real corpus. This is
// A1's deliverable and the measurement this box is best placed to make: 8
// homogeneous cores with SMT, no P/E straggler to confound the curve.
//
// The sweep matters more than a single number. Speedup on a memory-bound
// workload stops tracking core count well before the core count runs out, and
// where it stops is what tells a caller what to pass — `concurrency <= 0` means
// NumCPU, which on a 16-thread box is a guess, not a measurement.
func BenchmarkEncodeBatch(b *testing.B) {
	const dir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		b.Skipf("no static model at %s — see testdata/README.md", dir)
	}
	m, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		b.Fatal(err)
	}
	texts := goSourceChunks(b)
	b.Logf("corpus: %d chunks", len(texts))

	// The serial loop every caller writes today, as the baseline.
	b.Run("serialLoop", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			out := make([][]float32, len(texts))
			for i, t := range texts {
				out[i] = m.Encode(t)
			}
			sinkBatch = out
		}
	})
	for _, c := range []int{1, 2, 3, 4, 6, 8, 12, 16} {
		b.Run(fmt.Sprintf("c%d", c), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkBatch = m.EncodeBatch(texts, c)
			}
		})
	}
}

var sinkBatch [][]float32

// BenchmarkPreTokenize A/Bs the two implementations in one binary — the sliced
// path against the Builder rebuild it replaced, which is still present because
// invalid UTF-8 needs it. No cross-invocation drift to argue about, and the
// ratio it reports is the one that belongs to A4 rather than to the whole
// tokenizer stage around it.
func BenchmarkPreTokenize(b *testing.B) {
	const dir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		b.Skipf("no static model at %s — see testdata/README.md", dir)
	}
	m, err := LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		b.Fatal(err)
	}
	tok := m.tokenizer
	// Normalized text, which is what preTokenize actually receives.
	chunks := goSourceChunks(b)
	texts := make([]string, len(chunks))
	var n int
	for i, c := range chunks {
		texts[i] = tok.normalize(c)
		n += len(texts[i])
	}
	b.Logf("%d chunks, %d normalized bytes", len(texts), n)

	b.Run("rebuild", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(n))
		for b.Loop() {
			for _, t := range texts {
				sinkWords = tok.preTokenizeRebuild(t)
			}
		}
	})
	b.Run("sliced", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(n))
		for b.Loop() {
			for _, t := range texts {
				sinkWords = tok.preTokenize(t)
			}
		}
	})
	// The validity scan the sliced path pays for exactness on every input.
	b.Run("validityScanOnly", func(b *testing.B) {
		b.SetBytes(int64(n))
		for b.Loop() {
			ok := true
			for _, t := range texts {
				ok = ok && utf8.ValidString(t)
			}
			sinkValid = ok
		}
	})
}

var (
	sinkWords []string
	sinkValid bool
)

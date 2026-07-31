package bench

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/chunk"
	_ "github.com/townsendmerino/aikit/chunk/regex" // registers the "regex" chunker
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/fuse"
)

// End-to-end workload benchmarks — W1 (index a repository) and W2 (hybrid
// query). Step 0c of docs/internal/archive/perf-campaign-2026-07/task-perf-handoff-linux.md.
//
// WHY THESE EXIST AND WHAT THEY ARE FOR. Every other benchmark in this repo
// measures one function. These measure the two things a user actually runs, and
// they carry a sub-benchmark per stage so the Amdahl split falls out directly:
// a 3× win on a stage worth 5% of the run is a 1.7% win, and only a table like
// this says which is which. docs/internal/archive/perf-campaign-2026-07/task-perf-lens-scans.md §5 has the
// prior table, measured on a 2-core Xeon with a SYNTHETIC vocabulary; this is
// the first one on real hardware with the real checkpoint.
//
// Read the stage numbers as a decomposition, not as separate facts: `sum` runs
// the same pipeline end to end, so stage-sum versus `sum` is a completeness
// check. If they diverge, the decomposition is missing something and the
// percentages are wrong.
//
// The corpus is aikit's own tree, which is what every measurement in the lens
// docs used. `benchmarks/` is excluded: it is a separate Go module.

const workloadChunkSize = chunk.DefaultChunkSize // 1500, as the lens tables used

// repoCorpus reads aikit's own .go files. It returns paths and contents rather
// than pre-chunked text so the chunking stage is inside the measurement.
func repoCorpus(tb testing.TB) (paths []string, sources [][]byte) {
	tb.Helper()
	root, err := filepath.Abs("..")
	if err != nil {
		tb.Fatal(err)
	}
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// benchmarks/ is a separate module; testdata/ holds checkpoints and
			// generated fixtures, not source.
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
		rel, _ := filepath.Rel(root, p)
		paths = append(paths, rel)
		sources = append(sources, b)
		return nil
	})
	if err != nil {
		tb.Fatal(err)
	}
	if len(paths) == 0 {
		tb.Fatal("no .go files found; the corpus walk is broken")
	}
	return paths, sources
}

// loadStaticModel opens the Model2Vec checkpoint the examples use. Skips
// without it, so this file stays green on a machine that has no fixtures.
func loadStaticModel(tb testing.TB) *embed.StaticModel {
	tb.Helper()
	const dir = "../testdata/model"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		tb.Skipf("no static model at %s — see testdata/README.md", dir)
	}
	m, err := embed.LoadFromFS(os.DirFS(dir), ".")
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

// chunkAll is W1's first stage, factored out so later stages can reuse its
// output without re-measuring it.
func chunkAll(tb testing.TB, paths []string, sources [][]byte) []chunk.Chunk {
	tb.Helper()
	var out []chunk.Chunk
	for i, p := range paths {
		cs, err := chunk.ChunkFile("regex", p, sources[i], workloadChunkSize)
		if err != nil {
			tb.Fatal(err)
		}
		out = append(out, cs...)
	}
	return out
}

// BenchmarkW1 indexes a repository the way both shipped examples do it:
// serially, one chunk at a time.
//
// The sub-benchmarks are the deliverable. `sum` is the whole pipeline; the rest
// decompose it. Every stage runs over the identical corpus, so the ratios are
// directly comparable and stage-sum should land within a percent or two of
// `sum`.
func BenchmarkW1(b *testing.B) {
	m := loadStaticModel(b)
	paths, sources := repoCorpus(b)
	chunks := chunkAll(b, paths, sources)
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	docs := make([][]string, len(texts))
	for i, t := range texts {
		docs[i] = bm25.Tokenize(t)
	}
	vecs := make([][]float32, len(texts))
	for i, t := range texts {
		vecs[i] = m.Encode(t)
	}
	ix := ann.NewFlatI8(vecs)

	var bytes int
	for _, s := range sources {
		bytes += len(s)
	}
	b.Logf("corpus: %d files, %d bytes, %d chunks, dim %d", len(paths), bytes, len(chunks), m.Dim())

	b.Run("chunk", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for i, p := range paths {
				cs, err := chunk.ChunkFile("regex", p, sources[i], workloadChunkSize)
				if err != nil {
					b.Fatal(err)
				}
				sinkChunks = cs
			}
		}
	})
	b.Run("bm25Tokenize", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for _, t := range texts {
				sinkTokens = bm25.Tokenize(t)
			}
		}
	})
	b.Run("bm25Build", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkIndex = bm25.Build(docs)
		}
	})
	b.Run("embedEncode", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for _, t := range texts {
				sinkVec = m.Encode(t)
			}
		}
	})
	// A1: the same stage through EncodeBatch. Kept alongside the serial stage
	// rather than replacing it, because the serial loop is what every caller
	// wrote before EncodeBatch existed and is the baseline the ratio is against.
	b.Run("embedEncodeBatch", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkVecs = m.EncodeBatch(texts, 0)
		}
	})
	b.Run("newFlatI8", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkFlatI8 = ann.NewFlatI8(vecs)
		}
	})
	b.Run("marshalBinary", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			blob, err := ix.MarshalBinary()
			if err != nil {
				b.Fatal(err)
			}
			sinkBlob = blob
		}
	})
	b.Run("sumBatched", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cs := chunkAll(b, paths, sources)
			ts := make([]string, len(cs))
			ds := make([][]string, len(cs))
			for i, c := range cs {
				ts[i] = c.Text
				ds[i] = bm25.Tokenize(c.Text)
			}
			vs := m.EncodeBatch(ts, 0)
			sinkIndex = bm25.Build(ds)
			f := ann.NewFlatI8(vs)
			blob, err := f.MarshalBinary()
			if err != nil {
				b.Fatal(err)
			}
			sinkBlob = blob
		}
	})
	b.Run("sum", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cs := chunkAll(b, paths, sources)
			ds := make([][]string, len(cs))
			vs := make([][]float32, len(cs))
			for i, c := range cs {
				ds[i] = bm25.Tokenize(c.Text)
				vs[i] = m.Encode(c.Text)
			}
			sinkIndex = bm25.Build(ds)
			f := ann.NewFlatI8(vs)
			blob, err := f.MarshalBinary()
			if err != nil {
				b.Fatal(err)
			}
			sinkBlob = blob
		}
	})
}

// BenchmarkW2 is the hybrid retrieval query: tokenize, embed, run both
// retrievers, fuse. No rerank — see the lens doc's W2R, where the rerank is
// 99.8%+ of the query and every stage below is noise.
func BenchmarkW2(b *testing.B) {
	m := loadStaticModel(b)
	paths, sources := repoCorpus(b)
	chunks := chunkAll(b, paths, sources)
	docs := make([][]string, len(chunks))
	vecs := make([][]float32, len(chunks))
	for i, c := range chunks {
		docs[i] = bm25.Tokenize(c.Text)
		vecs[i] = m.Encode(c.Text)
	}
	bx := bm25.Build(docs)
	fx := ann.NewFlatI8(vecs)
	const k = 50
	const query = "parse the tokenizer config and build a vocabulary index"
	qTokens := bm25.Tokenize(query)
	qVec := m.Encode(query)
	lexHits := bx.TopK(qTokens, k)
	denseHits := fx.Query(qVec, k)
	b.Logf("corpus: %d chunks, k=%d, lex hits %d, dense hits %d", len(chunks), k, len(lexHits), len(denseHits))

	b.Run("bm25Tokenize", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkTokens = bm25.Tokenize(query)
		}
	})
	b.Run("embedEncode", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkVec = m.Encode(query)
		}
	})
	b.Run("bm25TopK", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkResults = bx.TopK(qTokens, k)
		}
	})
	b.Run("flatI8Query", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			sinkHits = fx.Query(qVec, k)
		}
	})
	b.Run("fuseRRF", func(b *testing.B) {
		b.ReportAllocs()
		lex := fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc })
		dense := fuse.Keys(denseHits, func(h ann.Hit) int { return h.Index })
		for b.Loop() {
			sinkFused = fuse.RRF(60, lex, dense)
		}
	})
	b.Run("sum", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			qt := bm25.Tokenize(query)
			qv := m.Encode(query)
			lh := bx.TopK(qt, k)
			dh := fx.Query(qv, k)
			sinkFused = fuse.RRF(60,
				fuse.Keys(lh, func(r bm25.Result) int { return r.Doc }),
				fuse.Keys(dh, func(h ann.Hit) int { return h.Index }))
		}
	})
}

// Sinks keep the compiler from eliding the work being measured.
var (
	sinkChunks  []chunk.Chunk
	sinkTokens  []string
	sinkIndex   *bm25.Index
	sinkVec     []float32
	sinkVecs    [][]float32
	sinkFlatI8  *ann.FlatI8
	sinkBlob    []byte
	sinkResults []bm25.Result
	sinkHits    []ann.Hit
	sinkFused   []fuse.Result[int]
)

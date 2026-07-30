package treesitter

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/chunk"
)

// loadFixture reads a real source file from this repo, relative to the repo
// root. It mirrors chunk/regex's benchmark exactly — same three fixtures — so a
// benchstat run can diff regex-vs-treesitter per-language throughput on
// identical input, which was always this file's stated intent.
//
// Both original paths were dead and skipped silently (perf-campaign item 1):
// `../../search/index.go` is a leftover `ken` path, and
// `../../../testdata/repo/…` resolves ABOVE the repo root. It now fails on a
// missing fixture rather than reporting a green benchmark that never ran.
//
// This package is a separate module, so it cannot import the regex package's
// generator; the TypeScript fixture below is duplicated deliberately and must
// stay byte-identical to it for the comparison to mean anything.
func loadFixture(b *testing.B, rel ...string) []byte {
	b.Helper()
	path := filepath.Join(append([]string{"..", ".."}, rel...)...)
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatalf("in-repo fixture %s is missing: %v", path, err)
	}
	return data
}

// typescriptFixture must stay byte-identical to chunk/regex's copy — see
// loadFixture. There is no .ts file checked into this repository and the
// chunker splits on syntax, so this is generated with real structure.
func typescriptFixture(n int) []byte {
	var b strings.Builder
	b.WriteString("import { Widget } from './widget';\nimport type { Config } from './config';\n\n")
	for i := range n {
		fmt.Fprintf(&b, `
export interface Shape%d {
  id: string;
  size: number;
}

export class Renderer%d {
  private cache: Map<string, Shape%d> = new Map();

  constructor(private readonly cfg: Config) {}

  public render(shape: Shape%d): string {
    if (this.cache.has(shape.id)) {
      return this.cache.get(shape.id)!.id;
    }
    this.cache.set(shape.id, shape);
    return shape.id;
  }
}

export function makeShape%d(id: string, size: number): Shape%d {
  return { id, size };
}
`, i, i, i, i, i, i)
	}
	return []byte(b.String())
}

func benchChunk(b *testing.B, lang string, source []byte) {
	c := New()
	// Warm up: first call materialises the parser pool entry for this
	// language (gotreesitter allocates a fresh parser on first use).
	// We want to measure steady-state cAST cost, not first-call pool
	// init, so do one call before ResetTimer.
	if _, err := c.Chunk(source, lang, chunk.DefaultChunkSize); err != nil {
		b.Fatalf("warm-up Chunk: %v", err)
	}
	b.SetBytes(int64(len(source)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := c.Chunk(source, lang, chunk.DefaultChunkSize)
		if err != nil {
			b.Fatalf("Chunk: %v", err)
		}
	}
}

func BenchmarkCAST_Go(b *testing.B) {
	benchChunk(b, "go", loadFixture(b, "ann", "hnsw.go"))
}

func BenchmarkCAST_TypeScript(b *testing.B) {
	benchChunk(b, "typescript", typescriptFixture(40))
}

func BenchmarkCAST_Python(b *testing.B) {
	benchChunk(b, "python", loadFixture(b, "scripts", "encoder_model.py"))
}

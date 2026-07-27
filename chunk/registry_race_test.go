package chunk

import (
	"sync"
	"testing"
)

// stubChunker is a minimal Chunker for the registry race test.
type stubChunker struct{}

func (stubChunker) Chunk(_ []byte, _ string, _ int) ([]Chunk, error) { return nil, nil }
func (stubChunker) SupportedLanguages() []string                     { return nil }
func (stubChunker) Name() string                                     { return "stub" }

// TestRegistry_concurrentRegisterAndGet is the -race regression for AUDIT #17:
// Register (documented runtime-callable — a config-reload handler, a custom
// chunker) writing the package map while Get reads it on every ChunkFile is an
// unguarded `concurrent map read and map write` fatal. Run under -race; without
// the mutex this crashes the process (not merely a race warning).
func TestRegistry_concurrentRegisterAndGet(t *testing.T) {
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			for j := range 50 {
				Register(string(rune('a'+i))+string(rune('0'+j%10)), stubChunker{})
			}
		}(i)
		go func() {
			defer wg.Done()
			for range 50 {
				_, _ = Get("line")
				_ = Names()
			}
		}()
	}
	wg.Wait()
}

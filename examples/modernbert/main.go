// Command modernbert is a standalone example built on the ModernBERT encoder
// family: hotchpotch/bekko-embedding-v1-a8m and cl-nagoya/ruri-v3-30m as
// embedders, and cross-encoder/ettin-reranker-17m-v1 as a reranker.
//
// Download the checkpoints from the repository root:
//
//	uvx --from huggingface_hub hf download hotchpotch/bekko-embedding-v1-a8m \
//	    --local-dir testdata/bekko-embedding-v1-a8m
//	uvx --from huggingface_hub hf download cl-nagoya/ruri-v3-30m \
//	    --local-dir testdata/ruri-v3-30m
//	uvx --from huggingface_hub hf download cross-encoder/ettin-reranker-17m-v1 \
//	    --local-dir testdata/ettin-reranker-17m
//
// Then run:
//
//	go run ./examples/modernbert -text "こんにちは、世界"
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/encoder"
)

const (
	ruriRepo  = "cl-nagoya/ruri-v3-30m"
	ruriDir   = "./testdata/ruri-v3-30m"
	bekkoRepo = "hotchpotch/bekko-embedding-v1-a8m"
	bekkoDir  = "./testdata/bekko-embedding-v1-a8m"
	ettinRepo = "cross-encoder/ettin-reranker-17m-v1"
	ettinDir  = "./testdata/ettin-reranker-17m"
)

func main() {
	text := flag.String("text", "The quick brown fox jumps over the lazy dog.", "text to embed and rerank against")
	flag.Parse()

	emb, err := loadEmbedder(bekkoRepo, bekkoDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "modernbert: load bekko: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := emb.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "close bekko: %v\n", err)
		}
	}()

	v, err := emb.Encode(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("bekko embedding: dim %d, first 5: %v\n", len(v), v[:min(5, len(v))])

	emb2, err := loadEmbedder(ruriRepo, ruriDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "modernbert: load ruri: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := emb2.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "close ruri: %v\n", err)
		}
	}()

	v2, err := emb2.Encode(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("ruri embedding: dim %d, first 5: %v\n", len(v2), v2[:min(5, len(v2))])

	rr, err := loadReranker(ettinRepo, ettinDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "modernbert: load ettin: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := rr.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "close ettin: %v\n", err)
		}
	}()

	for _, doc := range []string{
		"The fox is a fast animal.",
		"Quantization shrinks neural nets.",
	} {
		s, err := rr.Score(*text, doc)
		if err != nil {
			fmt.Fprintf(os.Stderr, "score: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("score(%q | %q) = %.4f\n", *text, doc, s)
	}
}

// loadEmbedder wraps LoadModernBERT with a download hint when the checkpoint
// directory is absent, so the first-run failure points directly at the fix.
func loadEmbedder(repo, dir string) (*encoder.ModernBERT, error) {
	m, err := encoder.LoadModernBERT(dir)
	if err != nil {
		downloadHint(repo, dir)
	}
	return m, err
}

// loadReranker is the corresponding first-run wrapper for the Ettin reranker.
func loadReranker(repo, dir string) (*encoder.ModernBERTCrossEncoder, error) {
	m, err := encoder.LoadModernBERTCrossEncoder(dir)
	if err != nil {
		downloadHint(repo, dir)
	}
	return m, err
}

func downloadHint(repo, dir string) {
	if _, err := os.Stat(dir); err == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "modernbert: no checkpoint at %s — download it first:\n"+
		"  uvx --from huggingface_hub hf download %s \\\n"+
		"      --local-dir %s\n", dir, repo, dir)
	os.Exit(1)
}

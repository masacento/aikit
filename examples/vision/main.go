// Command vision is the end-to-end vision+text hybrid retrieval pipeline in one
// file: a corpus that mixes CODE CHUNKS and IMAGES, searched together.
//
// The README's capability matrix calls aikit "the only cgo-free image embedder"
// here, specifically for "image→image similarity and image-as-document indexing" —
// deliberately NOT cross-modal (text query → image result by pixel content), since
// aikit has no joint text/image embedding space (no CLIP-style text tower pairs
// with the vision package's SigLIP/ViT tower). This example demonstrates exactly
// the two claimed capabilities, composed the way a real mixed corpus would use
// them:
//
//  1. image-as-document indexing: every image gets a short caption, and that
//     caption is just another chunk.Chunk — embedded (embed.Model2Vec) and
//     BM25-indexed alongside the code snippets, in the SAME fused ranked list
//     (fuse.RRF, exactly as examples/rag does for text-only). A text query can
//     therefore surface an image via its caption.
//  2. image→image similarity: separately, every image gets a SigLIP embedding
//     (vision.LoadEncoder → Preprocess → Forward → mean-pool the patch sequence
//     → L2-normalize) indexed with its own ann.Flat. Landing on an image hit
//     pivots into "visually similar images" by pixel content, not caption text —
//     a second, independent retrieval leg the caption-only fused search can't do.
//
// It needs two local models (skipped-clean if absent, so `go build ./...` always
// compiles and `go run` without flags just prints guidance):
//
//	go run ./examples/vision \
//	    --embed-model  testdata/model \
//	    --vision-model testdata/siglip-model \
//	    --q "a picture with red in it"
//
// See the repo README and scripts/README.md's "Fetching testdata/siglip-model"
// section for how to fetch the vision checkpoint.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/chunk"
	_ "github.com/townsendmerino/aikit/chunk/regex" // registers the "regex" chunker via init()
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/fuse"
	"github.com/townsendmerino/aikit/vision"
)

// The same small code corpus examples/rag and examples/splade use, so all three
// examples are directly comparable on similar queries.
var textCorpus = []struct{ name, src string }{
	{"readlines.go", "func readLines(path string) ([]string, error) {\n\tf, err := os.Open(path)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\tdefer f.Close()\n\tvar lines []string\n\ts := bufio.NewScanner(f)\n\tfor s.Scan() {\n\t\tlines = append(lines, s.Text())\n\t}\n\treturn lines, s.Err()\n}"},
	{"json.go", "func parseConfig(b []byte) (*Config, error) {\n\tvar c Config\n\tif err := json.Unmarshal(b, &c); err != nil {\n\t\treturn nil, fmt.Errorf(\"parse config: %w\", err)\n\t}\n\treturn &c, nil\n}"},
	{"server.go", "func handler(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode(map[string]string{\"ok\": \"true\"})\n}"},
	{"math.go", "func fib(n int) int {\n\tif n < 2 {\n\t\treturn n\n\t}\n\treturn fib(n-1) + fib(n-2)\n}"},
}

// The image half of the corpus. Generated in-process (stdlib image/draw, no
// external asset files to fetch or commit) rather than photographs, so the
// example needs nothing beyond the SigLIP checkpoint itself — but each pattern
// is visually distinct enough that SigLIP's patch embeddings genuinely cluster
// by similarity (the two red variants nearest each other, the two blues nearest
// each other), not just by caption text.
var imageCorpus = []struct {
	name, caption string
	render        func(size int) image.Image
}{
	{"red-square.png", "a solid bright red square", func(n int) image.Image { return solid(n, color.RGBA{220, 30, 30, 255}) }},
	{"red-vertical.png", "a vertical split of dark red and light red", func(n int) image.Image {
		return vsplit(n, color.RGBA{150, 20, 20, 255}, color.RGBA{240, 120, 120, 255})
	}},
	{"blue-square.png", "a solid deep blue square", func(n int) image.Image { return solid(n, color.RGBA{30, 60, 200, 255}) }},
	{"blue-circle.png", "a blue circle centered on a white background", func(n int) image.Image { return circle(n, color.White, color.RGBA{40, 80, 210, 255}) }},
	{"checkerboard.png", "a black and white checkerboard pattern", func(n int) image.Image { return checker(n, color.Black, color.White) }},
}

func main() {
	embedDir := flag.String("embed-model", "", "dir with a Model2Vec checkpoint for embed.Load")
	visionDir := flag.String("vision-model", "", "dir with a SigLIP vision checkpoint for vision.LoadEncoder")
	query := flag.String("q", "a picture with red in it", "search query")
	topN := flag.Int("n", 5, "results to show")
	similarN := flag.Int("similar", 2, "visually-similar images to show per image hit")
	flag.Parse()

	if *embedDir == "" || *visionDir == "" {
		fmt.Println(`vision — end-to-end aikit vision+text hybrid retrieval example.

Needs two local models:
  --embed-model  <dir>   Model2Vec              (e.g. testdata/model)
  --vision-model <dir>   SigLIP vision encoder   (e.g. testdata/siglip-model)

Without them this just prints guidance; the pipeline code below is the point.`)
		return
	}

	em, err := embed.LoadFromFS(os.DirFS(*embedDir), ".")
	check(err, "load embed model")
	enc, err := vision.LoadEncoder(*visionDir, false)
	check(err, "load vision model")
	// Match the checkpoint's own trained resolution rather than assuming
	// Gemma 3's 896×896 — a plain SiglipVisionModel like siglip-base-patch16-224
	// trains at 224×224, and Forward rejects a pixel count that disagrees with
	// enc.Cfg.ImageSize.
	pcfg := vision.Config{
		Size:      enc.Cfg.ImageSize,
		Mean:      [3]float32{0.5, 0.5, 0.5},
		Std:       [3]float32{0.5, 0.5, 0.5},
		MaxPixels: 16 << 20,
	}

	// 1) CHUNK the code half — same as examples/rag/splade.
	var docs []chunk.Chunk // shared id space: docs[i] backs both dense and lexical indices below
	for _, d := range textCorpus {
		cs, err := chunk.ChunkFile("regex", d.name, []byte(d.src), 60)
		check(err, "chunk "+d.name)
		docs = append(docs, cs...)
	}
	firstImage := len(docs) // docs[firstImage:] are the image captions, one chunk.Chunk each

	// 2) RENDER + EMBED the image half. Each image becomes one more chunk.Chunk
	//    (its caption is the "text") AND one SigLIP vector kept in imgVecs, indexed
	//    by its own position (0..len(imageCorpus)) — a second, image-only id space
	//    for the "visually similar" pivot in step 6.
	imgVecs := make([][]float32, len(imageCorpus))
	for i, im := range imageCorpus {
		png := encodePNG(im.render(pcfg.Size))
		vec, err := embedImage(enc, pcfg, png)
		check(err, "embed image "+im.name)
		imgVecs[i] = vec
		docs = append(docs, chunk.Chunk{File: im.name, Text: im.caption, StartLine: 1, EndLine: 1})
	}
	imageAnn := ann.New(imgVecs)

	// 3) EMBED + index the UNIFIED corpus (code chunks + image captions) for dense
	//    search — identical to examples/rag from here on; a caption is just text.
	docTexts := make([]string, len(docs))
	for i, d := range docs {
		docTexts[i] = d.Text
	}
	dense := ann.New(em.EncodeBatch(docTexts, 0))

	// 3b) BM25 lexical index over the same unified corpus.
	lexDocs := make([][]string, len(docTexts))
	for i, t := range docTexts {
		lexDocs[i] = bm25.Tokenize(t)
	}
	lexical := bm25.Build(lexDocs)

	// 4) RETRIEVE + FUSE, same RRF as examples/rag — the fused list can contain
	//    either a code chunk or an image caption, ranked together.
	lexHits := lexical.TopK(bm25.Tokenize(*query), 20)
	denHits := dense.Query(em.Encode(*query), 20)
	fused := fuse.RRF(fuse.DefaultK,
		fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc }),
		fuse.Keys(denHits, func(h ann.Hit) int { return h.Index }),
	)

	// 5) Final ranked output.
	fmt.Printf("query: %q\n\n", *query)
	n := min(*topN, len(fused))
	for rank, r := range fused[:n] {
		d := docs[r.Key]
		if r.Key < firstImage {
			fmt.Printf("%d. %.4f  [text]  %s:%d-%d\n     %s\n", rank+1, r.Score, d.File, d.StartLine, d.EndLine, firstLine(d.Text))
			continue
		}
		// 6) IMAGE hit — pivot from "caption matched the query" to "images that
		//    look like this one", via the separate SigLIP-embedding index. This is
		//    the image→image similarity leg the caption-only fused search above
		//    cannot do on its own.
		imgIdx := r.Key - firstImage
		fmt.Printf("%d. %.4f  [image] %s\n     %q\n", rank+1, r.Score, d.File, d.Text)
		for _, h := range imageAnn.Query(imgVecs[imgIdx], *similarN+1) {
			if h.Index == imgIdx {
				continue // the image itself — always its own top match, not interesting
			}
			fmt.Printf("       ~ %.4f  %s\n", h.Score, imageCorpus[h.Index].name)
		}
	}
}

// embedImage runs the full vision path for one PNG: decode/resize/normalize,
// the SigLIP forward, mean-pool the patch sequence into one vector, L2-normalize
// (ann.Flat requires unit vectors, same contract embed.Encode's text vectors meet).
//
// Mean-pooling is a deliberate, documented choice: Forward returns the raw patch
// sequence (last_hidden_state) — the vision package is the ViT trunk a VLM's
// projector consumes, with no attention-pooling head of its own. Mean pooling is
// the standard general-purpose reduction for ViT patch features into one
// similarity-comparable vector when no trained pooling head is present.
func embedImage(enc *vision.Encoder, cfg vision.Config, pngBytes []byte) ([]float32, error) {
	pv, err := vision.Preprocess(pngBytes, cfg)
	if err != nil {
		return nil, err
	}
	hidden, err := enc.Forward(pv.Data)
	if err != nil {
		return nil, err
	}
	D := enc.Cfg.HiddenSize
	np := len(hidden) / D
	mean := make([]float32, D)
	for p := 0; p < np; p++ {
		row := hidden[p*D : (p+1)*D]
		for j, v := range row {
			mean[j] += v
		}
	}
	for j := range mean {
		mean[j] /= float32(np)
	}
	return embed.L2Normalize(mean), nil
}

// --- synthetic image generation (stdlib only, no external assets) ---

func solid(n int, c color.Color) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

func vsplit(n int, left, right color.Color) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if x < n/2 {
				img.Set(x, y, left)
			} else {
				img.Set(x, y, right)
			}
		}
	}
	return img
}

func checker(n int, c1, c2 color.Color) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	const cell = 16
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if (x/cell+y/cell)%2 == 0 {
				img.Set(x, y, c1)
			} else {
				img.Set(x, y, c2)
			}
		}
	}
	return img
}

func circle(n int, bg, fg color.Color) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	cx, cy, r := float64(n)/2, float64(n)/2, float64(n)/3
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			dx, dy := float64(x)-cx, float64(y)-cy
			if math.Hypot(dx, dy) <= r {
				img.Set(x, y, fg)
			} else {
				img.Set(x, y, bg)
			}
		}
	}
	return img
}

func encodePNG(img image.Image) []byte {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err) // in-process synthetic image; a codec failure here is a bug, not bad input
	}
	return buf.Bytes()
}

func firstLine(s string) string {
	for i, r := range s {
		if r == '\n' {
			return s[:i]
		}
	}
	return s
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "vision: %s: %v\n", what, err)
		os.Exit(1)
	}
}

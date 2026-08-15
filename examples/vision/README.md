# vision — vision+text hybrid retrieval

A corpus that mixes code chunks and images, searched together — the two
capabilities the `vision` package actually provides: **image-as-document
indexing** (each image's caption joins the fused dense+lexical search as just
another chunk) and **image→image similarity** (landing on an image hit pivots
into a separate SigLIP-embedding index). Deliberately NOT cross-modal
text→image search — aikit has no joint text/image embedding space. See
`main.go`'s doc comment for the full design rationale. Images are generated
in-process (stdlib `image/draw`), so no asset files to fetch.

## Run it

Needs two local models — a Model2Vec checkpoint and a SigLIP vision encoder:

```sh
go run ./examples/vision \
    --embed-model  testdata/model \
    --vision-model testdata/siglip-model \
    --q "a picture with red in it"
```

## Real output

The run below used `testdata/siglip-tiny` (the tiny, random-weight parity
fixture `scripts/oracle/pin_siglip_vision.py` generates — not a real
pretrained model). It's enough to prove the plumbing end to end — the
visual-similarity clustering already tracks color correctly even with random
higher-layer weights — but for a demo whose embeddings actually mean
something, fetch a real checkpoint first: see `scripts/README.md`'s "Fetching
`testdata/siglip-model`" section.

```
query: "a picture with red in it"

1. 0.0325  [image] red-square.png
     "a solid bright red square"
       ~ 0.9865  red-vertical.png
       ~ 0.3038  checkerboard.png
2. 0.0325  [image] red-vertical.png
     "a vertical split of dark red and light red"
       ~ 0.9865  red-square.png
       ~ 0.3169  checkerboard.png
3. 0.0317  [image] blue-circle.png
     "a blue circle centered on a white background"
       ~ 0.5971  blue-square.png
       ~ 0.4422  checkerboard.png
4. 0.0308  [image] blue-square.png
     "a solid deep blue square"
       ~ 0.5971  blue-circle.png
       ~ 0.1542  checkerboard.png
5. 0.0297  [image] checkerboard.png
     "a black and white checkerboard pattern"
       ~ 0.4422  blue-circle.png
       ~ 0.3169  red-vertical.png
```

The `~` lines under an `[image]` hit are the image→image pivot — the top
matches from that image's own SigLIP embedding, not from caption text.

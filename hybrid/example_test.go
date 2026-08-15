package hybrid_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/hybrid"
)

// Compose a dense index and a lexical index (built however you like — this
// uses the plain in-memory ones) into one Retriever, and get back the fused
// ranking in one call instead of hand-wiring fuse.RRF + fuse.Keys yourself
// (see examples/rag for what that looks like written out).
func Example() {
	vecs := [][]float32{{1, 0}, {0, 1}}
	docs := [][]string{{"go", "channel"}, {"python", "generator"}}

	r := hybrid.New(ann.New(vecs), bm25.Build(docs))

	fused := r.Query([]float32{1, 0}, []string{"go"}, 10)
	fmt.Println(fused[0].Key)
	// Output: 0
}

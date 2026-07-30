package ann_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/townsendmerino/aikit/ann"
)

// ExampleFlatBinary shows the two-stage retriever: a Hamming prefilter over the
// whole corpus, then an exact rerank of the survivors. The API is Flat's, so it
// is a drop-in wherever Hit / Query(q, k) is what a caller consumes.
func ExampleFlatBinary() {
	// A corpus with structure — a few clusters of near-duplicates, which is what
	// any approximate index needs in order to have neighbors to find.
	rng := rand.New(rand.NewSource(38))
	const dim = 128
	var vecs [][]float32
	var centers [][]float32
	for range 20 {
		c := make([]float32, dim)
		for j := range c {
			c[j] = float32(rng.NormFloat64())
		}
		centers = append(centers, normalize(c))
		for range 250 {
			v := make([]float32, dim)
			for j := range v {
				v[j] = c[j] + 0.4*float32(rng.NormFloat64())
			}
			vecs = append(vecs, normalize(v))
		}
	}

	exact := ann.New(vecs)
	approx := ann.NewFlatBinary(vecs)

	// Recall over one query per cluster, against the exact scan. Membership is
	// approximate — that is the trade — so this is the number to watch, and
	// NewFlatBinaryOverquery is the knob that moves it.
	const k = 10
	agree, total := 0, 0
	for _, q := range centers {
		truth := map[int]bool{}
		for _, h := range exact.Query(q, k) {
			truth[h.Index] = true
		}
		for _, h := range approx.Query(q, k) {
			if truth[h.Index] {
				agree++
			}
			total++
		}
	}

	// Scores, unlike membership, are exact: the survivors are rescored with the
	// same float32 dot product Flat uses.
	q := centers[7]
	got, want := approx.Query(q, 1), exact.Query(q, 1)

	fmt.Printf("indexed %d vectors, overquery %d\n", approx.Len(), approx.Overquery())
	fmt.Printf("recall@%d over %d queries: %.2f\n", k, len(centers), float64(agree)/float64(total))
	fmt.Printf("top hit agrees, with an identical score: %v\n",
		got[0].Index == want[0].Index && math.Abs(got[0].Score-want[0].Score) < 1e-12)

	// Output:
	// indexed 5000 vectors, overquery 16
	// recall@10 over 20 queries: 0.90
	// top hit agrees, with an identical score: true
}

func normalize(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

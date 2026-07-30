package ann

import (
	"fmt"
	"math/rand"
	"slices"
	"sort"
	"testing"
)

// BenchmarkSortSites isolates the mechanism behind A5's non-fuse half:
// sort.Slice / sort.SliceStable swap through reflect.Swapper, while
// slices.SortFunc swaps typed values directly.
//
// It exists because the change measured INSIDE NOISE where it actually runs, and
// this benchmark is what separates "the mechanism is fake" from "the mechanism
// is real but the stage is too small". Isolated, it is real — 2.95× at n=50.
// In a controlled A/B on the W2 query path the same change moved `FlatI8.Query`
// −2.9% and `bm25.TopK` +2.5%: opposite directions, both a hair outside the
// box's drift floor, i.e. nothing.
//
// The gap has a cause worth remembering. This fixture sorts RANDOM data with
// many ties; the real sites sort the array `topk.Selector.Result` hands back,
// which is a heap and therefore already partially ordered. pdqsort and
// reflect-based quicksort respond very differently to partially-ordered input,
// so a microbenchmark on random data does not predict either one on this input.
// See docs/internal/measuring-performance.md §1.4.
func BenchmarkSortSites(b *testing.B) {
	for _, n := range []int{10, 50, 200, 1000, 10000} {
		rng := rand.New(rand.NewSource(int64(n)))
		src := make([]Hit, n)
		for i := range src {
			src[i] = Hit{Index: i, Score: float64(rng.Intn(n/2 + 1))}
		}
		buf := make([]Hit, n)
		b.Run(fmt.Sprintf("n%d/sortSlice", n), func(b *testing.B) {
			for b.Loop() {
				copy(buf, src)
				sort.Slice(buf, func(a, c int) bool {
					if buf[a].Score != buf[c].Score {
						return buf[a].Score > buf[c].Score
					}
					return buf[a].Index < buf[c].Index
				})
			}
			sinkHitSort = buf
		})
		b.Run(fmt.Sprintf("n%d/slicesSortFunc", n), func(b *testing.B) {
			for b.Loop() {
				copy(buf, src)
				slices.SortFunc(buf, hitCmp)
			}
			sinkHitSort = buf
		})
	}
}

var sinkHitSort []Hit

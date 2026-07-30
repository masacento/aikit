package ann

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/townsendmerino/aikit/topk"
)

// TestFlatQuery_shardedMatchesSerial is the gate for perf-campaign item 16. The
// sharded scan must return EXACTLY what the serial scan returns — same indices,
// same scores, same order — because Flat is the exact retriever and callers
// compare its output against HNSW's approximation.
//
// The interesting case is ties. Flat's contract is "k highest by score, ties by
// ascending index", which the serial path gets from scanning in index order with
// topk's strict-> rule (first-seen wins). A sharded merge only reproduces that
// if the merge re-applies the same total order — so the adversarial input here
// is vectors that all score IDENTICALLY, where every returned index is decided
// purely by tie-breaking.
func TestFlatQuery_shardedMatchesSerial(t *testing.T) {
	for _, tc := range []struct {
		name string
		n, d int
		mode string
	}{
		{"random_5k_x128", 5000, 128, "random"},
		{"random_20k_x64", 20000, 64, "random"},
		{"all_identical_5k", 5000, 128, "identical"}, // every score ties
		{"two_score_levels", 8000, 96, "twolevel"},   // massive tie groups
		{"tiny_below_threshold", 100, 8, "random"},   // stays serial
		{"n_not_divisible", 4097, 96, "random"},      // ragged last shard
	} {
		f, q := buildFlatCase(tc.n, tc.d, tc.mode)
		for _, k := range []int{1, 5, 10, 64} {
			if k >= tc.n {
				continue
			}
			// Serial reference: force workers to 1 by using the filter-free
			// scan directly, which is what flatQueryWorkers returns 1 for.
			want := sortedHits(scanTopK(q, f.vecs, 0, k, nil))
			got := sortedHits(f.queryShards(q, k, 8, nil))
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s k=%d: sharded result differs from serial\n got %v\nwant %v",
					tc.name, k, got, want)
				continue
			}
			// And the public entry point, which picks its own fan-out.
			pub := f.Query(q, k)
			if !reflect.DeepEqual(hitsOf(pub), want) {
				t.Errorf("%s k=%d: Query() differs from serial\n got %v\nwant %v",
					tc.name, k, hitsOf(pub), want)
			}
		}
	}
}

// TestFlatQuery_shardWidthIsInert checks the result does not depend on HOW MANY
// shards were used — a property the merge must have and a plain determinism test
// would not catch, since any single width is self-consistent.
func TestFlatQuery_shardWidthIsInert(t *testing.T) {
	f, q := buildFlatCase(6000, 96, "twolevel")
	const k = 17
	want := sortedHits(scanTopK(q, f.vecs, 0, k, nil))
	for _, w := range []int{1, 2, 3, 5, 8, 16, 64, 6000} {
		got := sortedHits(f.queryShards(q, k, w, nil))
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("workers=%d: result differs from serial\n got %v\nwant %v", w, got, want)
		}
	}
}

// TestFlatQueryFilter_shardedMatchesSerial is the exactness gate for the
// FILTERED path, which now shards too. The filter interacts with the per-shard
// threshold — each shard rejects candidates against its own running k-th score —
// so this is not implied by the unfiltered gate and is tested separately, on the
// same adversarial tie inputs.
func TestFlatQueryFilter_shardedMatchesSerial(t *testing.T) {
	for _, mode := range []string{"random", "identical", "twolevel"} {
		f, q := buildFlatCase(6000, 96, mode)
		for _, sel := range []struct {
			name string
			keep func(int) bool
		}{
			{"all", func(int) bool { return true }},
			{"none", func(int) bool { return false }},
			{"evens", func(id int) bool { return id%2 == 0 }},
			{"sparse", func(id int) bool { return id%997 == 0 }},
			{"prefix", func(id int) bool { return id < 50 }},
			{"suffix", func(id int) bool { return id >= 5950 }},
		} {
			for _, k := range []int{1, 10, 64} {
				want := sortedHits(scanTopK(q, f.vecs, 0, k, sel.keep))
				got := sortedHits(f.queryShards(q, k, 8, sel.keep))
				if !reflect.DeepEqual(got, want) {
					t.Errorf("%s/%s k=%d: sharded filtered result differs from serial\n got %v\nwant %v",
						mode, sel.name, k, got, want)
				}
				if !reflect.DeepEqual(hitsOf(f.QueryFilter(q, k, sel.keep)), want) {
					t.Errorf("%s/%s k=%d: QueryFilter differs from serial", mode, sel.name, k)
				}
			}
		}
	}
}

// TestFlatQuery_fanOutIsReached is the vacuity guard for every sharding test
// here: if the gate stopped firing they would all pass while measuring nothing.
func TestFlatQuery_fanOutIsReached(t *testing.T) {
	f, _ := buildFlatCase(20000, 64, "random")
	if w := flatQueryWorkers(len(f.vecs), f.dim); w <= 1 {
		t.Fatalf("query did NOT fan out (workers=%d) — the sharding tests prove nothing", w)
	}
	if w := flatQueryWorkers(100, 8); w != 1 {
		t.Errorf("a tiny index fanned out to %d workers; the threshold is not being applied", w)
	}
}

func buildFlatCase(n, d int, mode string) (*Flat, []float32) {
	rng := rand.New(rand.NewSource(int64(n*31 + d)))
	vecs := make([][]float32, n)
	unit := make([]float32, d)
	unit[0] = 1
	for i := range vecs {
		switch mode {
		case "identical":
			v := make([]float32, d)
			copy(v, unit)
			vecs[i] = v
		case "twolevel":
			v := make([]float32, d)
			if i%2 == 0 {
				copy(v, unit)
			} else {
				v[1] = 1
			}
			vecs[i] = v
		default:
			v := make([]float32, d)
			var norm float64
			for j := range v {
				v[j] = float32(rng.NormFloat64())
				norm += float64(v[j]) * float64(v[j])
			}
			inv := float32(1 / math.Sqrt(norm))
			for j := range v {
				v[j] *= inv
			}
			vecs[i] = v
		}
	}
	f := &Flat{vecs: vecs, dim: d}
	q := make([]float32, d)
	copy(q, unit)
	return f, q
}

type hit struct {
	Index int
	Score float64
}

func sortedHits(items []topk.ItemWithScore[int]) []hit {
	out := make([]hit, len(items))
	for i, it := range items {
		out[i] = hit{Index: it.Item, Score: it.Score}
	}
	return out
}

func hitsOf(hs []Hit) []hit {
	out := make([]hit, len(hs))
	for i, h := range hs {
		out[i] = hit(h)
	}
	return out
}

// TestFlatI8_pooledScratchIsInert guards the hazard item 4 introduces: the score
// buffer and the kernel Workspace are now pooled, so a query receives whatever
// the previous query left behind. Any element the kernel fails to overwrite
// would silently score a document using a stale value.
//
// Poisoning the pool and comparing against a known-good result is the only way
// to see it — running the same query twice would agree, and agree wrongly.
func TestFlatI8_pooledScratchIsInert(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const n, d = 3000, 96
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, d)
		var norm float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		inv := float32(1 / math.Sqrt(norm))
		for j := range v {
			v[j] *= inv
		}
		vecs[i] = v
	}
	f := NewFlatI8(vecs)
	q := vecs[7]

	want := f.Query(q, 10)

	// Poison every scratch the pool can hand back.
	held := make([]*flatI8Scratch, 0, 32)
	for range 32 {
		sc := flatI8ScratchPool.Get().(*flatI8Scratch)
		if cap(sc.dst) < n {
			sc.dst = make([]float32, n)
		}
		for i := range sc.dst[:n] {
			sc.dst[i] = 12345
		}
		held = append(held, sc)
	}
	for _, sc := range held {
		flatI8ScratchPool.Put(sc)
	}

	got := f.Query(q, 10)
	if !reflect.DeepEqual(hitsOf(got), hitsOf(want)) {
		t.Fatalf("query after a poisoned pool differs\n got %v\nwant %v", hitsOf(got), hitsOf(want))
	}
	// Vacuity: the poison must actually be reachable, i.e. the pool is used.
	sc := flatI8ScratchPool.Get().(*flatI8Scratch)
	if cap(sc.dst) == 0 {
		t.Error("pool never received a sized scratch — the query path is not using it")
	}
	flatI8ScratchPool.Put(sc)
}

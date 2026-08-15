package ann

import (
	"runtime"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

// FlatBinary is Flat with a two-stage scan: a 1-bit sign-quantized Hamming
// PREFILTER over the whole corpus, then an EXACT float32 rerank of the
// survivors. It exposes the same Hit / Query(q, k) shape as Flat and FlatI8.
//
// What it trades. This is the first index in the package whose result set is
// APPROXIMATE — Flat and FlatI8 both score every vector and are exact over
// their respective score functions. Here the true top-k can be missed outright
// if the prefilter does not rank it in the top Overquery·k by Hamming distance.
// The scores that come back are exact (same kernel Flat uses); the membership
// is not.
//
// What it buys. Only the SCAN shrinks, not the index: the float32 vectors are
// still held for the rerank, and the codes add dim/8 bytes per vector on top —
// about 3% at dim 768. If it is memory you need, FlatI8 is the sibling that
// gives you a quarter of the bytes. What this gives you is a first stage that
// reads 96 B per candidate instead of 3072 B, measured at 14× the throughput of
// the exact scan on this corpus shape (perf-campaign item 38).
//
// It composes with embed.Truncate rather than competing with it: truncating to
// dim d shrinks both stages, and the prefilter's cost is linear in d.
//
// One shape where it does NOT pay: a very small candidate set. The prefilter
// materializes a distance per document and makes a second pass over them, which
// at overquery·k below roughly 100 on a 200k corpus costs more than the
// heap-based selection the filtered path uses — measured at 877 µs against
// 614 µs for k=10, overquery 4 (BenchmarkFlatBinaryPrefilterPaths). Above that
// the counting sort wins by 1.4× to 6×, and it is flat in overquery where the
// heap is not. The dispatch is on the presence of a filter alone rather than on
// a fitted candidate-count threshold, because that crossover was measured at
// one corpus size on one machine and scales with n.
//
// Like Flat, a built *FlatBinary is read-only and Query is safe to call
// concurrently; the constructors are the single-threaded builders.
type FlatBinary struct {
	// exact backs the rerank and the k<=0 / k>=Len passthrough. Sharing it means
	// the exact stage is literally Flat's scan, not a second copy of it.
	exact *Flat

	codes []uint64 // [n*words] packed sign bits of (vec - center)
	// center is the corpus mean, subtracted before the sign is taken.
	//
	// Real embedding sets have a common component: on the Model2Vec corpus, 25
	// of 256 dimensions have 90%+ of the vectors agreeing in sign. Those
	// dimensions contribute a near-constant to every distance and so carry
	// almost no information about which document is which. Centering moves each
	// dimension's split point to where the corpus actually divides.
	//
	// Measured, not assumed: recall@10 of 0.9625 centered against 0.9375
	// uncentered (TestFlatBinary_centeringIsWhatMakesRealCorporaWork). Real but
	// modest — smaller than the effect of one step of overquery — and it costs
	// one subtract per query plus dim floats of storage, so it stays on. The
	// synthetic gates cannot see it at all: their cluster centers are drawn from
	// a zero-mean Gaussian, so the corpus mean is already ~0.
	center []float32

	n, dim, words int
	over          int
}

// DefaultOverquery is the prefilter's candidate multiplier: Query(q, k) reranks
// the DefaultOverquery·k nearest by Hamming distance. Higher trades scan time
// for recall.
//
// 16 is where the recall curve reaches 1.0 — on the real Model2Vec corpus
// (0.5625 / 0.7250 / 0.8375 / 0.9625 / 1.0000 at overquery 1 / 2 / 4 / 8 / 16,
// logged by TestFlatBinary_recallReal_Model2Vec) and on the synthetic clusters
// alike. It is set there rather than at 8 because the prefilter's counting-sort
// selection is FLAT in this knob: at d=768, N=200k, k=10 the query costs 877 µs
// at overquery 4 and 901 µs at 16, so the last 4 points of recall are worth
// about 3% of the query. That was not true of the first implementation — see
// prefilterHist — and it is the measurement to redo before changing this.
//
// It is a starting point, not a universal constant. A corpus with tighter
// clusters wants more; measure on your own data with the same recall-versus-
// overquery sweep those two tests run.
const DefaultOverquery = 16

// NewFlatBinary builds a binary-prefiltered index over vecs (each assumed
// L2-normalized, the package invariant) with DefaultOverquery. Vectors are used
// by reference, not copied.
func NewFlatBinary(vecs [][]float32) *FlatBinary {
	return newFlatBinary(vecs, DefaultOverquery, true)
}

// NewFlatBinaryOverquery is NewFlatBinary with an explicit candidate
// multiplier. overquery <= 1 means "rerank exactly k candidates", which makes
// the result the prefilter's own ranking and is almost never what you want;
// raise it until recall@k stops improving on your data.
func NewFlatBinaryOverquery(vecs [][]float32, overquery int) *FlatBinary {
	return newFlatBinary(vecs, overquery, true)
}

// newFlatBinary is the shared builder. center is a parameter rather than always
// true so the recall test can measure what centering is worth instead of
// asserting it; there is no reason for a caller to turn it off.
func newFlatBinary(vecs [][]float32, overquery int, center bool) *FlatBinary {
	f := &FlatBinary{exact: New(vecs), n: len(vecs), over: max(overquery, 1)}
	if f.n == 0 {
		return f
	}
	f.dim = len(vecs[0])
	f.words = linalg.PackedWords(f.dim)
	if center {
		// float64 accumulation: at n in the millions a float32 running sum loses
		// the low bits of the mean, which is exactly the quantity the sign test
		// is sensitive to.
		acc := make([]float64, f.dim)
		used := 0
		for _, v := range vecs {
			if len(v) != f.dim {
				continue // ragged input; the code for it is left all-zero below
			}
			used++
			for i, x := range v {
				acc[i] += float64(x)
			}
		}
		if used > 0 {
			f.center = make([]float32, f.dim)
			for i, s := range acc {
				f.center[i] = float32(s / float64(used))
			}
		}
	}

	f.codes = make([]uint64, f.n*f.words)
	row := make([]float32, f.dim)
	for i, v := range vecs {
		if len(v) != f.dim {
			continue
		}
		linalg.PackSignBitsRow(f.codes[i*f.words:(i+1)*f.words], f.centered(row, v))
	}
	return f
}

// centered writes v - center into dst and returns it (or v itself when the
// index is uncentered, avoiding the copy).
func (f *FlatBinary) centered(dst, v []float32) []float32 {
	if f.center == nil {
		return v
	}
	for i, x := range v {
		dst[i] = x - f.center[i]
	}
	return dst
}

// Len is the number of indexed vectors.
func (f *FlatBinary) Len() int { return f.n }

// Overquery is the candidate multiplier this index was built with.
func (f *FlatBinary) Overquery() int { return f.over }

// Query returns approximately the k highest cosine-similarity vectors to q,
// descending, ties broken by ascending index. k <= 0 or k >= Len returns all,
// sorted — and is EXACT, since there is nothing for a prefilter to discard.
//
// The scores are exact float32 dot products, so a hit's Score means the same
// thing it does for Flat. Which hits come back can differ: see the type's doc.
func (f *FlatBinary) Query(q []float32, k int) []Hit {
	return f.query(q, k, nil)
}

// QueryFilter is Query restricted to documents for which keep(id) is true. A
// nil keep is exactly Query.
//
// keep is applied in the PREFILTER, not after it, so the k results are the best
// live documents rather than whatever survives filtering a fixed candidate set
// — the difference matters when the filter is selective. As in
// Flat.QueryFilter, keep must be a pure predicate safe for concurrent use: the
// prefilter is sharded, and each shard's running threshold means the set of ids
// it asks about, and their order, differ from a serial scan.
func (f *FlatBinary) QueryFilter(q []float32, k int, keep func(id int) bool) []Hit {
	return f.query(q, k, keep)
}

func (f *FlatBinary) query(q []float32, k int, keep func(int) bool) []Hit {
	if f.n == 0 || len(q) != f.dim {
		return nil
	}
	cand := k * f.over
	// Two ways the prefilter has nothing to do: the caller wants everything, or
	// the candidate set is the whole corpus. Both defer to the exact scan, which
	// is faster than prefiltering and then reranking all of it.
	if k <= 0 || k >= f.n || cand >= f.n || cand < 0 /* overflow */ {
		return f.exact.query(q, k, keep)
	}

	sc := binScratchPool.Get().(*binScratch)
	defer binScratchPool.Put(sc)
	sc.qc = ensureU64(sc.qc, f.words)
	sc.qf = ensureF32b(sc.qf, f.dim)
	linalg.PackSignBitsRow(sc.qc, f.centered(sc.qf, q))

	ids := f.prefilter(sc.qc, sc, cand, keep)
	return f.rerank(sc, q, ids, k)
}

// prefilter returns the cand nearest ids by Hamming distance, ascending by id.
//
// Ascending is not incidental: the rerank pushes in that order, and topk's
// first-seen-wins tie rule then reproduces Flat's "ties broken by ascending
// index" exactly. Handing the rerank a distance-ordered list would break that
// tie-break for documents with equal dot products.
//
// Two implementations, and which one runs depends only on whether there is a
// filter. See prefilterHist for why the unfiltered case gets its own.
func (f *FlatBinary) prefilter(qc []uint64, sc *binScratch, cand int, keep func(int) bool) []int32 {
	if keep == nil {
		return f.prefilterHist(qc, sc, cand)
	}
	return f.prefilterHeap(qc, sc, cand, keep)
}

// prefilterHist selects the cand nearest by COUNTING SORT over the distance
// histogram, which is available here and almost nowhere else: a Hamming
// distance is an integer in [0, dim], so dim+1 counters describe the whole
// corpus exactly.
//
// It replaced a per-shard top-cand heap, and the reason was measured. The heap
// made the candidate multiplier expensive — at d=768, N=200k, k=10 a query cost
// 594 µs at overquery 4 and 2042 µs at overquery 32, a 3.4× swing on a knob
// that only controls RECALL. The scan does not change at all across that sweep;
// what changed was 16 shards each maintaining, allocating and merging a
// cand-sized heap. The histogram is O(n) with no per-shard structure
// proportional to cand, so overquery costs only the extra exact reranks it
// actually buys — which is what let DefaultOverquery be set where the recall
// curve flattens instead of where the heap stopped hurting.
//
// The result is exactly the same candidate set the heap produced: all documents
// strictly nearer than the threshold distance, then documents AT it in
// ascending id order until the budget is spent — which is "(distance asc, id
// asc), take cand" written out.
func (f *FlatBinary) prefilterHist(qc []uint64, sc *binScratch, cand int) []int32 {
	nb := f.dim + 1
	workers := binQueryWorkers(f.n, f.words)
	sc.dists = ensureU16(sc.dists, f.n)
	sc.hist = ensureI32(sc.hist, workers*nb)
	clear(sc.hist)

	if workers <= 1 {
		f.scanHist(qc, sc.dists, sc.hist[:nb], 0, f.n)
	} else {
		per := (f.n + workers - 1) / workers
		var wg sync.WaitGroup
		for w := range workers {
			lo := w * per
			if lo >= f.n {
				break
			}
			hi := min(lo+per, f.n)
			wg.Add(1)
			// Each worker owns a disjoint span of dists AND its own histogram
			// slice, so there is nothing to synchronize and nothing to allocate
			// per worker.
			go func(w, lo, hi int) {
				defer wg.Done()
				f.scanHist(qc, sc.dists, sc.hist[w*nb:(w+1)*nb], lo, hi)
			}(w, lo, hi)
		}
		wg.Wait()
		for w := 1; w < workers; w++ {
			for b := range nb {
				sc.hist[b] += sc.hist[w*nb+b]
			}
		}
	}

	// The threshold distance: the smallest t whose cumulative count reaches
	// cand. Everything nearer is taken outright; t itself is taken partially.
	cum, t := 0, 0
	for ; t < nb; t++ {
		if cum+int(sc.hist[t]) >= cand {
			break
		}
		cum += int(sc.hist[t])
	}
	budget := cand - cum

	ids := sc.ids[:0]
	for i, d := range sc.dists[:f.n] {
		switch {
		case int(d) < t:
			ids = append(ids, int32(i))
		case int(d) == t && budget > 0:
			ids = append(ids, int32(i))
			budget--
		}
	}
	sc.ids = ids
	return ids
}

// scanHist fills dists[lo:hi] and tallies hist, a block at a time so the
// distances are still in L1 when the histogram reads them back.
func (f *FlatBinary) scanHist(qc []uint64, dists []uint16, hist []int32, lo, hi int) {
	for base := lo; base < hi; base += binBlockRows {
		end := min(base+binBlockRows, hi)
		blk := dists[base:end]
		linalg.HammingRows(qc, f.codes[base*f.words:end*f.words], f.words, end-base, blk)
		for _, d := range blk {
			hist[d]++
		}
	}
}

// prefilterHeap is the filtered path: a per-shard top-cand selector, which
// calls keep only for candidates that could actually be retained.
//
// The histogram cannot serve this case without either calling keep for every
// document in the corpus — turning a cheap live-set lookup into n
// non-inlinable calls — or iterating thresholds upward until enough survivors
// appear. The heap already has the property that matters (keep is consulted
// O(cand·log n) times, not O(n)), so the filtered path keeps it and the two are
// gated against each other by TestFlatBinary_histAndHeapAgree.
func (f *FlatBinary) prefilterHeap(qc []uint64, sc *binScratch, cand int, keep func(int) bool) []int32 {
	workers := binQueryWorkers(f.n, f.words)
	if workers <= 1 {
		sc.parts = append(sc.parts[:0], nil)
		f.scanShard(qc, sc, 0, f.n, cand, keep, &sc.parts[0])
		return finishCandidates(sc.parts, cand, sc)
	}

	per := (f.n + workers - 1) / workers
	parts := make([][]topk.ItemWithScore[int32], workers)
	var wg sync.WaitGroup
	for w := range workers {
		lo := w * per
		if lo >= f.n {
			break
		}
		hi := min(lo+per, f.n)
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			// Each shard needs its OWN block buffer, so it takes its own scratch
			// from the pool. The packed query is passed by value rather than
			// stored on that scratch: a borrowed scratch goes back to the pool at
			// return, and stashing the parent's buffer in it would leave two
			// scratches sharing one array on some later query.
			s := binScratchPool.Get().(*binScratch)
			defer binScratchPool.Put(s)
			f.scanShard(qc, s, lo, hi, cand, keep, &parts[w])
		}(w, lo, hi)
	}
	wg.Wait()
	// Merging per-shard top-cand lists is exact under the same argument
	// Flat.queryShards makes: any element of the global top-cand under
	// (distance asc, id asc) is also in its own shard's top-cand, since a shard
	// is a subset and the order is total.
	return finishCandidates(parts, cand, sc)
}

// scanShard computes Hamming distances for rows [lo, hi) a block at a time and
// keeps the cand nearest.
//
// Blocked rather than one big distance array because the block buffer then
// stays in L1 between the kernel writing it and the selection loop reading it,
// and because a whole-corpus []uint16 would be a 2 MB per-query allocation at
// n = 10⁶ for no benefit.
func (f *FlatBinary) scanShard(qc []uint64, sc *binScratch, lo, hi, cand int, keep func(int) bool, out *[]topk.ItemWithScore[int32]) {
	sel := topk.New[int32](cand)
	// Score is NEGATED distance: topk keeps the highest, and nearest means
	// smallest. Distances are small integers exactly representable in float64,
	// so this is a relabeling, not an approximation.
	th := sel.Threshold()
	sc.dists = ensureU16(sc.dists, binBlockRows)
	for base := lo; base < hi; base += binBlockRows {
		end := min(base+binBlockRows, hi)
		rows := end - base
		linalg.HammingRows(qc, f.codes[base*f.words:end*f.words], f.words, rows, sc.dists[:rows])
		for j, d := range sc.dists[:rows] {
			s := -float64(d)
			if s > th && (keep == nil || keep(base+j)) {
				sel.Push(int32(base+j), s)
				th = sel.Threshold()
			}
		}
	}
	*out = sel.Result()
}

// finishCandidates merges the shards' selections, truncates to cand under
// (distance asc, id asc), and returns the ids in ASCENDING ID order.
func finishCandidates(parts [][]topk.ItemWithScore[int32], cand int, sc *binScratch) []int32 {
	merged := sc.merged[:0]
	for _, p := range parts {
		merged = append(merged, p...)
	}
	sc.merged = merged
	slices.SortFunc(merged, topk.ItemCmp[int32])
	if len(merged) > cand {
		merged = merged[:cand]
	}
	ids := sc.ids[:0]
	for _, m := range merged {
		ids = append(ids, m.Item)
	}
	sc.ids = ids
	slices.Sort(ids)
	return ids
}

// rerank scores the candidates exactly and returns the k best.
//
// It gathers the candidate vectors and runs them through scanFlat — the same
// kernel Flat.Query uses — rather than a Dot per candidate, so a document's
// score here is computed the way Flat computes it. (A document that lands in
// the ragged tail of one grouping and the 8-wide body of the other can still
// differ in the last bits; that is the sub-ULP variation Flat's own package doc
// already describes, not a second source of error.)
//
// The gather itself is not worth optimizing: at k=10, overquery 8 the rerank is
// 80 candidates against a prefilter that just read the whole corpus.
func (f *FlatBinary) rerank(sc *binScratch, q []float32, ids []int32, k int) []Hit {
	gath := sc.gather[:0]
	for _, id := range ids {
		gath = append(gath, f.exact.vecs[id])
	}
	sc.gather = gath

	sel := topk.New[int](k)
	th := sel.Threshold()
	scanFlat(q, gath, func(i int, score float64) {
		if score > th {
			sel.Push(int(ids[i]), score)
			th = sel.Threshold()
		}
	})
	items := sel.Result()
	slices.SortFunc(items, topk.ItemCmp[int])
	hits := make([]Hit, len(items))
	for j, s := range items {
		hits[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return hits
}

// binBlockRows is the Hamming scan's blocking factor: 1024 rows of uint16 is a
// 2 KB distance buffer, comfortably L1-resident alongside the codes being
// streamed through it.
const binBlockRows = 1024

// binParallelThreshold is the scanned-word count at or above which sharding the
// prefilter pays for the goroutine spawn.
//
// It is NOT flatParallelThreshold converted to words. That constant fans out
// when the exact scan reaches roughly 80 µs on this box; matching that TIME —
// which is what the goroutine spawn has to be amortized against — takes ~14×
// more vectors here, because that is how much cheaper this scan is per vector.
// Time-matched, not swept, and amd64-measured.
const binParallelThreshold = 1 << 17

func binQueryWorkers(n, words int) int {
	if n < 2 || int64(n)*int64(words) < binParallelThreshold {
		return 1
	}
	return min(runtime.NumCPU(), n)
}

// binScratch is one query's reusable buffers. Pooled for the same reason
// flatI8Scratch is: the per-query allocation is otherwise proportional to the
// corpus, and a retrieval path is called in a loop.
type binScratch struct {
	qc     []uint64
	qf     []float32
	dists  []uint16
	hist   []int32
	merged []topk.ItemWithScore[int32]
	ids    []int32
	gather [][]float32
	parts  [][]topk.ItemWithScore[int32]
}

var binScratchPool = sync.Pool{New: func() any { return new(binScratch) }}

func ensureU64(b []uint64, n int) []uint64 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]uint64, n)
}

func ensureI32(b []int32, n int) []int32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]int32, n)
}

func ensureU16(b []uint16, n int) []uint16 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]uint16, n)
}

func ensureF32b(b []float32, n int) []float32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]float32, n)
}

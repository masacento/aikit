# Task (aikit): memoization audit — where caching a pure function actually pays

> **Companion to** `perf-campaign-2026-07-28.md`. Standalone — it repeats the
> context it needs. Cross-references to that doc's numbered items are marked
> **[campaign #N]**.
>
> **Constraints:** core stays pure-Go, no cgo. No public API breakage.

Memoization is a narrow tool. It needs three things at once: a **pure function**,
**repeated identical inputs**, and a **lookup cheaper than the recompute**. Most
of aikit's hot code is streaming numeric kernels over never-repeated data — dot
products across distinct vectors, transformer forwards over distinct token
sequences — so it fails condition (b) or (c) almost everywhere.

But there is one shape that recurs, and it is worth naming because it is easy to
miss in review:

> **A pure function of immutable model/index state, recomputed per call.**

Once a model or index is loaded, its config is frozen. Anything derived purely
from that config is a constant that the code is nevertheless rebuilding on every
`Encode`, every `Score`, every `Query`. Four such sites exist. Two are worth
doing, one is a trap, one is precomputation already tracked elsewhere.

---

## 0. Measurement caveats

Everything below was measured on an **x86 Xeon @2.8 GHz, 2 cores, Go 1.24.7** —
not on the M1 Pro that arbitrates every other decision in these docs. The repo
does not build in that sandbox (`go.mod` pins Go 1.26.5, toolchain download
blocked), so measurements come from standalone reproductions of the exact loop
bodies, not through the real build. Reproduction source is in the appendix — it
is self-contained and should be re-run on arm64 before committing to anything.

**Ratios between two Go loops transfer well. Absolute times do not.** One
specific distortion to keep in mind for §1: the WordPiece benchmark uses a
*synthesized* vocab, so the per-call cost is indicative only. The load-bearing
number there — the **98.1% repetition rate** — is vocab-independent.

---

## 1. `wordPiece` — the real one. 98% of calls are repeats.

**Where:** `embed/tokenize.go:527` (`func (t *Tokenizer) wordPiece(word string) []int32`),
called from `encodeSegment:270`.

**Why it's a candidate.** It is a pure function of `(word, vocab)`, and the vocab
is immutable after `LoadTokenizer`. **There is no cache today** — grep for
`cache`/`memo`/`sync.Map` in `embed/tokenize.go` returns nothing.

**Why it's expensive.** Greedy longest-match: for a word of L runes it probes up
to O(L²/2) prefixes, each one rebuilding a byte buffer (with the `##`
continuing-prefix on non-initial pieces) and hitting `t.vocab`. The pooled
`wpBufPool` (audit #11) already removed the *allocations* per probe, but not the
quadratic probe count. A 20-rune identifier can cost ~200 buffer builds and ~200
map lookups.

**Measured** on aikit's own Go source, 502,677 pre-tokenized words:

```
502,677 total words → 9,463 unique   =  98.1% of wordPiece calls are repeats
mean word length:  2.66 over all words,  8.84 over unique words

BenchmarkWordPiece_Cold   237.6 ms   5,816,370 B/op   569,186 allocs/op
BenchmarkWordPiece_Memo    63.3 ms     194,082 B/op     7,427 allocs/op
                           └ 3.75×                      └ 77× fewer allocs
```

`TestMemoEquiv` verifies the memoized path returns **identical ids across all
502,677 words**.

**The length asymmetry sharpens the win.** Unique words average 8.84 runes
against 2.66 for the corpus as a whole — i.e. the expensive long identifiers are
exactly the ones you pay for once and never again. The realized speedup is
therefore *better* than the raw 98.1% hit rate implies, which is why the measured
3.75× exceeds what a flat cost model would predict.

**Why this matters beyond the tokenizer.** The tokenizer is **~65% of
`embed.StaticModel.Encode`** (141 µs of tokenize against 76.5 µs of pooling for a
1.88 KB passage). So it is the dominant cost of bulk corpus embedding — the
378k-chunk workload `embed/model.go:179` calls out. It is *not* hot for the
`encoder` forward pass (~141 µs against a forward measured in hundreds of ms);
this is a Model2Vec / indexing-throughput item, not an inference item.

### Design notes

1. **Concurrency.** `Encode` is documented goroutine-safe — the `wpBufPool`
   comment at `:573-575` says so explicitly. An unsynchronized
   `map[string][]int32` would race. `sync.Map` is the wrong shape here (it
   optimizes for disjoint key sets per goroutine; this is a shared read-mostly
   set). Prefer a **sharded map** — N shards keyed by a cheap hash of the word,
   each behind its own `RWMutex` — so read contention stays flat as workers scale.
   `bm25`'s pooled-buffer precedent (ADR-027/028) is the in-repo pattern to
   follow for the shape, not the sync strategy.
2. **Returning cached slices is safe.** `encodeSegment:270` does
   `ids = append(ids, t.wordPiece(w)...)`, which copies the elements. No caller
   retains or mutates the returned slice. Verify this stays true if new callers
   appear.
3. ~~**Kill the remaining allocations.** Storing `[]int32` per entry is what leaves
   7,427 allocs. A flat `[]int32` arena plus `map[string][2]int32` (offset, len)
   makes it ~one allocation amortized, and improves locality on hit.~~
   **Tried and reverted (2026-07-30) — the premise was a misread.** Those ~9k
   allocs are `wordPieceCompute`'s own `out` slices, NOT the cache's storage: the
   per-entry design keeps `out` *as* the entry (zero-copy), so the arena can't
   remove them — it *copies* `out` in and discards it, making population ~3%
   WORSE (10,204 vs 9,865 mallocs on 8,983 distinct words). Steady-state hit
   throughput is identical (both return a view/slice, no alloc). The only real
   gain is ~4.7% retained compactness (4,708 vs 4,938 KB here; a few MB at the
   262k-word cap) — not worth the append-only-arena concurrency reasoning and the
   worse populate path. To actually cut the population allocs you'd have to compute
   *directly into* the arena, which entangles the pooled-buffer compute path and
   the differential-fuzz story for the same marginal memory win. Left as-is.
4. **Bound it.** 9,463 entries for this corpus is trivial, and for natural text
   the unique-word set converges near vocab size. But adversarial or multilingual
   input can grow it unboundedly. Either cap with a simple clock/random eviction
   (a miss costs only the original recompute), or bound by construction: skip
   caching words longer than `maxCharsPerWord`, which already short-circuit to
   `unkID` anyway.
5. **Numerics:** none. Token ids are exact. Gate with a differential fuzz against
   the uncached path — the same discipline `TestTokenize_AdversarialParity` uses.

**Win** ~3.75× on tokenization, ~1.9× on `embed.StaticModel.Encode`.
**Risk** low (concurrency design is the only real surface). **Effort** S–M.

### 1b. The same insight in `bm25` — string interning

`bm25/tokenize.go:209-225` — `lowerString`'s slow path ends in
`return string(*scratch)`, one allocation per token containing an uppercase byte.
Measured: **787 allocations for one 20.7 KB Go file** (`ann/hnsw.go`). Go source
is dense in CamelCase, so the "fast path returns a view into `text`" optimization
misses most identifiers.

Interning through a `map[string]string` in the pooled buffers **is** memoization,
and at 98% repetition it near-eliminates those allocations. It has a second-order
benefit the allocation count doesn't show: duplicate tokens across documents come
to share one backing array, shrinking the `[][]string` corpus representation that
`Build` consumes. Output is byte-identical.
*(Tracked as **[campaign #30]** with the arena alternative; the interning variant
is the memoization framing of the same fix and is the better one if `Build`
memory matters.)*

---

## 2. `newRopeTable` — free, bit-identical, and the standing objection doesn't apply

**Where:** `encoder/rope.go:23`, called fresh per forward in six places —
`forward.go:78`, `forward_batch.go:125`, `forward_q8.go:55`, `forward_q8.go:122`,
`forward_tokens.go:47`, `gte.go:209`.

**Measured:**

```
newRopeTable(512, 64, 1000)   544 µs   131 KB   4 allocs
newRopeTable(200, 64, 1000)   230 µs    55 KB   4 allocs
newRopeTable( 80, 64, 1000)    85 µs    21 KB   4 allocs
```

The cost is `seqLen × headDim/2` iterations of scalar `math.Cos` + `math.Sin`
(plus `math.Pow` per frequency) — 16,384 transcendental pairs at seqLen=512. This
is the same scalar-`math.*` mechanism as **[campaign #13]**, but here the right
fix is not to vectorize it: it's to stop calling it.

**The doc comment at `rope.go:19-22` explains the current design:**

> *"Cheap; called fresh per Forward so cache size is bounded by the largest seqLen
> ever encountered (typical 512 → 512×32 f32 each = 64 KB total — trivially
> cheaper than reusing a global cache with sync)."*

**The objection is sound but the premise is wrong: you don't need a cache.**

`newRopeTable` computes `cos[m*half + d] = cos(m · invFreq[d])` where
`invFreq[d] = base^(-2d/headDim)` (`rope.go:37-47`). Both terms depend only on
`m`, `d`, `headDim`, and `base` — **never on `seqLen`**. The layout is
`[seqLen, halfDim]` row-major, so row `m` sits at the same offset regardless of
the table's total length. Therefore:

> **A table built at `MaxSeqLength` contains every shorter table as an exact
> prefix.** `t.cos[:L*half]` for a 512-table is byte-identical to a freshly built
> L-table.

Verified bit-for-bit for L ∈ {1, 7, 80, 200, 511, 512} (`TestPrefix`, appendix).

So the fix is **one immutable table per model, built once at load time at
`MaxSeqLength`, sliced per forward.** No cache, no keying, no eviction, and no
synchronization — it is read-only after load, exactly like the weights beside it.
For a model whose `headDim` or `RoPEBase` can vary per call, key the single table
on the config struct at load, not per invocation.

**Honest sizing.** 544 µs against a long `Encode` is ~0.2–0.5% — small. The case
where it's visible is the **cross-encoder reranker**: 230 µs × 50 candidates =
**~11.5 ms of pure table construction per rerank query**, plus 2.7 MB of garbage.
That's ~1–3% of a rerank, for a change that removes code rather than adding it.

Also fold in: `gte.go:209`'s rope table is the *second* per-`Encode` allocation in
that function, alongside the 12.6 MB `upGate` buffer at `:230` (**[campaign #8]**).
Fixing both together is one patch.

**Numerics:** bit-identical (proven above). **Risk** low. **Effort** S.

---

## 3. The trap: the Q8 dequantized weight matrix

**Where:** `encoder/linalg_q8.go:52-60`.

This is the most literal instance of "repeated recompute of immutable data" in
the repo, and it is the one place memoization is the **right diagnosis and the
wrong cure.** Worth writing down so it doesn't get proposed as a cache later.

The loop widens the entire `[N,K]` int8 weight matrix to f32 on **every matmul
call** — 113 M scalar converts per Nomic forward, independent of L, and
**480 widens of the same immutable weights per rerank query** (8 workers × 12
layers × 5 matmuls). It costs ~113 ms/forward plus ~0.9 GB of DRAM round-trip.

Caching the widened matrices would eliminate all of it. It is also the wrong
move: the working set is ~9.4 MB per matrix (fc11 at D=768, I=3072), so caching
them all means **holding the f32 weights resident — which defeats the entire
point of storing int8.** The cache would trade the thing the quantization was
bought for.

**The correct fix is fusion, not caching:** fold the widen into `blockedFill`'s
b-panel pack (`linalg/matmul_blocked.go:211-259`) so only an 8×kBlock tile
(≤24 KB, L1-resident) is ever materialized. Same order ⇒ bit-identical, and it
additionally removes `scratch.deqW`'s up-to-9.4 MB pinned per pooled scratch.
Tracked as **[campaign #22]**.

**Rule of thumb this generalizes to:** when the memoized value is large relative
to its inputs, the win is usually in *narrowing the recompute's working set*, not
in storing the result.

---

## 4. Precomputation, not memoization — `bm25` scoring constants

Two things in `bm25/query.go` are pure functions of immutable index state,
recomputed per query. They belong in this doc for completeness but the fix is
build-time precomputation, and they are already tracked as
**[campaign #10 / #29]**:

- **`ix.idf(term)`** (`query.go:19`, called at `:52`) — a `map[string]float64`
  lookup plus a `math.Log` per query term, computing a value fixed since `Build`.
  Fold `idf` into a single `map[string]*termEntry` alongside the postings slice:
  removes the second map lookup *and* the `Log`. Small (query terms number in the
  tens to low hundreds), but free.
- **The per-posting norm** (`query.go:58-62`) — `float64(ix.docLen[p.doc]) / ix.avgdl`
  runs per posting, and the full impact
  `(tf·(k1+1)) / (tf + k1·(1−b+b·dl/avgdl))` is a build-time constant since the
  index is immutable after `Build`. Storing it as a `float32` in the posting
  collapses the query loop to `scores[p.doc] += idf * p.w` — one FMA and one
  scattered write, exactly `sparse`'s shape.

  **Note the locality subtlety:** a *per-doc* precomputed norm array is the same
  random access as `docLen[p.doc]`, so it removes the division but not the second
  cache miss per posting. Only putting the value **in the posting** makes the
  access sequential. That's Lucene's precomputed-impact design, and it hands you
  the per-term max impact that WAND (**[campaign #39]**) needs for free.

  Interim, near-zero-effort step: hoist `invAvgdl := 1/ix.avgdl` out of the term
  loop — halves the divisions per posting from two to one without touching the
  posting struct.

---

## 5. Speculative extension — `StaticModel` word → vector presum

**Where:** `embed/model.go:195-225`.

Model2Vec pooling is: for each token id, gather its embedding row and accumulate
into an f64 sum, tracking `wsum`; divide at the end. Since §1 establishes that
98% of *words* repeat, you could memoize one step further — cache
`word → (weighted f64 sum over its subwords, wsum)` — and skip **both**
`wordPiece` *and* the per-subword gather. A 3-subword word becomes one vector add
instead of three, so it also cuts pooling FLOPs ~3×.

**Two honest problems.**

1. **Not provably bit-identical.** The accumulator is f64 (`:195`), and f64
   addition is non-associative: today's order is `((s + t₁) + t₂) + t₃`, the
   presummed order is `s + (t₁ + t₂ + t₃)`. The drift is ~2⁻⁵² relative,
   accumulated over a few hundred words, against a final narrowing to f32 at
   `:222` (`out[j] = float32(sum[j]/wsum)`) — so it would *almost certainly* round
   to identical f32 output. "Almost certainly" is not a proof. Settle it
   empirically on the real corpus before committing; if it holds bit-exactly
   across the parity pins, ship it, and if not, this whole item is dead.
2. **Memory is real.** `unique_words × dim × 8 B` — ~29 MB for this corpus at
   dim=384, ~61 MB for a 30k unique-word set at dim=256. Storing the presum in
   f32 halves it but adds a second rounding, which makes problem (1) worse.

**Recommendation:** do §1 first. It is bit-identical, small, and captures most of
the tokenizer win. Only look at this if profiling shows the pooling gather is
still material afterwards. `m.weights[id]` is per-id (`:213`), so the presum must
fold the weights in and `wsum` must be cached alongside — don't split them.

---

## 6. Where memoization does not help, and why

The useful half of the audit. Each of these looks like a candidate and isn't:

- **Any dot / matmul kernel** (`linalg/*`) — inputs never repeat, and a hash probe
  costs more than the arithmetic it would skip. Non-starter on both (b) and (c).
- **HNSW pairwise similarities during build.** Tempting, because `prune` and
  `selectHeuristic` (`ann/hnsw.go:315-327`, `:493-526`) genuinely recompute
  similarities between the same node pairs across different inserts. But: a map
  probe is ~20–50 ns against a ~100 ns d=256 dot, the key space is O(n·M²) so the
  cache grows superlinearly with corpus size, and the visited set already prevents
  intra-layer re-scoring. **Batching those dots through `Dot8x4`
  (**[campaign #17]**) is strictly the better answer** — same work eliminated, no
  memory, no lookup.
- **`softmax` / `GELU` / `SiLU` / `layerNorm`** — continuous domain, nothing to key
  on. A quantized lookup table with interpolation would break parity for a worse
  result than the SIMD kernel in **[campaign #13]**.
- **Regex compilation** (`chunk/regex/*.go`) — already hoisted. Every
  `regexp.MustCompile` sits inside a `…Rules()` constructor invoked from
  `init()` (`golang.go:35`, and the same pattern in `python.go`, `java.go`,
  `rust.go`, `typescript.go`), and `chunk/regex/chunker.go:64-66` registers a
  single shared `*Chunker`. Nothing is recompiled per chunk call. **No action.**
- **`normalize` / `preTokenize`** (`embed/tokenize.go:355`, `:485`) — keyed on
  whole-document text, which never repeats. The wins there are fast paths
  (ASCII short-circuit, byte-offset slicing), not caching.
- **`WeightMat.Row` int4, for the LM head** (`linalg/weightmat.go:151-163`) — all
  rows are read every token, so a complete cache means holding the f32 weights,
  i.e. §3's trap again. For the *tied embedding* lookup token ids do repeat during
  generation, so a small LRU is arguable — but fixing the per-element `SDIV`
  (**[campaign #19]**) makes the recompute 2–4× cheaper and largely removes the
  motive. It's goinfer's decode loop anyway, so it's their call, not aikit's.
- **`fuse`** — nothing derived and reused; the wins there are the parallel maps
  and `slices.SortStableFunc`.
- **Query-level result caching** (same query → same hits) — real, but it belongs
  in the *consumer*, not the library. `ann`/`bm25` should not grow a result cache;
  a caller that has repeated queries can wrap them trivially, and a library-level
  cache would need invalidation semantics on a type documented as immutable.

---

## 7. Sequencing — outcomes (2026-07-29)

1. **§2 `newRopeTable` — ✅ DONE.** `gte.go` already went through a `ropeCache`
   under **[campaign #8]** (GTE-only). This pass extended the same cache to the
   five remaining forwards (`forward.go`, `forward_q8.go` ×2, `forward_tokens.go`,
   `forward_batch.go`), added the field to `Weights`/`WeightsQ8`, and gated it on
   the prefix-identity test. Bit-identical, `-race`-clean. **b001891.**
2. **§1 `wordPiece` memo — ✅ DONE.** Sharded-RWMutex cache + differential fuzz +
   concurrent parity test landed together. Measured on the **real minilm vocab**
   (not the synthesized one): **4.59×** cold→memo, 96.8% repeat rate — beating the
   §1 appendix's indicative 3.75×. Byte-identical. **6b69133.**
3. **§1b `bm25` interning — ✅ DONE, in two halves.** The per-token *allocation*
   half shipped earlier as **[campaign #30]** (`bm25.Tokenize`, 983→2 allocs,
   −44.7%). The *retained-memory* half — Index map keys aliasing (and pinning) the
   source text / arena — shipped as **bc4387a**: `Build` `strings.Clone`s each
   term's first occurrence, so a 356-file index reclaims 6.0 MB of source corpus.
   **Landed in `Build`, not `Tokenize`, on purpose:** a Tokenize-side interner
   measured a **1.56× throughput regression** (172→110 MB/s) from a lock+map probe
   on the zero-cost fast-path view; `Build` is single-threaded and interns only the
   ~11k retained keys, so tokenize stays at baseline (+8% Build, index-time only).
4. **§4 `bm25` constants — deferred/conditional; interim already done.** The
   `invAvgdl` hoist is **[campaign #10]** (K1/B-safe half, bit-identical); the
   16 B posting struct is **[campaign #29]** — both DONE. The remaining
   per-posting precomputed-impact change belongs there (it mutates the posting
   struct), not in this memo audit.
5. **§5 `StaticModel` presum — conditional, not started.** Only if profiling
   still points at `StaticModel` pooling after §1, and only if the f64
   reassociation proves bit-exact in practice. Speculative until then.

§3 and §6 are **documentation-only, not doing**: §3 (Q8 dequant is a *trap* —
caching the dequantized rows defeats int8; the real fix is fusion, tracked in
**[campaign #22]**) and §6 (the "doesn't help" list). They exist so the ideas
don't get re-proposed.

---

## Appendix — reproduction

Both benchmarks are self-contained (no aikit imports), so they run on any box.
**Re-run on arm64 before acting on the ratios.**

`§2` — rope table cost and the prefix property:

```go
// rope_test.go — drop in any module, `go test -run TestPrefix -v` then `-bench .`
package rope

import ("math"; "testing")

type ropeTable struct{ headDim, halfDim, seqLen int; cos, sin []float32 }

func newRopeTable(seqLen, headDim int, base float64) *ropeTable {
	half := headDim / 2
	t := &ropeTable{headDim: headDim, halfDim: half, seqLen: seqLen,
		cos: make([]float32, seqLen*half), sin: make([]float32, seqLen*half)}
	invFreq := make([]float64, half)
	for d := range half {
		invFreq[d] = 1.0 / math.Pow(base, float64(2*d)/float64(headDim))
	}
	for m := range seqLen {
		row, mf := m*half, float64(m)
		for d := range half {
			theta := mf * invFreq[d]
			t.cos[row+d] = float32(math.Cos(theta))
			t.sin[row+d] = float32(math.Sin(theta))
		}
	}
	return t
}

func BenchmarkRope512(b *testing.B) { for b.Loop() { newRopeTable(512, 64, 1000) } }
func BenchmarkRope200(b *testing.B) { for b.Loop() { newRopeTable(200, 64, 1000) } }
func BenchmarkRope80(b *testing.B)  { for b.Loop() { newRopeTable(80, 64, 1000) } }

// The load-bearing claim: a 512-table's rows ARE the shorter tables' rows.
func TestPrefix(t *testing.T) {
	big := newRopeTable(512, 64, 1000)
	for _, L := range []int{1, 7, 80, 200, 511, 512} {
		small := newRopeTable(L, 64, 1000)
		for i := range small.cos {
			if small.cos[i] != big.cos[i] || small.sin[i] != big.sin[i] {
				t.Fatalf("L=%d idx=%d mismatch", L, i)
			}
		}
	}
}
```

`§1` — the repetition rate is the number that matters, and it needs no vocab:

```go
// Walk any real corpus, pre-tokenize (BERT basic: lowercase, split on space +
// punct), and count total vs unique words. On aikit's own *.go sources:
//   502,677 total -> 9,463 unique = 98.1% of wordPiece calls are repeats
//   mean word len 2.66 (all) vs 8.84 (unique)
//
// The cold-vs-memo benchmark additionally needs a vocab. Mine was synthesized
// (all single chars + "##"-prefixed singles + the ~8k most frequent whole words
// with count >= 40 and len > 2), so 237.6 ms -> 63.3 ms is INDICATIVE, not a
// measurement of the real 30k WordPiece vocab. Re-run against
// testdata/minilm-model's real vocab for a number worth quoting.
```

Verify equivalence over the whole corpus, not a sample — the failure mode of a
memo bug is a rare word, not a common one.

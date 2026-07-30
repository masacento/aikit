# Task (aikit): five lens scans — compiler-mechanical, rank-invariance, amortization, materialization, workload-driven

> **Companion to** `perf-campaign-2026-07-28.md` (subsystem scan) and
> `task-perf-memoization.md` (the first lens scan). Cross-references to the
> campaign's numbered items are marked **[campaign #N]**.
>
> **Constraints:** core stays pure-Go, no cgo. New hand-written `.s` allowed.

A **lens scan** picks one optimization technique and sweeps the entire codebase
for its signature, instead of reading package-by-package and reporting whatever
turns up. It finds different things, because within any single package an
instance of the pattern just looks like how the code works — the pattern is only
visible when you hold the technique fixed and vary the location.

The memoization scan established this. These five confirm it, and — more usefully
— establish **which lenses are worth running on this codebase and which are not.**

> ## ⚠ Read §5 before acting on §§1–4
>
> **Lens 5 (workload-driven) re-ranked everything.** It measured stage-by-stage
> Amdahl weights for a real index run and a real query, and several findings that
> every prior scan — including this doc's own §§1–4 — ranked highly turn out to be
> worth ~1% end-to-end, while the largest single item in any of the five scans is
> not a kernel or a loop at all. **§6's suggested order is the lens-5-corrected
> one.** The per-lens orderings inside §§1–4 are kept as written so the record of
> what each lens saw in isolation stays intact — but they are superseded.

---

## 0. Yield summary — read this first

| Lens | Method | Yield | Verdict |
|---|---|---|---|
| **1. Compiler-mechanical** | `-d=ssa/check_bce/debug=1`, `-m`, `-m=2` over the whole repo | **1,117** bounds checks, **1,003** heap escapes, **317** inline failures enumerated → **one** measurable win (1.25×) | **Low yield, high confidence.** Worth having run once; not worth repeating. |
| **2. Rank-invariance** | manual sweep + differential tests | 2 assigned leads both **measure at exactly 0%**; 1 real win (5–8%, 25% with its neighbour) | **Low yield — but it found a latent correctness trap.** |
| **3. Amortization of per-item overhead** | manual sweep + microbenchmarks | 12 findings, several large: **1.62× on the default chunker**, **2.83× on BPE**, **1.36× on WordPiece**, **3.3–4.6×** on 8 sort sites | **Highest yield of the four.** Run this lens on any codebase. |
| **4. Intermediate materialization** | enumerate every `make()` on a request path | 11 findings; the largest is **7.4–15% of the transformer trunk, bit-identical, verified** | **High yield, especially for footprint.** |
| **5. Workload-driven** | trace one full index run / query / cold start end to end; profile the seams | **1.74× on 2 cores (~4–6× on 8) from a missing API**, plus an Amdahl table that demotes four prior "top" findings | **Highest-value lens of the five — and the only one that can tell you what the others are worth.** |

**The meta-result: two of the five lenses were near-misses, and that is worth as
much as the hits.** Lens 1 proves the codebase is already tight where the compiler
can see; lens 2 proves the SIMD kernels are so dominant that scalar arithmetic
around them is invisible. Both were plausible a priori. Now they are settled, and
neither needs re-running.

**The second meta-result, from lens 5: four of the five lenses can only rank
findings *within* themselves.** They measure ratios on a stage, not the stage's
weight. Lens 5 supplies the weight — and once it does, "20–30× on index
serialization" becomes 0.083% of an index run and "biggest win-per-hour in the
doc" becomes a 1.4% item. **Run a workload lens before committing engineering time
to any technique lens's output.**

**Measurement environment for everything below:** Intel Xeon @2.8 GHz, 2 cores,
AVX2 (no VNNI), Go 1.24.7, Linux. The repo does not build in the sandbox
(`go.mod` pins Go 1.26.5, module downloads blocked), so measurements come from
verbatim package copies in a `go 1.24` scratch module or exact-body reproductions.
**Ratios transfer to the M1 Pro; absolute times do not.** Re-run before shipping.

---

# Lens 1 — Compiler-mechanical sweep

**Method.** The whole repo (minus `gpu/` and `chunk/treesitter`, plus two
five-line stubs for `x/text/unicode/norm` and `x/sys/unix`, which are the only
external imports in the tree) compiles clean under Go 1.24. That makes the
compiler's own diagnostics available across all 15 packages:

```
go build -a -gcflags='all=-d=ssa/check_bce/debug=1' ./...   # surviving bounds checks
go build -a -gcflags='all=-m'   ./...                       # escape analysis
go build -a -gcflags='all=-m=2' ./...                       # inlining decisions
```

The appeal of this lens is that it is **exhaustive and machine-generated** — no
judgment, no sampling. The result:

| report | aikit-owned hits |
|---|---:|
| surviving bounds checks | **1,117** (encoder 350, linalg 202, embed 200, vision 117, ann 94, chunk 42, topk 39, bm25 19, sparse 11, mmap 1) |
| heap escapes | **1,003** |
| inline failures | **317** |

### The honest result: almost none of it is hot

Intersecting those 1,117 bounds checks with the loops that actually run hot:

**`linalg/matmul_blocked.go:123-124` — 8 bounds checks, and they don't matter.**
`accumRowRange`'s inner loop takes eight `&b[(n+i)*K+k0]` addresses for the
`Dot8x4` call. The compiler can't prove `(n+7)*K+k0 < len(b)`, so all eight
survive. But each `Dot8x4` call performs `k4·4 ≈ kSpan` MACs across 8 columns —
**~6,144 MACs at kSpan=768.** Eight bounds checks per 6,144 MACs is noise by
construction. The k-tail at `:135-142` has 9 more per iteration, over 0–3
iterations. **Not a finding.** Recorded so nobody "optimizes" it.

**`topk/topk.go:107` — eliminated, measured at zero.** `score <= s.heap[0].score`
in the reject path carries a bounds check that runs once per rejected candidate —
N per query. A rewrite that hoists `h := s.heap` and gives the compiler `len(h) > 0`
removes it entirely (verified: the check disappears from the BCE report). Measured
over 400k candidates at k=10: **1,136,066 ns → 1,133,533 ns.** Nothing. The branch
is perfectly predicted and the loop is dominated by the non-inlined call —
which is exactly what **[campaign #3]** already measured at 1.43× by attacking the
call, not the check.

### The one real win

**`linalg/quant.go:112-115` — `q8Span`'s widen loop, 1.25×, bit-identical.**

```go
bq := bQ[j*K : j*K+K]
for k := range K {
    deq[k] = float32(bq[k])       // two bounds checks per element
}
```

`for k := range K` gives `k ∈ [0,K)`, and `len(bq) == K` — but the compiler does
not connect the *variable* `K` to `len(bq)` through the slice expression, and
`deq`'s length is unknown. So both indexes are checked, per element, on a loop
that **[campaign #22]** measured at 113 M elements per forward.

The standard idiom fixes it:

```go
bq := bQ[j*K : j*K+K]
d := deq[:len(bq)]
for k, v := range bq {
    d[k] = float32(v)
}
```

Verified: both per-element checks vanish from the BCE report (only the two
once-per-column slice expressions remain). Measured, K=768 × 512 columns:
**312.3 µs → 250.6 µs, 1.25×**, output bit-identical.

**Caveat that halves its value:** if you land **[campaign #22a]** (a SIMD
`VPMOVSXBD`+`VCVTDQ2PS`+`VMULPS` widen, 6–8×) this becomes moot. Take it as the
five-minute interim, not as the fix.

### Escape analysis and inlining: clean bills of health

- **`linalg/shapecheck.go` is correctly built.** It reported 26 escapes, all from
  `fmt.Sprintf` in panic branches — `rows`/`cols` box only if the panic fires.
  The always-on `requireLen`/`requireExactLen` cost is a handful of comparisons
  plus one integer divide in `mul`'s overflow check, per *public matmul entry*.
  And `checks_off.go` compiles the richer contract asserts to genuinely empty
  functions with concrete (non-interface) parameters, so there is no boxing on the
  hot path. **The design comment is accurate. No action.**
- **Inlining: no actionable finding.** The 317 failures are dominated by large
  functions called once per matmul (`accumRowRange` cost 965, `MatmulBTW8A8Into`
  374, `blockedFill` 159) — they are supposed to be big, and the work inside
  dominates the call. The only mildly interesting class is `defer` blocking
  inlining on `bm25.Tokenize` (`tokenize.go:108`), `FlatI8.query`
  (`flat_i8.go:99`), and `packedFill` (`matmul_blocked.go:211`) — all per-call,
  not per-item. Cosmetic.
- **`bm25/tokenize.go`'s six `string(*scratch) escapes to heap`** confirm
  **[campaign #30]** mechanically rather than by inspection.

### Verdict on lens 1

**Run it once, keep the output, don't repeat it.** It produced one 1.25×, proved
three plausible worries false (the GEMM checks, the topk check, shapecheck
boxing), and cost an afternoon. Its real value is that "are we leaving bounds
checks in the hot loops?" is now **answered**, not assumed. Re-run only after a
major rewrite of a hot loop.

*Reproduction: the stubs + scratch module take ~5 minutes to rebuild; see the
appendix.*

---

# Lens 2 — Rank-invariance / monotone-transform hoisting

**The signature:** a value consumed only by a comparison, ordering, selection, or
top-K, carrying a per-candidate transform that cannot change the ordering — a
positive loop-invariant scalar, an additive constant, or any monotone map. Hoist
it past the selection or delete it.

This lens was proposed because **[campaign #2]** (SPLADE's `log1p` inside a
max-reduce) is an instance of it — filed under memoization, but actually this.

### Both obvious leads are mathematically valid and worth exactly nothing

**`ann/hnsw.go:236` — `sim()`'s `qv.scale`.** Provably rank-invariant, with an
unusually clean proof: `|D| ≤ dim·127²`, and `float64(D)·qs` is *exact* whenever
`dim < 33,285`, so the hoisted key is exactly `Pᵢ/qs` and ordering by it is a
refinement of today's ordering — it can only break rounding-induced ties, never
invert one. An adversarial search over 40 M near-tied tuples found **0
inversions**; end-to-end over 1,500 queries the hits were **bit-identical**.

Measured (n=20,000, d=384, k=10, 256 rotating queries):

```
current   212533 / 177480 / 191400 ns
hoisted   202560 / 212670 / 219227 ns
```

**Indistinguishable.** One scalar multiply per `sim()` call — ~2,000 per query at
ef=64 — is entirely hidden behind a 384-element SIMD int8 dot.

**`linalg/quant.go:225` via `ann/flat_i8.go:125` — the W8A8 query scale.** Same
shape, one loop-invariant `float32` multiply per document before `topHits`.
Measured across three dimensions:

| dim | current | hoisted |
|---|---:|---:|
| 64 | 333,840 ns | 341,158 ns |
| 384 | 1,095,497 ns | 1,110,221 ns |
| 768 | 2,093,873 ns | 2,148,820 ns |

**No win at any dimension.** (An earlier count=3 run showed 3.5%; it was noise.
This is a good reminder to use min-of-N.) The scan is bound by the int8 dot, not
by the scalar arithmetic around it. Worse, unlike the HNSW case this one is *not*
bit-identical — the float32 chain has two roundings, and a targeted search over
60 M near-tied tuples found **1,824 genuine order inversions (3×10⁻⁵)**. Within
the documented sub-ULP tolerance, but a real cost for a zero gain.

**Recommendation: do neither.**

### The one that pays — and the trap that would have shipped a bug

**`bm25/query.go:62` — hoisting `(K1 + 1)`, ~5–8%.**

```go
scores[p.doc] += idf * (float64(p.tf) * (ix.K1 + 1)) / denom
```

`K1+1` is a query-wide constant multiplied in on **every posting of every term**,
and it is a common factor of every summand, so it distributes out of the entire
accumulation. Measured (400k docs, 24 terms, ~8k postings/term, min-of-6):
**1,647,917 ns → 1,558,226 ns = 5.4%** (median-of-6: 8.3%). It pays here — and
only here — because the BM25 loop is a light scatter-add where scalar arithmetic
is actually visible, unlike a dot product.

Two caveats that must be in the implementation:

1. Pulling a factor out of a *floating-point sum* is not bit-exact
   (`Σ fl(aₜ·c) ≠ c·Σ fl(aₜ)`). Verified: top-k set and order differed **0 times**
   over 1,600 synthetic queries. Exact for single-term queries.
2. **`K1` is an exported, unvalidated field** (`bm25/index.go:20`). `K1 < -1`
   makes the factor negative and **inverts the ranking**. Guard it.

`Result.Score` and the exported `Scores()` expose the true BM25 value, so:
keep `Scores()` as-is, add an internal reduced-score path for `TopK`, apply
`(K1+1)` to the ≤k survivors.

**Do the neighbour at the same time.** The same line recomputes
`K1*(1-B+B*docLen[d]/avgdl)` — a pure function of `docLen[d]`, including a
division — per posting. Precomputing it at `Build`: **16.8%**. Both together:
**25.0%.** That is **[campaign #10/#29]** and `task-perf-memoization.md` §4; this
measurement is the number those were missing.

### ⚠ The latent trap — the reason this lens earned its keep

A naive `hnsw.sim` hoist **silently corrupts the graph.** `ann/hnsw.go:508`
evaluates:

```go
if h.simIDs(e.id, sel.id) > e.sim {
```

That is a **cross-comparison between a node–node similarity and a node–query
similarity**. It is correct today only because in `Add`, `qv = h.prepare(vec)` and
`h.bq[id]` are quantized from the same row by the same pure function, so
`qv.scale == h.scales[id]` exactly and `sim(qv,c) ≡ simIDs(id,c)`. Drop `qv.scale`
from `sim()` alone and you change the units of one side of that comparison only —
producing a **different graph**, with no test failure and no crash, just quietly
worse recall.

**This belongs in a comment on `sim`/`simIDs` regardless of whether anything is
optimized.** The invariant "these two must stay in matching units" is currently
undocumented and load-bearing.

*(Related, same file: `h.scales[e.id]` is a positive factor on **both** sides of
that comparison and cancels exactly — the correct way to think about the test,
though it too will measure at ~0.)*

### Smaller findings, recorded

- **`ann/hnsw.go:494`, `:558` — re-sorting an already-sorted slice.**
  `searchLayer` returns `slices.SortFunc(out, candCmp)`-sorted output
  (`:401`); `selectHeuristic` immediately calls `sortCandsDesc`, which copies and
  applies **the identical comparator**. ~1,500 redundant comparisons and a
  **3.2 KB allocation per insert per layer**. Not redundant when called from
  `prune` (`:321`), which passes unsorted input — so pass a `sorted bool`.
- **`encoder/mlp.go:118-135`** — the router runs a full softmax to feed a top-2
  argmax. Only the final normalize over all experts is removable (6 of 8
  multiplies), because the *value* is used at `:152`. Negligible.
  **Hazard for anyone who tries the obvious version:** the `scores[bestIdx] = -1`
  sentinel at `:135` is valid *only because scores are post-softmax and in [0,1]*.
  Moving the argmax to raw logits makes `-1` a live value and silently re-selects
  a consumed expert.
- **`embed/model.go:222-225`** — dividing by `wsum` immediately before
  `L2Normalize` is a redundant positive rescale. **Contract-blocked:** the godoc
  at `:183-189` pins the float32 round-trip as a numpy-parity requirement ("do not
  'optimize' it away") with a golden test. Also `wsum < 0` is reachable with
  negative SIF weights, where the current code returns the negated unit vector.
  Leave it.

### Checked and clean

`encoder/crossencoder.go:92-102` returns the raw logit with no sigmoid — already
correct. `fuse/rsf.go:62-73`'s min-max normalization is per-ranking and summed
across rankings, so it is *not* a common factor — correct as written.
`fuse/fuse.go:93`, `sparse/sparse.go:123-129` already hoist their weights.
`ann/flat.go:151-191` applies no scale at all.

### Verdict on lens 2

**Low yield, and now settled.** The library's hot loops are SIMD dot products;
one scalar multiply per candidate is invisible next to them. The lens pays only
where the inner loop is *not* a dot product — which in this codebase means
`bm25`/`sparse` and nothing else. Don't re-run it broadly; do apply it
reflexively when writing new scatter-add-shaped code.

---

# Lens 3 — Amortization of per-item overhead

**The signature:** a fixed cost paid once per *item* that could be paid once per
*batch*, *call*, or *corpus*. The work is necessary; the overhead around it is
being re-paid at too fine a granularity.

**This was the highest-yield lens of the four**, and it found the single largest
untouched win in the repo — in `chunk/regex`, a package both prior scans
essentially skipped.

## 3.1 · `scanDepth` pays a closure call + `runtime.memequal` per **byte of source** — 1.73×

`chunk/regex/chunker.go:228-349`, the highest-frequency loop in the repo.

```go
for i := 0; i < len(src); i++ {
    atLineStart(i)                                        // :263 closure, per byte
    c := src[i]
    switch state {
    case normal:
        switch {
        case cmtMark != "" && hasPrefixAt(src, i, cmtMark):   // :268
        case hasPrefixAt(src, i, "/*"):                       // :271
        case cfg.tripleQuote && hasPrefixAt(src, i, `"""`):   // :274
```

**Frequency:** once per byte of every indexed file, ×3 `hasPrefixAt` calls in the
`normal` state. A 378k-chunk corpus at ~1.5 KB/chunk ≈ 570 MB ⇒ **~570 M
iterations, ~1.7 B `hasPrefixAt` calls.**

**Mechanism.** `cmtMark` is a *string variable* (`cfg.lineComment`), so `len(s)`
is unknown at compile time and `string(src[i:i+len(s)]) == s` (`:355`) cannot be
specialized into two byte compares — it lowers to a **`runtime.memequal` call.**
Profile of `chunkWith` over aikit's own sources: `memeqbody` **13.7% flat**,
`hasPrefixAt` **24.2% cumulative**, `scanDepth.func1` (the per-byte closure, whose
body is a bounds-checked load that is false ~97% of the time) **8.1% flat**.
`scanDepth` overall is **63.7% cumulative of `chunkWith`** — the regexes everyone
would suspect are only 12.1%.

**Fix (bit-identical):** cache `nextPos := lineStart[nextLineIdx]` as a scalar and
compare `i` against it; gate each `hasPrefixAt` behind its first byte
(`c == '/'`, `c == '"'`, `c == cmt0`).

**Measured**, 322 files / 1.63 MB, depth output compared line-for-line:

```
scanDepth current   9.19 ns/byte
scanDepth gated     5.31 ns/byte      1.73×, bit-identical
```

`scanDepth` is unexported — **no API change.**

## 3.2 · `anyMatch` runs a full `regexp.Match` per line × per rule — 9.6× on the predicate, 1.62× on the whole chunker

`chunk/regex/chunker.go:138` → `:202-209`. **Frequency:** `len(r.defs)` calls per
candidate line plus `len(r.attach)` per attach walk-back. Go = 4+3;
**TypeScript = 10 defs + 2 skip + 4 attach**; Rust 14.

**Mechanism, verified in the stdlib:** `regexp.Regexp.doExecute` → `re.get()` →
`matchPool[re.mpool].Get()` with a matching `.put()`
(`/usr/local/go/src/regexp/regexp.go:232,260,226`). **Every `Match` is a
`sync.Pool` round-trip** plus `inputs.init`, `bitState.reset`, and onepass/
backtrack dispatch. For `^func\b` — a pattern whose entire semantics is "does this
line start with `func`" — that is ~100% overhead.

**Measured**, 47,369 lines of Go source, boolean result identical on every line:

```
anyMatch(goDefs, line)   4 regexps    219.3 ns/line
byte-prefix equivalent                 22.8 ns/line     9.6×
one regexp.Match, no-match             74.6 ns          ← the per-call floor
anyMatch(tsDefs, line)  10 regexps    1400   ns/line
```

**Fix (bit-identical).** `regexp.Regexp.LiteralPrefix()` returns exactly what is
needed; compute it once per rule at `LanguageRules` construction and skip the
`Match` when `!bytes.HasPrefix(line, prefix)`. An `^lit…`-anchored pattern
provably cannot match a line lacking `lit`. Verified prefixes: `^func\b`→`func`,
`^type\b`→`type`, `^//`→`//`, `^/\*`→`/*`. TypeScript's `^(export\s+)?…` yields an
empty prefix and needs a hand-written first-byte class instead.

**Combined with 3.1 and a pre-sized `lineStart`:**

```
chunkWith, 322 files   current   23.6 ms   6,450 allocs
                       fixed     14.6 ms   4,503 allocs     1.62×
```

Chunk-for-chunk identical output including `Text`, over every file. All fields
unexported — **no API change.**

> **Measured dead end — record it.** Compiling the N patterns into one alternation
> so a single machine acquisition covers all of them is **9× *slower*** (Go 2,043
> ns/line, TS 5,643 ns/line): the union defeats onepass and literal-prefix
> optimizations. **Prescreen, don't union.**

## 3.3 · `bpeBackend.bpe` allocates three times per merge step — 2.83×

`embed/tokenize_bpe.go:120-154` (granite-embedding-english / RoBERTa family).
Per GPT-2 pre-token piece: one `string(r)` **per rune** (`:123`), a fresh backing
array **per merge step** (`:141`), a `a+c` string concat **per merge** (`:144`).

**The insight:** every BPE symbol is, by induction, a **contiguous substring of
`mapped`** — initial symbols are single runes in order, and a merge joins two
adjacent ranges. So `symbols` can be `[]struct{lo,hi int32}` offsets into `mapped`
held in two caller-owned double-buffered scratch slices, the map key becomes
`[2]string{mapped[lo1:hi1], mapped[lo2:hi2]}` (sub-slices, no allocation), and
`a+c` becomes `span{lo1, hi2}`.

**Measured**, 503,948 pieces, symbol-for-symbol identical:

```
bpe current      1304 ns/piece   147.6 MB   4,784,137 allocs
bpe span-based    461 ns/piece     5.8 KB           9 allocs     2.83×
```

**Conservative:** the synthesized merge table has 3,323 entries vs GPT-2's ~50k,
so real inputs run more merge rounds and pay `:141`/`:144` more often.
Bit-identical, no API change.

## 3.4 · The WordPiece pipeline — the calibration item, decomposed

`embed/tokenize.go:542-543` confirmed. Priced in isolation: **`sync.Pool` Get+Put
= 15.2 ns; with defer = 17.9 ns** (the defer is open-coded — `deferwrap1` inlines
— so 2.7 ns, the cheap kind). At 500,488 words that is ~9.0 ms/corpus.

Two things in the same chain cost more:

- **`:545` `var out []int32`** — copied and discarded at `:270`
  (`append(ids, …...)` copies), so the per-word slice is pure garbage.
- **`:486-504` `preTokenize`** rebuilds every token through a `strings.Builder`,
  when every emitted token is already a contiguous byte range of the normalized
  text. `:495`'s `out = append(out, string(r))` heap-allocates **per punctuation
  character** — confirmed by lens 1's escape report
  (`tokenize.go:495:30: string(r) escapes to heap`). Code is punctuation-dense.

**Measured**, 322 files / 500,488 words / 98.1% repeat rate (reproducing the
memoization doc's number), ids identical at every step:

| variant | ms | allocs | Δ |
|---|---:|---:|---|
| V1 current | 267 | 1,097,920 | — |
| V2 pool+defer hoisted to `encodeSegment` | 255 | 1,097,918 | 4.4% |
| V3 + per-word `out []int32` removed | 233 | 530,752 | 8.6% |
| V5 + `preTokenize` byte-slicing | **197** | **666** | **1.36×, 1,649× fewer allocs** |

`preTokenize` alone: **74.8 ms → 28.2 ms (2.65×)**, 526,690 → 328 allocations.
Bit-identical, no API change. This **composes with** the memoization fix — the
memo removes 98% of the calls, this makes the remaining 2% and the pre-tokenize
pass cheap.

`embed/tokenize_unigram.go:223,271,280` has the identical shape (three allocations
per piece; `make([]node, size+1)` alone is **138.8 ns → 9.3 ns** with a reused
scratch, ~65 ms/corpus). **All three tokenizer backends allocate a per-word result
slice the caller immediately copies away** — one scratch threaded through
`encodeSegment` fixes all three.

## 3.5 · `scorePaged` re-quantizes the query once per paging block

`ann/flat_i8_mmap.go:113-124` calls `MatmulBTW8A8Into` per block, and
`linalg/quant.go:187-191` unconditionally re-quantizes the activation rows inside.
`flat_i8.go:56-58` states the operating point itself: **"~9766 blocks per query on
a 10M-vector index."**

`quantizeRowInt8` is two full O(K) passes plus a `math.Round` that is **not an
amd64 intrinsic** (**[campaign #26]**). Measured **3.92 µs at K=768, 5.36 µs at
K=1024** ⇒ at their own stated 9,766 blocks, **~52 ms per query** recomputing a
value that is constant for the whole scan, into the same buffer every time.

The irony: `MatmulBTW8A8Into`'s own doc (`quant.go:180-184`) advertises exactly
this fix one level down — *"quantizes each activation row ONCE (into ws) rather
than once per worker"* — and the block loop reintroduces it one level up. The
`f.ws` field was added to hoist the *allocation* out of this loop (audit #13); the
*quantization* it wraps was left behind.

Bit-identical. **Needs one additive exported `linalg` entry** (`MatmulBTW8A8Pre`,
or `Workspace.QuantizeRows` + an exported span) since `w8a8Span` is unexported.
Additive only.

## 3.6 · `sort.Slice`/`SliceStable` survives in 8 sites — audit #24 only landed in `hnsw.go`

`ann/flat.go:111,131`, `ann/flat_i8.go:144,160`, `bm25/query.go:95,116`,
`sparse/sparse.go:149,167`. Four of those are **full-sort paths, O(corpus)**.

```
k=10     sort.SliceStable   450 ns / 3 allocs  →  slices.SortStableFunc    98 ns / 0    4.6×
k=100                      25.7 µs / 3 allocs  →                          5.7 µs / 0    4.5×
k=1000                      572 µs / 3 allocs  →                          171 µs / 0    3.3×
```

**Bit-identical at all eight sites** by exactly the argument `ann/hnsw.go:531`
already makes: every comparator's second key is a unique id (`Index`, `Doc`,
`Item`), so each is a total order and stability is irrelevant. No API change.

## 3.7 · `bm25.Build` does three map operations per (document, term) — 1.33×

`bm25/index.go:53-56`. `m[k] = append(m[k], v)` is a `mapaccess2` *plus* a
`mapassign` — two independent hashes of the same string key — and `ix.df[term]++`
is a third, in a *different* map. Measured on the campaign's corpus shape
(200k docs × 120 tokens): **23,905,452 pairs ⇒ ~72 M map operations where ~24 M
would do.**

```
Build current    9.15 s   1.239 GB   319,792 allocs
Build one-map    6.89 s   1.233 GB   349,487 allocs     1.33×
```

`map[string]*termEntry{postings, df}`. Postings and `df` verified identical for
every term. Compose with **[campaign #29]**'s posting-struct redesign — same file.

## 3.8 · `keep == nil` re-tested per candidate — 1.23× on top of [campaign #3]

`ann/flat.go:107,124`, `ann/flat_i8.go:140,155`. `keep` is fixed for the whole
call; the test runs N times per query.

| variant | ns/item |
|---|---:|
| current | 3.66 |
| `keep == nil` hoisted into two loops | 3.31 |
| threshold hoisted above `Push` — **[campaign #3]** | 2.01 |
| **both** | **1.63** |

Campaign #3 is the big half (1.82× here vs the 1.43× it measured). The nil hoist
is worth a further **1.23× on top**. At N=400k: 1.45 ms → 0.63 ms per query.
Bit-identical (`sel.seq++` must still advance on the skip path — verified).

## 3.9 · Smaller

- **`ann/backend.go:73`** — `QueryBatch`'s CPU fallback loops `f.Query` per query,
  streaming the whole corpus M times, when `w8a8Span` is already column-outer and
  documents weight reuse across M rows. **Mechanism-only: my scalar-kernel
  measurement showed 1.03×**, because a scalar kernel never reaches the memory
  ceiling. The campaign's own N=100k (67.2 ns/row) vs N=400k (120.5 ns/row) gap is
  the evidence that the real kernel does. **Gate this on [campaign #12]** — a
  faster kernel is a more memory-bound kernel.
- **`vision/preprocess.go:127`** — `(v - Mean[c]) / Std[c]` per output element,
  **2.4 M float32 divisions per image**, with `Mean`/`Std` loaded from a by-value
  struct under variable indexing (so they can't stay in registers). Hoisting the
  loads is bit-identical unconditionally; hoisting the reciprocal is exact only
  for `Std = 0.5` (Gemma 3), where `1/0.5 = 2` exactly.
- **`bm25/query.go:57-62`** — the loop-invariant `ix.avgdl > 0` branch and a
  duplicated `float64(p.tf)`: **1.23×**. Subsumed by lens 2's 25% item; do it there.

## Measured dead ends from lens 3

- **Fusing the `scores[i] *= scale` pass into `softmaxRow` is worth nothing
  today.** Provably bit-identical (verified L ∈ {1,7,80,200,512}), but measured
  **16.0 vs 16.0 ns/elem at L=80**; `math.Exp` at ~16 ns/element swamps the
  ~0.3 ns/element pass. **Revisit after [campaign #13]** lands SIMD `expF32` —
  then it becomes ~20% of the softmax instead of 2%.
- **`scanFlat`'s per-vector `emit` closure**: 1024 → 991 ns/vec, **1.03×**. Leave it.

---

# Lens 4 — Intermediate-materialization elimination

**The signature:** a buffer allocated, fully written, then read exactly once by a
single sequential consumer — so it never needed to exist. Report both the
bandwidth win and the **peak-footprint** win; this library targets laptops.

## 4.1 · The last transformer layer computes all L rows; a CLS-pooled model reads one — 7.4–15%, bit-identical

`encoder/crossencoder.go:149-150`:

```go
h := ce.bert.hiddenStates(ids, segs)
cls := h[0:D]     // only row 0 is ever read
```

Same at `encoder/forward.go:93`, `gte.go:189`, `forward_q8.go:66`,
`forward_batch.go:142`.

The final block produces `s.out[:L*D]`, `s.val[:L*I]`, `s.gate[:L*I]`,
`s.mid[:L*D]` and the residual at full L. At L=512, D=768, I=3072 that is
**31.5 MB written, 1.5 KB read.** The other L−1 rows would feed layer N+1's K/V —
and there is no layer N+1.

**Fusion.** In the final layer only: keep K and V over all L rows (attention still
reads them), restrict Q, OutProj, LayerNorm and the whole MLP to row 0. `Wqkv` is
`[3D, D]` row-major, so `Wq`/`Wk`/`Wv` are contiguous slices — no repacking.

**Measured** (one full layer vs one trimmed layer):

| shape | full | trimmed | ratio | share of trunk |
|---|---:|---:|---:|---:|
| D=768 I=3072 L=512 (Nomic) | 503.8 ms | 48.8 ms | **10.3×** | **7.5%** of 12 layers |
| D=384 I=1536 L=200 (MiniLM dims) | 60.3 ms | 5.5 ms | **11.0×** | **15.1%** of 6 layers |

For MiniLM-L6's actual dense-GELU MLP the layer is 384.8 M → 60.6 M MAC = **14.0%
of the trunk** — so read the second row as ~13–15%. **Compounds with
[campaign #13]:** the last layer's softmax `exp` count drops from `heads·L²` to
`heads·L` — at MiniLM, 480,000 → 2,400 calls.

**Bit-identical — verified.** `TestTrimmedMatchesRow0`: 0 of D bits differ,
max|Δ| = 0. The proof is `linalg/matmul_blocked.go:68-70` — each output reduces
over k-tiles determined solely by `kBlock` and `K`; `M` only picks the m-block
boundary (the same M-invariance contract **[campaign #14]** relies on).

**One caveat that will bite.** `encoder/linalg.go:66` routes `M*K*N < 4_000_000`
to `matmulBTNaiveInto`. At M=1, D=768, I=3072 the product is 2.36 M ⇒ the *naive*
path, whose accumulation order differs. The trimmed row must call
`matmulBTBlockedInto` directly — which is exactly what `encoder/mlp.go:148`
already does for the identical reason, comment included. Reuse that precedent.
(Or land **[campaign #27]** and delete the threshold.)

**Applicability.** `Weights`/`GTE` default to `poolCLS` ⇒ applies.
`CrossEncoder.scoreIDs` reads `h[0:D]` unconditionally ⇒ **always** applies, and
that is the rerank path. `BERT.Embed` defaults to `poolMean` and `SPLADE` max-pools
over all L ⇒ neither applies.

## 4.2 · `swigluMLP`'s `gate` is 47% of the 190 MB/worker arena

`encoder/mlp.go:30-37` — `gate` is `[L, I]`, written by one matmul, read by
exactly one sequential loop. At `B*Lmax = 3584`: **44.0 MB**, and `val`+`gate`
together are **88.1 MB of a 188.7 MB arena.**

Column-blocking `gate` and folding it into `val[:, j0:j1]` immediately is
bit-identical by construction — `blockedFill`'s `nStart/nEnd` parameter already
exists for this — and re-streams no weight.

| variant | L=512 | scratch | B·L=3584 | scratch |
|---|---:|---:|---:|---:|
| today | 320.5 ms | 13.50 MB | 2126 ms | **94.50 MB** |
| `gate` col-block jb=256 | 307.4 ms | **8.00 MB** | 2123 ms | **56.00 MB** |
| full row-tile rb=64 | 323.6 ms | 1.69 MB | 2258 ms | 1.69 MB |

**Honest reading: a footprint win, not a latency win.** Column-blocking is
latency-neutral; full row-tiling reaches a 1.7 MB arena but costs ~6% because the
9.4 MB weight matrices get re-streamed per row tile. Report it as **94.5 → 56 MB
per worker, free** — at 8 workers, 1.51 GB → 1.20 GB of pinned scratch on a laptop.

Bit-identity verified at jb ∈ {128,256,512,1024} and rb ∈ {8,16,32,48,64,96,100,128,256}.
**On arm64 use an even row tile** so `blockedFill`'s `for ; i+1 < iEnd; i += 2`
pairing (`matmul_blocked.go:80`) assigns the same rows to `Dot2x8` vs
`accumRowRange`; a multiple of 32 is provably identical.

This also subsumes **[campaign #8]** — `gte.go:230`'s `upGate := make([]float32, L*2*I)`
is the same buffer, and column-blocking removes it rather than relocating it.

## 4.3 · `MarshalBinary` doubles peak RSS; there is no `WriteTo`

`ann/flat_i8_persist.go:42-56`, `ann/hnsw_persist.go:48-113`. FlatI8 at n=1M,
dim=768: a **772 MB** blob held *simultaneously* with the 772 MB index ⇒ **peak
1.54 GB to save a 772 MB index.** HNSW f32 at the same shape: **~6.4 GB peak.**

`grep -n 'WriteTo\|ReadFrom\|io.Writer' ann/` returns nothing —
`encoding.BinaryMarshaler` is the only serialization surface, so **there is no way
to save an index without doubling peak RSS.**

Add `WriteTo(w io.Writer)` streaming through a 1 MiB staging buffer
(`unsafe.Slice` over `f.bq` as `[]byte` is sound — int8 has no alignment
constraint, the argument `flat_i8_mmap.go:170` already makes). Peak intermediate
**772 MB → 1 MB**. Orthogonal to **[campaign #5]**, which is about the *speed* of
the byte-at-a-time loop — do both; #5's bulk `copy` is what makes streaming fast.

**Load side, same file:** `hnsw_persist.go:277-281` and `:290-297` do
`make([]float32, dim)` per doc and `make([]int32, cnt)` per node ⇒ **>2.1 M
allocations on a single `Load`** at 1M docs, where two flat arenas with sub-slices
would be 2. (The campaign's recorded dead end — "contiguous vs scattered doesn't
speed up the *scan*" — explicitly leaves GC mark cost and allocation open. Load is
where they land.)

## 4.4 · `encmetal` stages 453 MB on the Go heap to copy it into host memory

`gpu/encmetal/backend.go:220-228` builds `deq := make([]float32, N*K)` then calls
`NewBufferFloats(deq)` — which is `newBufferWithBytes:`, a memcpy into a
shared/UMA MTLBuffer whose `.Floats()` is documented "zero-copy on UMA". So `deq`
exists purely to be copied into memory the CPU could already write directly.

Per Nomic layer 37.8 MB × 12 = **453 MB allocated, zeroed, written and read** on
the first forward, all immediately garbage. Fix: `NewBufferLen(N*K)` first,
dequantize straight into `buf.Floats()`. **Two lines.** Bit-identical.

(`gpu/enccuda/backend.go:223` is identical code, but there the host buffer is a
genuine PCIe staging requirement — the fix is a *reused* scratch, not removal.)

## 4.5 · `LoadWeightsQ8` materializes the whole f32 model before quantizing

`encoder/weights_q8.go:68-114` calls `LoadWeights(dir)` — the entire f32 model —
then quantizes. Peak is **both**: ~547 MB f32 + ~140 MB int8 ≈ **690 MB to produce
a 140 MB model.** For a BF16/F16 checkpoint the f32 side is a real heap allocation
(`embed/safetensors.go:600-611`), not reclaimable page cache.

`linalg/quant.go:35-38` documents the escape hatch, unused:

> `QuantizeRowInt8` … *exposed so a loader can quantize each row as it is
> dequantized, without buffering the whole f32 matrix.*

Peak intermediate: `cols·4` = **3 KB**. Bit-identical.

## 4.6 · `vision.Preprocess` builds a full-resolution NRGBA that is ≥74% never read

`vision/preprocess.go:90-96`. A 12 MP photo → **48.8 MB**, live alongside the
decoded YCbCr (~18.3 MB) ⇒ **~67 MB peak to produce a 9.6 MB `pixel_values`**.
`resizeNormalize` samples at most `896²·4 = 3.21 M` of 12.19 M source pixels —
**≥74% is written by `draw.Draw` and never read.**

**[campaign #32]** measured `draw.Draw` at 2.3× and framed the fix as "make the
copy faster." The materialization framing says **delete it**: type-switch on
`*image.YCbCr`/`*image.RGBA`/`*image.NRGBA` and sample directly. Not bit-identical
for YCbCr (a hand-written path must reproduce `color.YCbCr.RGBA()` exactly); exact
for RGBA/NRGBA sources. Campaign #32's x-map hoist is independent and still applies.

## 4.7 · `QueryBatch` allocates the whole M×N score matrix to take k per row

`ann/backend.go:60-64`. M=64 over N=1M ⇒ **256 MB**, mirrored by the device
`outBuf` ⇒ **512 MB resident for a call whose answer is 640 hits.** Host-side
streaming (process query m as its row arrives) is a two-line change: 256 MB → 4 MB.
The real fix is **[campaign #21 step 2]** — on-device top-K so the transfer is M·k.

**Note the contrast one directory over:** `ann/flat.go:101-127` already does this
right — `scanFlat` emits into `sel.Push` with no dense array at all. **`Flat` is
the fused reference implementation and `FlatI8` is the unfused one.** That sharpens
the known `FlatI8.Query` finding: the fix is not "pool the buffer," it is "be `Flat`."

## 4.8 · `chunk/regex` buffer sizing — 41% of the chunker's allocated bytes

`chunker.go:100-105` grows `lineStart` by `append` when the count is one
`bytes.Count` away:

| bench (643 KB, 18,431 lines) | ns/op | B/op | allocs/op |
|---|---:|---:|---:|
| append (today) | 832,948 | 660,729 | 21 |
| prealloc via `bytes.Count` | **529,117** | **147,456** | **1** |

**1.57× on that loop, 41% of the chunker's total allocated bytes, one line.**
(`bytes.Count` is an assembly SIMD scan, so the extra pass is nearly free.)
Same in `chunk/lines.go:40-45`.

Same function: `depth := make([]int, n)` (`:230`) is 8 B/line for a value compared
against `maxDepth ∈ {-1,0,1}` — `[]int32` halves it.

**And the `Text` copies** (`:177`): the package doc guarantees chunks are a
contiguous non-overlapping partition, so the K copies sum to `len(src)`. One
`s := string(src)` plus substrings gives the same bytes in **1 allocation instead
of K**: 430 → 2 allocations, latency-neutral. State the tradeoff explicitly — a
retained chunk pins the whole source string, which is right for index-everything
and wrong for filter-then-keep-a-few.

## 4.9 · Smaller

- **`encoder/attention.go:61,83-88`** (and `gte.go:252,271-274`) — `ctxHead` is a
  fourth full copy per layer: 12 heads × 12 layers × 384 KB = **55 MB of traffic
  per forward** for a value that could land in `ctx` directly. Giving
  `blockedFill`/`matmulBTInto` a destination row stride is mechanical (every write
  is `dst[i*N+n] += …`; replace `N` with `dstStride`) and bit-identical. The same
  change would let **[campaign #31]**'s QKV split disappear too.
- **`ann/hnsw.go:399-402` + `:494`** — `searchLayer` returns a sorted copy;
  `selectHeuristic` copies and sorts it again. At efConstruction=200, building 1M
  vectors ⇒ **~3.5 GB of transient allocation and ~1.1 M redundant sorts.** Names
  the largest single allocation inside **[campaign #17]**. Also `queryEf:446-449`
  builds `found` at `ef` then truncates to `k` — 95% of that sort is wasted at
  ef=200, k=10.
- **`encoder/splade.go:102`** — tiling the vocab projection by column removes
  `pooled` too, not just `logits`: peak **62.6 MB → ~1 MB**, and the final full-V
  scan disappears because terms can be emitted per block in ascending order.
  Extends the known finding one consumer further; composes with **[campaign #2]**.

## Checked and clean

`encoder/linalg_q8.go:29-33`'s inline `make([]float32, N*K)` is test-only in
production — a landmine, not a finding. `embed/safetensors.go`'s typed accessors
are already zero-copy views (`reinterpretLE`, `:518-537`). **`embed/model.go:190-228`
`encodeIDs` is already fully fused** — single pass, one f64 accumulator, no
staging. Worth citing internally as the reference pattern.

---

# Lens 5 — Workload-driven

**The signature:** don't sweep for a technique — trace one complete workload from
entry to exit and find what lives in the **seams**. Work done twice across a
package boundary; representation churn nobody owns; stages that could overlap;
cold start as a whole; and above all the **Amdahl weight of every stage**, which
is the one thing no technique lens can produce about itself.

**Method.** Verbatim package copies in a `go 1.24` scratch module. No checkpoints
were staged, so a **potion-code-16M-shaped** Model2Vec artifact was synthesized
(real `tokenizer.json` with a 7,857-entry WordPiece vocab harvested from the
corpus, real `config.json`, hand-written `model.safetensors`) and run through
`embed.LoadFromFS` → `StaticModel.Encode` **unmodified** — so the embed stage is
measured through the real code path. Cross-check: the measured tokenize/pool split
is **63.5% / 36.5%** against `task-perf-memoization.md` §1's independently measured
**65% / 35%** on the real model. Transformer stages remain derived, not measured.

**Corpus:** aikit's own tree — 322 `.go` files, 1,649,771 bytes → **1,566 chunks**,
312,144 bm25 tokens, 536,450 WordPiece ids, dim 256, shortlist k=50.

## 5.1 · W1 — index a repository (static embedder, single-threaded, as both shipped examples do it)

| stage | call site | ms | % of run |
|---|---|---:|---:|
| `chunk.ChunkFile("regex", …)` | `chunk/registry.go:120` | 20.84 | 3.69% |
| `bm25.Tokenize` ×1566 | `bm25/tokenize.go:108` | 29.12 | 5.16% |
| `bm25.Build` | `bm25/index.go:30` | 39.76 | 7.05% |
| **`embed.StaticModel.Encode` ×1566** | `embed/model.go:170` | **462.24** | **81.95%** |
| ↳ `Tokenizer.Encode` | `embed/tokenize.go:213` | 293.48 | 52.03% |
| ↳ `encodeIDs` (pool + gather + L2) | `embed/model.go:190` | 162.10 | 28.74% |
| `ann.NewFlatI8` | `ann/flat_i8.go:68` | 5.29 | 0.94% |
| `MarshalBinary` | `ann/flat_i8_persist.go:41` | **0.47** | **0.083%** |
| unattributed (GC, outer append) | — | 6.34 | 1.13% |
| **end-to-end** | | **564.06** | **100%** |

## 5.2 · W2 — hybrid query, retrieval only (n=1566, k=50)

| stage | call site | µs | % |
|---|---|---:|---:|
| `bm25.Tokenize(query)` | `bm25/tokenize.go:108` | 0.48 | 0.4% |
| `embed.Encode(query)` | `embed/model.go:170` | 8.96 | 7.1% |
| `bm25.TopK` | `bm25/query.go:84` | 15.08 | 11.9% |
| `ann.FlatI8.Query` | `ann/flat_i8.go:88` | 66.20 | 52.2% |
| ↳ `MatmulBTW8A8` | `linalg/quant.go:174` | 46.6 | 36.7% |
| ↳ `topHits` | `ann/flat_i8.go:136` | 21.0 | 16.5% |
| **`fuse.RRF`** | `fuse/fuse.go:56` | **36.45** | **28.7%** |
| **end-to-end** | | **126.89** | **100%** |

Stage sum 127.17 µs vs measured 126.89 µs — the decomposition is complete.

## 5.3 · W2R — the same query with the rerank `examples/rag` performs

| stage | ms | % |
|---|---:|---:|
| tokenize + embed + retrieve + fuse (measured) | 0.127 | **0.06 – 0.18%** |
| `encoder` rerank, 9 forwards (derived) | 69 – 207 | **99.8 – 99.9%** |

> **The single most important number in this document: once you rerank, the entire
> retrieval stack is between 1/500th and 1/1500th of the query.** Every
> retrieval-side finding in all five scans is noise in a reranked pipeline.

## 5.4 · W3 — cold start, genuinely cold process

A faithful transcription of `examples/embedded-corpus/main.go:47-78`:

| stage | source line | ms | % |
|---|---|---:|---:|
| `embed.LoadFromFS` | `main.go:48` | 10.06 | 11.1% |
| `os.ReadFile` + `ann.LoadFlatI8` | `main.go:52` | **0.65** | **0.7%** |
| `json.Unmarshal(corpusJSON)` | `main.go:57` | 16.67 | 18.4% |
| **`TokenizePlain` ×N + `bm25.Build`** | `main.go:62-66` | **60.69** | **66.9%** |
| first query (cold) | — | 0.22 | 0.24% |
| **to first result** | | **90.74** | **100%** |

The dense half — the offline-quantized `//go:embed`ed int8 blob the example is
*about* — is **0.7%** of its own startup.

## 5.5 · Re-ranking the prior scans by end-to-end impact

**This is the deliverable no technique lens could produce.** Percentages are
(prior scan's measured ratio) × (lens-5 measured stage weight).

### W1 — index a repo

| prior finding | its claim | stage weight | **end-to-end** | verdict |
|---|---|---:|---:|---|
| memo §1 — `wordPiece` memo | 3.75× | 22.7% | **≈16%** | **#1 for W1. Confirmed.** |
| lens 3 §3.4 — `preTokenize` byte-slicing | 1.36× | 52.0% | **≈13%** | **#2. Composes.** |
| **new N2** — carve-out rebuild | **26.5×** | 6.8% | **6.8%** | **#3, and it deletes code.** |
| lens 3 §3.7 — `bm25.Build` single-map | 1.33× | 7.05% | 1.8% | real, small |
| lens 3 §3.1+3.2 / §4.8 — **the chunker, 1.62×** | *"biggest win-per-hour"* | **3.69%** | **1.4%** | **Overweighted.** Best ratio-per-hour; not a big absolute item. |
| campaign #30 — bm25 token interning | 787→10 allocs | 5.16% | ≤1.3% | marginal |
| **campaign #5 — serialization bulk copy** | **"20–30×", Tier 0** | **0.083%** | **0.08%** | **Not a latency item.** Re-file under §4.3 (peak RSS). |
| lens 1 — `q8Span` BCE fix | 1.25× | 0% | **0%** | not on this path |

### W2 — retrieval only

| prior finding | its claim | stage weight | **end-to-end** | verdict |
|---|---|---:|---:|---|
| campaign #12 — rewrite `dotI8AVX2` | 2–3× | 36.7% | **18–24%** | **#1 for retrieval.** |
| campaign #3 + lens 3 §3.8 — topk threshold + nil hoist | 2.25× | ≈12.6% | ≈7% | real |
| campaign #16 — shard `Flat.Query` | 1.74–2.08× (2 cores) | 36.7% | ≈18% | real, but at n=1566 `M*N*K = 4.0e5 < 1<<24`, so the **threshold**, not the code, is what makes it serial |
| **campaign #11 — BM25 touched-set** | **"10–50×, >99% waste"** | 11.9% | **≈3–6%** | **Overstated — see N7.** |
| campaign #4 — pool FlatI8 `Workspace` | 10–25% | 10 allocs/query | ≈1% | small at this scale |
| campaign #15 — HNSW `Dot8x4` batching | 1.36–1.40× | **0%** | **0%** | not on the FlatI8 path either example uses |
| lens 3 §3.6 — `SortStableFunc` at 8 sites | 3.3–4.6× | ≈0.8% | ≈0.6% | …but it **missed the two `fuse` sites, worth 14%** — see N3 |

## 5.6 · New findings — the seams

### N1 · Neither `StaticModel` nor the examples have any parallelism — **1.74× measured on 2 cores, ~4–6× on 8**

`embed/model.go:170` is the *entire* public encode surface: `Encode(text string)`,
one string at a time. Compare `encoder/model.go:149` —
`EncodeBatch(texts, isQueries, concurrency)`. **The slow transformer model got
worker fan-out; the bulk-corpus model, whose own doc comment at `embed/model.go:179`
cites "a 378k chunk corpus", got nothing.**

So every caller writes a serial loop, including both shipped examples:
`examples/rag/main.go:82-85`, `examples/embedded-corpus/gen/main.go:73-75`, and a
*second* serial pass at `rag/main.go:89-92`.

Chunking, both tokenizers and pooling are pure functions of one chunk with no
shared mutable state (`StaticModel` documents "goroutine-safe for concurrent
Encode"). Nothing prevents fan-out except that no package offers it.

```
AB_W1_Serial     593.2 ms   84.41 MB   1,226,240 allocs
AB_W1_Parallel   341.6 ms   84.45 MB   1,226,250 allocs    1.74× on 2 cores
```

**This is larger than every kernel finding in all five scans combined, and it is
entirely an API/example defect.** Fix: `StaticModel.EncodeBatch(texts []string,
concurrency int)` mirroring `encoder/model.go:149`. Bit-identical — `encodeIDs`
touches no shared state.

### N2 · `Tokenizer.Encode` rebuilds every document rune-by-rune to find 5 literals — **26.5×, 6.8% of W1**

`embed/tokenize.go:233-250` runs `strings.HasPrefix(text[i:], k)` for all 5
`addedKeys` **per byte**, then `DecodeRuneInString` + `seg.WriteRune` to rebuild
the entire document through a `strings.Builder`. Because `addedKeys` holds
variable-length strings, each `HasPrefix` lowers to a **`runtime.memequal` call** —
confirmed in the profile (`memeqbody` 4.22% flat, 30.77% of its callers being
`internal/stringslite.HasPrefix`).

This is exactly the mechanism lens 3 §3.1 found in `chunk/regex.scanDepth`, **in a
function no lens looked at, on a hotter path.**

```
current (rebuild)   48.42 ms   5,246,404 B   13,908 allocs
first-byte gated     1.83 ms           0 B        0 allocs    26.5×
```

Segmentation verified identical (1,655 segments / 1,649,310 bytes). For BERT every
added key starts with `[`, so the scan collapses to `strings.IndexByte(text, '[')`.

**The subtle bit-identity argument, and it holds:** `WriteRune(DecodeRuneInString(…))`
turns invalid UTF-8 into U+FFFD where slicing preserves the raw bytes — but
`encodeSegment:262` → `normalize:356` → `cleanText:383` ranges over the string
(also yielding U+FFFD) and **drops `r == 0xFFFD` at `:387`**. Both paths erase them
identically. Gate on `t.cleanText` (true for every BertNormalizer config) and it is
provably byte-identical.

### N3 · `fuse.RRF` is 28.7% of a retrieval-only query — **2.18× from one line**

`fuse/fuse.go:78,79,101` (and identically `fuse/rsf.go:41,42,80`): two
`make(map[K]…)` **with no size hint** so both rehash as they grow, plus
`sort.SliceStable` on the reflection path (`reflectlite.Swapper` → `typedmemmove`,
7.55% + 5.12% in the profile).

**Lens 3 §3.6 enumerated 8 `sort.Slice` sites and missed both `fuse` ones** — which
are the only sites where the sort is the majority of its own function.

| shortlist | fused | current | fixed | ratio |
|---:|---:|---:|---:|---:|
| k=10 | ~17 | 4.57 µs | 2.40 µs | 1.90× |
| **k=50** (both examples) | ~89 | **34.3 µs / 22 allocs** | **15.7 µs / 7** | **2.18×** |
| k=200 | ~340 | 182.5 µs | 77.3 µs | 2.36× |

At k=200, `fuse.RRF` alone costs **more than the ANN scan, the BM25 scan and the
query embedding put together** (182 µs vs 90 µs). Verified identical
`{Key,Score}` at every rank over 89 items.

### N4 · `examples/embedded-corpus` spends 67% of cold start rebuilding an index it already had — because `bm25` has no serialization surface

`examples/embedded-corpus/main.go:60-66`, with a comment that concedes it:

```go
// BM25 has no on-disk form (it's cheap to build) — rebuild it from the embedded
// corpus at startup …
```

`grep -n 'MarshalBinary\|WriteTo\|gob' bm25/` returns **nothing**. `ann` has four.
**The asymmetry is the finding.** Measured: **60.7 ms of a 90.7 ms
time-to-first-result**, against 0.65 ms for the dense half the example is actually
about. `README.md:66` advertises "~50 ms startup"; the advertised engineering is
1.3% of it. Linear in corpus bytes ⇒ **~20 s at the 378k-chunk scale**, every
process launch.

Give `bm25.Index` a versioned `MarshalBinary`/`LoadIndex` mirroring
`ann/flat_i8_persist.go`. Five fields. **The single highest-value missing API in
the library** for the zero-deploy story it markets. (Second-order, same file:
`json.Unmarshal` at `:57` is another 18.4% at 118 MB/s, to parse text already
contiguous in the binary.)

### N5 · `ann.NewFlatI8` materializes a full f32 copy of the corpus it is about to discard — **2.02×**

`ann/flat_i8.go:74-78` allocates `make([]float32, n*d)`, copies every input vector
in, hands it to `QuantizeRowsInt8`, and drops it. `embed` produces one `[]float32`
per chunk; `ann` wants contiguous; neither offers the other a way to skip it.

The escape hatch is documented two files away and unused — `linalg/quant.go:35-38`:
*"exposed so a loader can quantize each row as it is dequantized, without buffering
the whole f32 matrix."* (§4.5 cited that same comment for `LoadWeightsQ8` and
didn't notice `NewFlatI8` needs it too.)

```
ann.NewFlatI8 (today)          100.14 ms   25,682,128 B/op
row-streaming QuantizeRowInt8   49.68 ms    5,202,944 B/op    2.02×
```

At 378k × 256 the `flat` array is **387 MB allocated and zeroed**, held alongside
the caller's `[][]float32`. Bit-identical.

### N6 · The `[][]string` seam between `bm25.Tokenize` and `bm25.Build` — 8.8× the corpus, held live, read once

`bm25/index.go:30` takes `docs [][]string`, so every consumer must materialize the
token stream of the **entire corpus** before `Build` may start. Measured over
1.65 MB: **14,530,218 B / 102,148 allocs**, essentially all live until `Build`
returns, read exactly once at `index.go:50`. At the 378k-chunk scale the
`[][]string` peak is **~5 GB** *plus* the index.

`Tokenize` is per-document and correct; `Build` is corpus-at-once and correct; the
*contract between them* forces O(corpus) peak. Additive fix: `Builder.Add(tokens)`
or `Build(iter.Seq[[]string])` so one token slice can be reused.

### N7 · Adversarial: campaign #11's "10–50×" rests on a selectivity assumption aikit's own tokenizer breaks

Campaign #11 measured a 3-term query touching 2,335 of 200,000 docs — **1.2%
selectivity**, synthetic uniform corpus. On the real corpus with real
`bm25.Tokenize` queries:

| query | touched | % of corpus |
|---|---:|---:|
| `read a file line by line` | 1,210 | **77.3%** |
| `build the bm25 inverted index` | 1,275 | **81.4%** |
| `quantize int8 vectors` | 978 | 62.5% |
| `hnsw graph neighbour search` | 106 | 6.8% |
| `reciprocal rank fusion` | 71 | 4.5% |

Median ~12%, mean ~33%, worst 81% — because `bm25/tokenize.go:159-191` emits
**compound + every sub-token**, so a 6-word query becomes 10–15 terms, several of
them corpus-common in code.

Decomposition: `TopK` 15.08 µs = `Scores()` 6.25 µs + selection 8.8 µs; of
`Scores()`, the bare `make([]float64, 1566)` is **2.64 µs = 42%**. So the win is
**~2–3× on `Scores`, ~3–6% of a query.**

**This does not kill #11** — the O(corpus) `make` + sweep is real and grows
linearly. It means **the win is the allocation, not the selectivity**, so the fix
that pays is the pooled/cleared accumulator, not the touched-set list.

### N8 · Both examples make four sequential passes over the corpus — and fusing them measures *slower*

`examples/rag/main.go` passes: chunk (`:74-78`), embed (`:83-85`), tokenize
(`:90-92`), build. Fusing the tokenize and embed passes made W1 **5.2% slower**
(593.2 vs 564.1 ms) — the combined working set of bm25's pooled buffers + the
WordPiece vocab map + the embedding table evicts more than the split passes cost.

Same verdict at query time: `ann.Query` and `bm25.TopK` are independent (81 µs of
127 µs), but overlapping them measured **21% slower** (147.5 vs 121.7 µs) —
goroutine spawn/join costs more than the 15 µs it hides.

**Record both.** The pipelining win here is N1 (across cores), not fusion (within
one). Both would flip at ~10× the corpus; neither flips here.

### N9 · Cold-start warm-up is real and nothing measures it

First query **0.22–0.33 ms vs warm 0.12–0.17 ms — 1.5–2×** at a 400 KB index, from
first-touch page faults and cold branch predictors. `bench/harness.go:68-94` has no
warm-up phase (**[campaign §3 item 1]** noted this; here is the end-to-end number
that justifies it), so its `p99` over a small query set is largely the cold first
query. On the mmap paths — which **[campaign §4a]** shows never call `Advise` —
this gap is the entire point of `MADV_WILLNEED`, and it is untested.

## 5.7 · What would change these conclusions

1. **A transformer embedder instead of Model2Vec.** If W1 uses `encoder.Model`,
   stage 5 goes from 462 ms to hours and every other row collapses below 0.01%.
   **The W1 table is specifically the static-embedder case** — which is what both
   shipped examples and the README's zero-deploy pitch use. Do not cite it for a
   transformer index.
2. **The synthetic vocab is 7,857 entries vs the real ~30k**, so failed WordPiece
   probe chains run longer ⇒ **tokenizer share probably over-stated**. Counterweight:
   the real potion-code-16M table is 64 MB vs the synthetic 8 MB, so real
   `encodeIDs` gathers are far more cache-hostile ⇒ **pooling share under-stated**.
   The errors oppose and the independent 65/35 split in `task-perf-memoization.md`
   §1 lands inside the measured 63.5/36.5. If a real-checkpoint rerun moves the
   split past ~55/45, N2 and the memo item shrink proportionally.
3. **2 cores.** N1's 1.74× is a *floor*. Both of N8's dead ends are *flattered*
   here and could flip on a wider box.
4. **n=1,566 is small.** `MatmulBTW8A8` is below its `1<<24` parallel threshold and
   the whole index fits in L2. At n=1M the ANN scan and BM25's sweep grow 640×,
   **`fuse.RRF` does not grow at all** (it is O(shortlist)), and W2's composition
   inverts — fuse drops from 28.7% to <0.1%. **N3 is a small-to-medium-corpus
   finding; say so.** Campaigns #11 and #12 become dominant exactly there.
5. **k=50.** N3 scales ~n log²n in the shortlist: a 10% item at k=10, 55% at k=200.

## 5.8 · Measured dead ends from lens 5

- **Fusing the bm25-tokenize and embed passes: 5.2% *slower*.**
- **Overlapping `ann.Query` with `bm25.TopK` on 2 cores: 21% *slower*.**
- **`examples/rag/main.go:132-143`'s `cosine` recomputes the query norm per
  candidate.** Real redundancy (8 × 768 wasted MACs), but ~10⁻⁶ of a rerank
  forward. **Not a finding** — recorded so it isn't "optimized."
- **`ann.MarshalBinary`'s byte-at-a-time loop** (**[campaign #5]**, "Tier 0,
  20–30×") measures **0.47 ms = 0.083% of an index run.** Real and bit-identical
  and five minutes — but re-file it under §4.3 (peak RSS), where it belongs.

---

# Cross-lens observations

**1. Technique lenses cannot rank their own output — and they systematically
mis-rank it.** Lens 3 called the chunker "the biggest win-per-hour in the doc."
It is: 1.62× for one afternoon in one file. It is also **1.4% of an index run**,
because chunking is 3.69% of the pipeline. Campaign #5's 20–30× is **0.083%.**
Meanwhile lens 5's N1 — a 1.74× that is not a loop optimization at all, just a
missing `EncodeBatch` — is worth more than every kernel finding in all five scans
combined, because it multiplies the 82% stage. **A technique lens measures a
ratio; only a workload lens supplies the weight.** Run the workload lens first, or
at least before committing time.

**2. Findings cluster by inner-loop shape, not by package.** Every lens-2 win and
most lens-3 wins land where the inner loop is a **scatter-add or a byte state
machine**; everything adjacent to a SIMD dot product measured at zero. That is a
reusable triage rule: *before* investigating scalar overhead, ask what the inner
loop's dominant instruction is. Lens 5 sharpened it into a second rule: **`memequal`
from a variable-length `HasPrefix` in a per-byte loop** turned up independently in
`chunk/regex.scanDepth` (lens 3 §3.1) and `embed/tokenize.go:236` (lens 5 N2). Grep
for that shape directly next time.

**3. Findings that only exist because two lenses crossed.** `q8Span`'s widen (lens
1's BCE + campaign #22's traffic analysis); `preTokenize`'s `string(r)` escape
(lens 1's escape report + lens 3's frequency count); the last-layer trim (lens 4 +
campaign #14's M-invariance proof); and now `fuse.RRF` (lens 5's Amdahl weight
finding the two `sort.Slice` sites lens 3's enumeration missed — **the only two
where the sort is the majority of its own function**). Running lenses in sequence
and letting each read the others' output is doing real work.

**4. The scans found each other's blind spots, symmetrically.** Lens 3 enumerated
8 of 10 `sort.Slice` sites; lens 5 found the 2 that mattered. Lens 4 cited
`linalg/quant.go:35-38`'s streaming-quantize comment for `LoadWeightsQ8` and
missed that `ann.NewFlatI8` needs it too; lens 5 found it. Lens 5's Amdahl model,
in turn, is only usable because the technique lenses supplied the per-stage ratios
it multiplies. **No single scan was sufficient and none was wasted.**

**5. The dead-end list is now as valuable as the findings list.** Across five
scans: the topk bounds check, both rank-invariance leads, the regex union, the
softmax-scale fusion, the `scanFlat` closure, contiguous vector storage, HNSW heap
pooling, `layerNorm` vectorization, pass fusion in the index loop, query-side
retrieval overlap, and the example's redundant `cosine` norm. **Every one is
something a well-intentioned future contributor would try.** Keep them written
down.

**6. Two of the five biggest items are missing APIs, not slow code.**
`StaticModel.EncodeBatch` (N1) and `bm25.Index` serialization (N4). Neither is
findable by reading a hot loop, because the defect is *the absence of a call*.
Only a workload trace surfaces them — and both are breaking-ish additions worth
deciding on before v1.0 freezes the surface.

---

# Suggested order — lens-5-corrected

Ordered by **measured end-to-end impact**, not by per-stage ratio. This supersedes
the per-lens orderings inside §§1–4.

**Phase A — the workload wins. Do these first; they dwarf everything else.**

1. **§5.6 N1 — `StaticModel.EncodeBatch` + use it in both examples.** 1.74×
   measured on 2 cores, ~4–6× on 8, on the 82% stage. **Largest item in any of the
   five scans.** Bit-identical; additive API.
2. **§5.6 N3 — `fuse/fuse.go:101` + `fuse/rsf.go:80` → `slices.SortStableFunc`,
   presize both maps.** 14% of a retrieval-only query, one function, verified
   identical. (Do this *with* §3.6's other 8 sites — same change, 10 call sites.)
3. **§5.6 N2 — `Tokenizer.Encode` first-byte gate, slice instead of rebuild.**
   26.5× on that loop = **6.8% of an index run**, removes 13,908 allocs/corpus,
   and deletes code. Gate on `t.cleanText` for the bit-identity argument.
4. **memo §1 — `wordPiece` memoization.** ≈**16% of an index run**, the highest
   re-ranked item from the prior docs.
5. **§3.4 — `preTokenize` byte-slicing + drop the per-word `[]int32`.** ≈**13%**,
   composes directly with #4. Do 3, 4, 5 as one tokenizer pass.

**Phase B — real, measured, smaller:**

6. **§5.6 N5 — `NewFlatI8` row-streaming.** 2.02×, deletes an `n·d·4` allocation.
7. **§3.8 + [campaign #3]** — topk threshold + `keep == nil` hoist, ≈7% of a query.
8. **[campaign #12] — rewrite `dotI8AVX2`.** 18–24% of a retrieval-only query, and
   the dominant retrieval item at scale. Bigger than it looks here because n=1,566
   understates it — see §5.7 item 4.
9. **§3.1 + §3.2 + §4.8 — the chunker.** Still the best ratio-per-hour in the doc
   (1.62× for one afternoon in one file) — but **1.4% end-to-end**. Do it because
   it is cheap, not because it is big.
10. **§2's bm25 `(K1+1)` + per-doc norm** (25% of `Scores`) and **§3.7's single-map
    `Build`** (1.8% of an index run) — same file, land together, inside
    **[campaign #29]**'s posting-struct work. Fold in **[campaign #11]** here too,
    but as the **pooled/cleared accumulator**, which is where N7 shows the win
    actually is — not the touched-set list.

**Phase C — the transformer path.** Only matters once you rerank — at which point
it is 99.8%+ of the query and *nothing else in this document is visible*:

11. **§4.1 last-layer trim** (13–15% of the trunk, always applies to the rerank
    path), **[campaign #13]** SIMD transcendentals, **[campaign #22]** Q8 widen,
    **[campaign #14]** length bucketing, **[campaign #28]** `CrossEncoder` batch
    API, **memo §2** rope table. Then **§3.5** (`scorePaged` re-quantization) and
    **§4.4** (`encmetal`, two lines, 453 MB).

**Phase D — footprint and cold start, as its own project.** Not latency wins; the
difference between "runs on a laptop" and "doesn't":

12. **§5.6 N4 — `bm25.Index` serialization.** 67% of the flagship example's cold
    start, ~20 s at the 378k-chunk scale. **The library's biggest missing API.**
13. **§5.6 N6** (`bm25.Build` streaming input, ~5 GB peak), **§4.3** (`WriteTo`),
    **§4.5** (streaming quantize), **§4.2** (gate column-blocking), **§4.6**
    (preprocess).

**Gated:** §3.9's `QueryBatch` batching on **[campaign #12]**; the softmax-scale
fusion on **[campaign #13]**.

**Re-file, don't schedule:** **[campaign #5]** (index serialization, "Tier 0,
20–30×") is **0.083% of an index run**. Move it out of Tier 0 and into §4.3's
peak-RSS group, which is where its real value is.

**Do immediately, costs nothing:**

- Add the `sim`/`simIDs` **matching-units invariant** comment from §2. It prevents
  a silent graph-corruption bug that no test would catch.
- Add a **warm-up phase** to `bench/harness.go:68-94` (§5.6 N9: first query is
  1.5–2× warm, so today's `p99` is largely the cold first query).
- Annotate **[campaign #11]**'s "10–50×" with §5.6 N7's selectivity crossover, and
  **[campaign #5]**'s tier, before either gets cited as settled.

---

# Appendix — reproducing the compiler-mechanical sweep

The whole repo compiles under Go 1.24 with two five-line stubs. `mmap` uses only
`unix.Madvise` + `unix.MADV_WILLNEED`; `embed` uses only `norm.NFD.String`.

```bash
# copy the tree, drop the darwin-only and separate-module dirs, drop tests
tar cf - --exclude=gpu --exclude='chunk/treesitter' --exclude=examples \
         --exclude=benchmarks --exclude=scripts --exclude=docs . \
  | (cd /tmp/lens1/aikit && tar xf -)
find /tmp/lens1/aikit -name '*_test.go' -delete

# go.mod: go 1.24 + replace directives to ../xtext and ../xsys
# xtext/unicode/norm/norm.go:
#   package norm
#   type Form int
#   const NFD Form = 0
#   func (f Form) String(s string) string { return s }
# xsys/unix/unix.go:
#   package unix
#   const MADV_WILLNEED = 3
#   func Madvise(b []byte, advice int) error { return nil }

go build -a -gcflags='all=-d=ssa/check_bce/debug=1' ./... 2>&1 | grep 'Found Is'
go build -a -gcflags='all=-m'   ./... 2>&1 | grep 'escapes to heap'
go build -a -gcflags='all=-m=2' ./... 2>&1 | grep 'cannot inline'
```

**Lens 5** additionally needs a Model2Vec-shaped artifact (real `tokenizer.json` +
`config.json` + hand-written `model.safetensors` with `embeddings` F32[V,256],
`mapping` I64[V], `weights` F64[V]) so `embed.LoadFromFS` succeeds; harvest the
vocab from the corpus itself. `embed` is then exercised through its real code path.

The stubs are wrong (identity NFD, no-op madvise) — that is fine, because the
compiler's diagnostics depend on shapes and control flow, not on behaviour.
**Never run the resulting binary.**

Sort the BCE output by file, then intersect with a profile. The raw count is not
the signal; **checks-per-unit-of-real-work** is. A bounds check that guards 6,144
MACs is free; one that guards a single float32 store is not.

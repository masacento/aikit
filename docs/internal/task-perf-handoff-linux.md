# Task (aikit): perf campaign Phase A — the Linux/amd64 half

> **For:** the aikit Claude, in `~/tmcode/aikit`, on the **Ryzen 7 3700X**
> (Zen 2, 8C/16T, AVX2, **no VNNI, no AVX-512**), Nobara Linux.
>
> **Why this machine:** one item here is amd64-only and cannot exist on Apple
> Silicon; the rest are architecture-neutral but need homogeneous cores and stable
> thermals to measure honestly. See §6 for what is deliberately NOT in scope.
>
> **Read first, in this order:**
> 1. `docs/internal/task-perf-lens-scans.md` — §0 (yield summary), §5 (workload
>    scan + the Amdahl tables), §6 (the corrected order). **This is the map.**
> 2. `docs/internal/task-perf-memoization.md` — §1 is item A3 below.
> 3. `docs/internal/perf-campaign-2026-07-28.md` — reference for anything marked
>    **[campaign #N]**. Do not work top-down from its ordering; §6 of the lens doc
>    supersedes it.

---

## 0. The one rule that matters most

**Every prior perf decision in this repo was arbitrated end-to-end, not by
microbenchmark, and that discipline exists because microbenchmarks lied here
before.** `task-perf-linalg.md` records a worker pool that was built correctly,
measured well in isolation, and was *neutral-to-slower* end-to-end — it got
pulled. The lens scans produced a 3.5% "win" at `-count=3` that vanished at
`-count=6`.

So:

- **`-count=6` minimum, report min-of-N**, never a single run.
- **Ratios, not absolutes.** Absolute numbers on this box are not the numbers of
  record — the **M1 Pro is the arbiter of record** for this project, because every
  figure in the existing campaign is M1 Pro and comparability is most of the
  campaign's value. Your job is to land correct, verified changes with a credible
  Linux ratio; the Mac produces the numbers that go in the CHANGELOG.
- **If a change measures inside noise, say so and do not ship it.** A negative
  result written down is a deliverable. §6 of the lens doc lists eleven of them
  already; adding to that list is success, not failure.
- **Bit-identity is claimed per item below. Verify it, don't assume it.** Where an
  item says "bit-identical", write a differential test over the *whole* corpus, not
  a sample — the failure mode of these fixes is a rare input, not a common one.

Check the toolchain first: `go.mod` pins **go 1.26.5**. The docs record Go 1.26.3
on this box as of June. Update before anything else.

---

## 1. Step 0 — make it measurable. Do this before any optimization.

Nothing below is verifiable until this exists, and three benchmark files are
currently dead.

**0a. Revive the dead benchmarks.** These reference `../search/index.go`, a
leftover `ken` path. **There is no `search/` directory in aikit**, so `b.Skipf`
makes them pass green and `bm25.Tokenize`, `bm25.TopK`, and the default regex
chunker have *zero live benchmark coverage*:

- `bm25/bm25_bench_test.go:19` (skips at `:22`)
- `chunk/regex/chunker_bench_test.go:43`
- `chunk/treesitter/cast_bench_test.go:47`

Separately, the chunk benches use `../../../testdata/repo/…` — from
`<root>/chunk/regex` that resolves to the **parent of the repo root**, so they
would skip even in a full checkout. Fix both. aikit's own `*.go` tree is a fine
fixture and is what every measurement in the lens docs used.

**0b. Fix `bench/harness.go:68-94`.** It measures exactly one thing: sequential
single-query latency. Add (i) a **warm-up pass** — measured first-query cost is
**1.5–2× warm** even at a 400 KB index, so today's `p99` over a small query set is
largely the cold first query; (ii) `runtime.ReadMemStats` deltas so allocation
regressions are visible; (iii) a **`-qps` concurrent mode**, without which
`FlatI8`'s internal goroutine fan-out looks free when under real load it
oversubscribes.

**0c. Build the end-to-end workload benchmarks.** This is what makes every ratio
below meaningful, and it does not exist yet. Two benchmarks over aikit's own tree
(322 `.go` files, ~1.65 MB → ~1,566 chunks at chunkSize 1500):

- **W1 — index a repo:** `chunk.ChunkFile` → `bm25.Tokenize` → `bm25.Build` →
  `embed.StaticModel.Encode` per chunk → `ann.NewFlatI8` → `MarshalBinary`, with
  a sub-benchmark per stage so you get the Amdahl split directly.
- **W2 — hybrid query:** `bm25.Tokenize` → `embed.Encode` → `bm25.TopK` +
  `FlatI8.Query` → `fuse.RRF`, k=50.

Run `go test -cpuprofile` + `go tool pprof -top` on W1 once and **write the stage
table into `docs/internal/`**. The lens-5 table was measured on a 2-core Xeon with
a *synthetic* 7,857-entry vocab; yours will be the first one on real hardware with
the real checkpoint. Expect the tokenize/pool split near 63/37 — if it moves past
~55/45, items A2 and A3 shrink proportionally and you should re-rank before
proceeding.

---

## 2. Phase A — in this order

Each item: one commit, one differential test, one before/after benchmark. Stop and
report if any item's measured ratio is under half its stated value.

### A1 · `StaticModel.EncodeBatch` — the largest item in the entire campaign

**Where:** `embed/model.go:170` is the *whole* public encode surface —
`Encode(text string) []float32`, one string at a time. Compare
`encoder/model.go:149`: `EncodeBatch(texts []string, isQueries []bool, concurrency int)`.
The slow transformer model got worker fan-out; the bulk-corpus model — whose own
doc comment at `embed/model.go:179` cites "a 378k chunk corpus" — got nothing.

So every caller writes a serial loop, including both shipped examples:
`examples/rag/main.go:82-85`, `examples/embedded-corpus/gen/main.go:73-75`, plus a
*second* serial pass at `rag/main.go:89-92`.

**Why it dominates:** `StaticModel.Encode` is **82% of an index run**. Chunking,
both tokenizers and pooling are pure functions of one chunk with no shared mutable
state (`StaticModel` documents "goroutine-safe for concurrent Encode"). Nothing
prevents fan-out except that no package offers it.

**Do:** add `EncodeBatch(texts []string, concurrency int) [][]float32` mirroring
`encoder/model.go:149`'s signature and its `concurrency <= 0 ⇒ NumCPU` convention.
Use it in **both** examples. Consider a variant writing into a caller-supplied flat
`[]float32`, which folds A6 in for free.

**Numerics:** bit-identical — `encodeIDs` touches no shared state. Assert exact
equality against the serial loop over the whole corpus.

**Expected:** 1.74× measured on 2 cores. **On your 8C/16T box expect 4–6×.** This
is the item your machine is best placed to measure — homogeneous cores, no P/E
straggler noise. **Sweep `concurrency` 1→16 and record the curve**; that curve is
the deliverable the M1 Pro cannot produce cleanly.

**API note:** additive, but it is new exported surface heading into v1.0. Get the
signature right the first time.

### A2 · `Tokenizer.Encode`'s added-token carve-out — 26.5×, 6.8% of an index run

**Where:** `embed/tokenize.go:233-250`. Per **byte** of every document it runs
`strings.HasPrefix(text[i:], k)` for all 5 `addedKeys`, then
`DecodeRuneInString` + `seg.WriteRune` to rebuild the entire document through a
`strings.Builder`.

**Mechanism:** `addedKeys` holds variable-length strings, so `HasPrefix` cannot be
specialised into byte compares — it lowers to a **`runtime.memequal` call**,
5 per byte. Confirmed in profile: `memeqbody` 4.22% flat, 30.77% of its callers
being `internal/stringslite.HasPrefix`. This is the identical mechanism found in
`chunk/regex.scanDepth`; **grep for that shape elsewhere while you're here.**

**Do:** precompute a `[256]bool` of added-key first bytes at `parseTokenizer`
(`tokenize.go:156-165`); skip bytes not in it; on a real match emit
`text[start:i]` as a **slice**, never a rebuild. For BERT every key starts with
`[`, so this collapses to `strings.IndexByte(text, '[')`.

**Numerics — read this carefully, the argument is subtle but holds.**
`WriteRune(DecodeRuneInString(…))` converts invalid UTF-8 to U+FFFD; slicing
preserves the raw bytes. But `encodeSegment:262` → `normalize:356` →
`cleanText:383` ranges over the string (also yielding U+FFFD) and **drops
`r == 0xFFFD` at `:387`**. Both paths erase them identically. **Gate the fast path
on `t.cleanText`** (true for every BertNormalizer config) or on
`utf8.ValidString`, and it is provably byte-identical. Fuzz it against the slow
path with invalid-UTF-8 inputs specifically.

**Expected:** 48.42 ms → 1.83 ms on the corpus (26.5×), 13,908 → 0 allocations.
Segmentation identical (1,655 segments / 1,649,310 bytes).

### A3 · `wordPiece` memoization — ≈16% of an index run

**Where:** `embed/tokenize.go:527`. Full spec in
`docs/internal/task-perf-memoization.md` §1 — read it, it covers the design.

**Headline:** greedy longest-match probes O(L²/2) prefixes per word, and **98.1% of
calls are repeats** (502,677 words → 9,463 unique on aikit's own tree). Measured
3.75×, 77× fewer allocations, identical ids across all 502,677 words.

**The two things the memo doc flags that will bite:**

1. **Concurrency.** `Encode` is goroutine-safe and A1 makes it *actually*
   concurrent. Use a **sharded map** (N shards keyed by a cheap hash), not
   `sync.Map` — this is a shared read-mostly key set, which is the pattern
   `sync.Map` is worst at.
2. **Bound it.** 9,463 entries is trivial and natural text converges near vocab
   size, but adversarial input grows it unboundedly. Skip caching words longer
   than `maxCharsPerWord` (they short-circuit to `unkID` anyway).

Returning cached slices is safe: `encodeSegment:270` does
`append(ids, t.wordPiece(w)...)`, which copies. **Re-verify that if you touch
`encodeSegment`.**

**Caveat on the number:** the 3.75× was measured with a *synthesized* vocab.
Re-measure against `testdata/minilm-model`'s real 30k vocab before quoting it.

### A4 · `preTokenize` byte-slicing — ≈13% of an index run

**Where:** `embed/tokenize.go:486-504`. Rebuilds every token through a
`strings.Builder` when every emitted token is already a contiguous byte range of
the normalized text; `:495`'s `out = append(out, string(r))` heap-allocates **per
punctuation character** (confirmed by escape analysis:
`tokenize.go:495:30: string(r) escapes to heap`). Code is punctuation-dense.

Also drop the per-word `var out []int32` at `:545` — `encodeSegment:270` copies it,
so it is pure garbage.

**Expected, as a stack with A2/A3** (measured on the 2-core box, ids identical at
every step):

| variant | ms | allocs |
|---|---:|---:|
| current | 267 | 1,097,920 |
| + pool/defer hoisted to `encodeSegment` | 255 | 1,097,918 |
| + per-word `out []int32` removed | 233 | 530,752 |
| + `preTokenize` byte-slicing | **197** | **666** |

`preTokenize` alone: 74.8 → 28.2 ms (2.65×), 526,690 → 328 allocations.

**Do A2, A3 and A4 as one tokenizer pass with one shared scratch threaded through
`encodeSegment`.** `embed/tokenize_bpe.go:120-154` and
`embed/tokenize_unigram.go:223,271,280` have the identical per-piece allocation
shape (2.83× and ~65 ms/corpus respectively) — **all three backends allocate a
per-word result the caller immediately copies away.** One scratch fixes all three.
See lens doc §3.3 and §3.4.

### A5 · `fuse.RRF` — 14% of a retrieval-only query, one function

**Where:** `fuse/fuse.go:78,79,101` and identically `fuse/rsf.go:41,42,80`. Two
`make(map[K]…)` with **no size hint** so both rehash as they grow, plus
`sort.SliceStable` on the reflection path.

**Do:** presize both maps from the input lists; switch to `slices.SortStableFunc`.
**While you're there, do the other 8 sites** lens doc §3.6 enumerates —
`ann/flat.go:111,131`, `ann/flat_i8.go:144,160`, `bm25/query.go:95,116`,
`sparse/sparse.go:149,167` — same change, 10 call sites total, 3.3–4.6× each.

**Numerics:** bit-identical at all 10 sites by the argument `ann/hnsw.go:531`
already makes — every comparator's second key is a unique id, so each is a total
order and stability is irrelevant.

**Expected:** 34.3 → 15.7 µs at k=50 (2.18×), 22 → 7 allocations.
**Scope honestly:** `fuse.RRF` is O(shortlist), so this is a small-to-medium-corpus
finding. At n=1M it drops from 28.7% of a query to <0.1%. Say so in the commit.

### A6 · `ann.NewFlatI8` row-streaming — 2.02×

**Where:** `ann/flat_i8.go:74-78` allocates `make([]float32, n*d)`, copies every
input vector in, quantizes, and drops it. The escape hatch is documented two files
away and unused — `linalg/quant.go:35-38`: *"exposed so a loader can quantize each
row as it is dequantized, without buffering the whole f32 matrix."*

**Expected:** 100.14 → 49.68 ms at n=20,000 × d=256; 25.7 MB → 5.2 MB. At
378k × 256 the discarded `flat` array is **387 MB**. Bit-identical —
`QuantizeRowsInt8` *is* a loop over `QuantizeRowInt8`. Keep a single d-sized
scratch for the ragged-row pad/truncate contract at `flat_i8.go:64-67`.

---

## 3. Item B — `dotI8AVX2`, the amd64-only one. **This box or nowhere.**

**[campaign #12]** — 18–24% of a retrieval-only query, and the dominant retrieval
item at scale. Full analysis in the campaign doc §12.

**Where:** `linalg/dot_amd64.s:311-340`. It handles **16 int8 per iteration** with
128-bit `VMOVDQU` on a 256-bit ISA, feeds two `VPMOVSXBW` widenings into one
`VPMADDWD`, accumulates into a **single** register, and is top-tested with an
unconditional `JMP` back.

**Mechanism:** `VPMOVSXBW` is a shuffle-domain op on one port; two per 16 MACs is a
hard ceiling of **8 MAC/cycle** regardless of caching. Measured L1-resident at
d=768: `DotI8` **7.9 MAC/cycle** vs `Dot8x4` (f32!) **14.9**. The int8 kernel is
slower per MAC than the f32 register-blocked kernel — so on amd64 the int8 index
buys a 4× memory cut and converts almost none of it into speed.

**Do:** 32 B/iteration with full `VMOVDQU ymm`; `VPMADDUBSW`-based products
(cleanest: store codes as `uint8 = code+128` so `VPMADDUBSW(row_u8, query_i8)` is
directly usable and the `128·Σq` correction is one scalar per query); **4
independent accumulators**; bottom-tested loop.

**Numerics:** integer arithmetic ⇒ any reassociation is bit-exact. The existing
differential tests against `dotI8Scalar` gate it. Run `go vet ./linalg/` —
`asmdecl` validates every asm stack offset against the Go signature and CI runs it.

**Expected 2–3×.** Note the arm64 side is already correct
(`dot_i8dp_arm64.s`: 4 accumulators, 64-wide SDOT loop) — **this gap is exactly
what you'd predict from a project tuned on Apple Silicon.**

**What you cannot do here:** the VNNI variant (`VPDPBUSD`) needs Zen 4+/Ice Lake+.
Zen 2 has no VNNI. Write the AVX2 kernel; leave VNNI as a documented follow-up
with the CPUID gate stubbed, exactly as `cpu-acceleration.md` §4 already does.

---

## 4. Free items — do these immediately, they cost nothing

- **Add the `sim`/`simIDs` matching-units invariant comment** to `ann/hnsw.go:234-239`
  and `:508`. `selectHeuristic` cross-compares a node–node similarity against a
  node–query similarity; it is correct today only because in `Add`,
  `qv.scale == h.scales[id]` exactly. **Anyone "optimizing" the loop-invariant
  `qv.scale` out of `sim()` alone silently produces a different graph** — no test
  failure, no crash, just worse recall. Lens doc §2 has the full argument.
- **Annotate two campaign items before they get cited as settled:**
  **[campaign #11]**'s "10–50×" with lens §5.6 N7's selectivity crossover (real
  queries touch a median 12% / worst 81% of the corpus, not 1.2%, because
  `bm25/tokenize.go:159-191` emits compound + every sub-token) — the win is the
  allocation, not the selectivity. And **[campaign #5]**'s Tier-0 placement:
  it is **0.083% of an index run**, an RSS item, not a latency item.
- **Fix the stale line** at `docs/internal/cpu-acceleration.md:163` — it says
  `forward_q8.go` still has the old scalar scores·V loop. It doesn't;
  `forward_q8.go:170,180` and `:238,250` both route through `s.mm`.

---

## 5. Definition of done for this phase

1. Step 0 landed: benchmarks live, harness has warm-up + allocs + QPS, W1/W2
   end-to-end benchmarks exist, **and the real-hardware Amdahl table is written
   into `docs/internal/`.**
2. A1–A6 and B landed, each with a differential test over the full corpus and a
   before/after `-count=6` number.
3. A short results doc — same shape as `task-perf-linalg.md`'s "RESULT" section —
   listing per item: measured ratio on this box, whether bit-identity held, and
   **anything that measured inside noise**, which gets written down as a dead end
   rather than quietly dropped.
4. Full suite green: `go test ./...`, `go test -race ./...`,
   `go test -tags aikit_checks ./...`, `go vet ./...`, and the Python parity pins in
   `scripts/` for anything numerical.
5. **Hand back to the Mac** for the numbers of record and the NEON half.

---

## 6. Explicitly NOT in scope on this box

**Needs arm64 — do on the MacBook:**
`[campaign #13]`'s NEON `expF32`/`erfF32`/`tanhF32`; `[campaign #25]`'s `Dot2x8`
→ 4×4 reshape; `[campaign #19]`'s `FRINTA` quantizer; `[campaign #37]`'s
by-element `FMLA` outer-product kernel; anything in `gpu/annmetal`.

**Will mislead you on this box:** `[campaign #26]` (`math.Round` is not an amd64
intrinsic but *is* an arm64 one) is an amd64-only win that measures at zero on the
Mac — fine to do here, just don't expect the Mac to confirm it. Conversely
`linalg`'s arm64 b-row packing and the `Dot2x8` path don't execute here at all.

**Unvalidatable on either machine — don't let them block anything:**
`[campaign #35]` VNNI (needs Zen 4+/Ice Lake+), `[campaign #36]` i8mm/`SMMLA`
(needs M2+/Graviton3+; the M1 Pro predates `FEAT_I8MM`).

**Do not re-chase — measured dead ends.** Full list in lens doc §6 obs. 5. The ones
most likely to look tempting from here:

- The spin-park worker pool. Built, measured end-to-end by goinfer, **pulled.**
- Bounds-check elimination in `accumRowRange` — 8 checks guard ~6,144 MACs.
- `topk.Push`'s bounds check — removable, measured at **exactly zero**.
- Both rank-invariance hoists (`hnsw.sim`'s `qv.scale`, `w8a8Span`'s `aScales[i]`) —
  provably valid, measured at 0% at three dimensions each.
- Compiling the chunker's regexes into one alternation — **9× slower.**
- Fusing the tokenize and embed passes in the index loop — **5.2% slower.**
- Overlapping `ann.Query` with `bm25.TopK` — **21% slower** (may flip at ~10×
  corpus; it does not flip at repo scale).
- Contiguous vs scattered vector storage — identical within noise.
- ~~Pooling HNSW's search heaps alone — slightly *slower*.~~ **OVERTURNED
  2026-07-31: −27.1% time and 19 → 2 allocations per query** when rebuilt and
  measured with an in-process A/B (campaign §7.41). This entry recorded a
  conclusion with no numbers and no method, so why the two readings differ cannot
  be reconstructed. **That is the lesson: a negative needs its numbers written
  down as fully as a positive**, or it becomes an unfalsifiable "don't bother".
- `layerNorm` vectorization — no bit-identical variant exists, and it's ~0.5% of a
  forward.

**Phase D (footprint / cold start) is deliberately deferred**, including
`bm25.Index` serialization — which is the library's biggest missing API (67% of the
flagship example's cold start) but is a design decision, not a Linux-box task.

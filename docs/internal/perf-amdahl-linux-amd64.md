# Amdahl table — real hardware, real checkpoint (Linux/amd64)

> **Step 0c of [`task-perf-handoff-linux.md`](task-perf-handoff-linux.md).** The
> prior stage table ([`task-perf-lens-scans.md`](task-perf-lens-scans.md) §5) was
> measured on a **2-core Xeon with a synthetic 7,857-entry vocabulary**. This is
> the first one on the target hardware with the real Model2Vec checkpoint.
>
> **Box:** AMD Ryzen 7 3700X (Zen 2, 8C/16T, AVX2, no VNNI/AVX-512), Nobara
> Linux, Go 1.26.5. **Method:** `-benchtime 2s -count=6`, **min of 6** reported;
> observed spread ≤ 7.3%, most ≤ 6%, consistent with the box's ~5% drift floor
> ([`measuring-performance.md`](measuring-performance.md) §1.6).
>
> **Absolute numbers here are not the numbers of record** — the M1 Pro is. These
> are for ranking work, and ratios are what transfer.

Reproduce with:

```sh
go test ./bench/  -run XXX -bench 'BenchmarkW1|BenchmarkW2' -benchtime 2s -count=6
go test ./embed/  -run XXX -bench BenchmarkEncodeSplit    -benchtime 2s -count=6
go test ./bench/  -run XXX -bench BenchmarkW1/sum -benchtime 6x -cpuprofile w1.prof
```

**Corpus:** aikit's own tree — 375 `.go` files, 2,009,777 bytes → **1,905
chunks** at chunkSize 1500 via the `regex` chunker; `testdata/model` (Model2Vec,
vocab 61,826, dim 256). `benchmarks/` is excluded as a separate module.

---

## 1 · W1 — index a repository

Serially, one chunk at a time, as both shipped examples do it.

| stage | call site | ms | % of run |
|---|---|---:|---:|
| `chunk.ChunkFile("regex", …)` ×375 | `chunk/registry.go:120` | 20.53 | 5.36% |
| `bm25.Tokenize` ×1905 | `bm25/tokenize.go:129` | 18.16 | 4.75% |
| `bm25.Build` | `bm25/index.go:47` | 30.24 | 7.90% |
| **`embed.StaticModel.Encode` ×1905** | `embed/model.go:170` | **297.85** | **77.82%** |
| ↳ `Tokenizer.Encode` | `embed/tokenize.go:218` | 181.01 | 47.30% |
| ↳ `encodeIDs` (gather + pool + L2) | `embed/model.go:190` | 106.69 | 27.88% |
| ↳ seam (the `ids` slice handoff) | — | 9.67 | 2.53% |
| `ann.NewFlatI8` | `ann/flat_i8.go:68` | 4.19 | 1.10% |
| `MarshalBinary` | `ann/flat_i8_persist.go:43` | **0.069** | **0.018%** |
| stage sum | | 371.04 | 96.94% |
| unattributed (GC, outer appends) | — | 11.70 | 3.06% |
| **end-to-end (measured)** | | **382.73** | **100%** |

The decomposition is 96.9% complete. The 3.1% gap is GC plus the `[][]string` /
`[][]float32` accumulation that `sum` performs and the per-stage benchmarks reuse
pre-built — a real cost of the pipeline that belongs to no single stage.

### The tokenize / pool split — 62.9 / 37.1

This is the number the handoff gates Phase A on, so it is measured **twice, by
independent methods**, and they agree to 0.1 pp:

| method | Tokenizer.Encode | encodeIDs | split |
|---|---:|---:|---|
| `BenchmarkEncodeSplit` (direct, `-count=6`) | 181.01 ms | 106.69 ms | **62.9 / 37.1** |
| `pprof` cum on W1 | 46.20% | 27.39% | **62.8 / 37.2** |

Predicted 63/37 (lens §5, cross-checked against memoization doc §1's 65/35). It
has **not** moved past the ~55/45 re-rank trigger.

`BenchmarkEncodeSplit` measures both halves directly rather than measuring
`Encode` and subtracting — `encodeIDs` is unexported, which is why that benchmark
lives in package `embed`. The alternative attributes the seam and every
mismeasurement to whichever half you were less careful about. Here the seam is
its own line (2.53%: the `[]int32` `Tokenizer.Encode` returns and `Encode` hands
on).

**Independent cross-check on the harnesses.** `BenchmarkEncodeSplit` uses a
plain line-splitter (1,551 chunks) while W1 uses the real `regex` chunker (1,905
chunks) over the same bytes. `W1/embedEncode` = 297.85 ms, `EncodeSplit/whole` =
297.37 ms — **0.16% apart**. Embedding cost is a function of total text, not of
chunk boundaries, and two separately-written harnesses agree on it.

### Where the time goes inside the tokenizer

`pprof -top -cum` on `BenchmarkW1/sum`, cumulative % of the whole run:

| | cum % of W1 |
|---|---:|
| `StaticModel.Encode` | **73.93%** |
| ↳ `Tokenizer.Encode` | 46.20% |
| ↳↳ **added-token carve-out loop** (`tokenize.go:238-255`) | **10.23%** |
| ↳↳ `encodeSegment` | 35.97% |
| ↳↳↳ `normalize` (of which `cleanText` 7.26%) | 14.52% |
| ↳↳↳ `preTokenize` | 11.88% |
| ↳↳↳ `wordPiece` (**already memoized** — see §3) | 7.59% |
| ↳ `encodeIDs` | 27.39% |
| `bm25.Build` | 5.94% |
| `bm25.Tokenize` | 3.96% |
| `chunk/regex.scanDepth` | 3.30% |
| GC (`gcBgMarkWorker`) | 6.27% |

The carve-out figure is line-level, from `pprof -list`:

```
     10ms       10ms    238:	for i := 0; i < len(text); {
     30ms       30ms    240:		for _, k := range t.addedKeys {
     40ms      140ms    241:			if strings.HasPrefix(text[i:], k) {
     30ms       30ms    246:		if matched != "" {
     30ms       40ms    252:		r, size := utf8.DecodeRuneInString(text[i:])
        .       50ms    253:		seg.WriteRune(r)
        .      1.09s    256:	flush()          ← the actual tokenization
```

310 ms of 3.03 s. `memeqbody` is 2.64% flat with `stringslite.HasPrefix` among
its callers, exactly the mechanism the handoff describes.

---

## 2 · W2 — hybrid query, retrieval only (n=1905, k=50)

| stage | call site | ns | % |
|---|---|---:|---:|
| `bm25.Tokenize(query)` | `bm25/tokenize.go:129` | 358 | 0.3% |
| `embed.Encode(query)` | `embed/model.go:170` | 7,667 | 6.9% |
| `bm25.TopK` | `bm25/query.go:73` | 28,133 | 25.3% |
| `ann.FlatI8.Query` | `ann/flat_i8.go:88` | 40,593 | 36.5% |
| **`fuse.RRF`** | `fuse/fuse.go:56` | **25,966** | **23.3%** |
| stage sum | | 102,718 | 92.3% |
| **end-to-end (measured)** | | **111,230** | **100%** |

The 7.7% gap is the two `fuse.Keys` projections `sum` performs, which the
`fuseRRF` sub-benchmark hoists out of its loop.

**Read W2 with the lens doc's W2R warning attached:** once a rerank is in the
pipeline, the entire retrieval stack is between 1/500th and 1/1500th of the
query, and every row above is noise.

---

## 3 · Re-ranking Phase A against these measurements

| item | handoff's predicted end-to-end | **measured stage weight** | expected at the stated ratio | verdict |
|---|---:|---:|---:|---|
| **A1** `EncodeBatch` fan-out | 82% of run, 4–6× | **73.93%** | **−55 to −62% of W1** | **#1 by a wide margin. Confirmed.** |
| **A4** `preTokenize` byte-slicing, 2.65× | ≈13% | **11.88%** | −7.4% | **#2. Confirmed.** |
| **A2** carve-out rebuild, 26.5× | 6.8% | **10.23%** | −9.8% | **#3 — and LARGER than predicted.** |
| **A5** `fuse.RRF` presize, 2.18× | 14% of a query | **23.3% of W2** | −12.6% of a retrieval query | larger than predicted; still W2-only |
| **A6** `NewFlatI8` row-streaming, 2.02× | — | **1.10%** | −0.55% | a memory item, not a latency one |
| ~~**A3**~~ `wordPiece` memo | ≈16% | — | — | **ALREADY LANDED** — see below |

### A3 is done, and that is why the profile disagrees with the prediction

`wordPiece` measures **7.59%** of a W1 run against the handoff's 22.7% stage
weight. That is not a failed prediction: `6b69133 embed: memoize wordPiece —
4.59× on the real vocab, byte-identical (memo §1)` is already in the tree, with
the sharded map and the per-shard bound the memo doc asked for, and it beat its
own 3.75× estimate on the real vocabulary. **7.59% is the post-memoization
residual**, and `pprof -peek` shows 52% of what remains is the cache lookup
itself.

The general point, which cost nothing here only because the profile happened to
name `wpCache.get`: **a handoff doc describes the tree as of when it was
written.** Check each item against the code before pricing it.

### What the free items look like from here

- **`MarshalBinary` is 0.018% of an index run** — five times smaller even than
  the lens doc's 0.083%, and it confirms the handoff's §4 note that
  **[campaign #5]**'s Tier-0 placement is wrong. It is an RSS item.
- **`bm25.TopK` is 25.3% of W2**, measured *after* item 39's WAND landed. The
  retrieval half of a query is now split roughly evenly between the two
  retrievers and the fusion.

---

## 4 · A1 — measured

`StaticModel.EncodeBatch`, landed. **8.21× at NumCPU**, against a predicted
4–6×. Sweep over the real corpus, min of 6:

| concurrency | ms | speedup | % of linear |
|---|---:|---:|---:|
| serial `Encode` loop | 294.70 | 1.00× | — |
| 1 | 295.03 | 1.00× | 100% |
| 2 | 161.38 | 1.83× | 91% |
| 3 | 111.56 | 2.64× | 88% |
| 4 | 84.71 | 3.48× | 87% |
| 6 | 58.50 | 5.04× | 84% |
| **8** (physical cores) | 47.05 | **6.26×** | 78% |
| 12 | 39.36 | 7.49× | 62% |
| **16** (SMT threads) | 35.91 | **8.21×** | 51% |

Two things the curve says that a single number would not. Scaling holds at
84–91% of linear out to 6 workers and is still 78% at 8, so this really is
embarrassingly parallel work rather than something bounded by shared state. And
SMT is worth having: 8 → 16 threads buys a further **1.31×** on top of the
physical-core figure. `c=1` matching the serial loop to 0.1% confirms the
single-worker path costs nothing.

End to end on W1:

| | serial | batched | |
|---|---:|---:|---:|
| embed stage | 311.45 ms | 36.21 ms | **8.60×** |
| **whole index run** | **387.78 ms** | **109.76 ms** | **3.53×** |

Allocations are unchanged — 757,044 → 757,065 for the whole run, the ~20 being
the goroutines themselves.

### What A1 does to the ranking of everything after it

The embed stage falls from **77.8% to 33.0%** of an index run, and the
decomposition of the batched run is 99.6% complete:

| stage | ms | % of batched run |
|---|---:|---:|
| `embed.EncodeBatch` | 36.21 | 33.0% |
| **`bm25.Build`** | **30.24** | **27.6%** |
| `chunk.ChunkFile` | 20.53 | 18.7% |
| `bm25.Tokenize` | 18.16 | 16.5% |
| `ann.NewFlatI8` | 4.19 | 3.8% |
| `MarshalBinary` | 0.069 | 0.06% |

**The largest remaining stage after A1 is `bm25.Build`, which is not in Phase A
at all** — it is lens doc §3.7 (three map operations per (document, term),
1.33×). The tokenizer items keep their share *within* the embed stage, but that
stage is now a third of the run rather than three quarters, so measured against
a batched index run:

| item | share of a serial run | share of a **batched** run |
|---|---:|---:|
| A2 carve-out | 10.23% | ≈4.2% |
| A4 `preTokenize` | 11.88% | ≈4.9% |
| lens §3.7 `bm25.Build` | 7.90% | **27.6%** |
| lens §3.1/§3.2 chunker | 5.36% | **18.7%** |

This is [`measuring-performance.md`](measuring-performance.md) §1.18 in its
ordinary form — an earlier item spends a later one's win — and it is worth
deciding explicitly whether Phase A's remaining order still holds, because two
items that were not in it now outrank two that are.

---

## 5 · A5 — measured, and it splits in two

The item bundled ten call sites under one ratio. They do not behave the same way,
and the bundling is the finding.

### The two `fuse` sites: 4.80×

| | before | after | |
|---|---:|---:|---:|
| `RRF` k=50, 2 lists | 46.42 µs | 8.685 µs | **5.34×** |
| `RRF` k=10 | 6.768 µs | 1.511 µs | 4.48× |
| `RRF` k=200 | 255.3 µs | 40.66 µs | 6.28× |
| `RRF` k=1000 | 1658.9 µs | 292.1 µs | 5.68× |
| `RSF` k=50 | 47.69 µs | 9.177 µs | 5.20× |
| geomean | | | **5.36×** (−81.3%) |

Allocations 22 → 4 at k=50 (−82%). Predicted 2.18× and 22 → 7.

On the real query path: `fuse.RRF` **25.97 → 5.41 µs (4.80×)**, and the whole W2
retrieval query **115.6 → 92.4 µs (1.25×, −20.0%)** with bytes per query −26%.

The presize was the smaller half of it. The larger half was removing a map
lookup from the sort comparator: first-appearance order used to live in a second
map consulted on every comparison — O(n log n) lookups — and now lives in the
slice's own positions, so a stable sort on score alone reproduces it exactly.
One map instead of two, no projection pass, and `slices.SortStableFunc` instead
of `sort.SliceStable`.

### The other eight sites: inside noise

`ann/flat.go`, `ann/flat_i8.go`, `ann/flat_binary.go`, `bm25/query.go`,
`bm25/wand.go`, `sparse/sparse.go` — `sort.Slice`/`SliceStable` → `slices.SortFunc`.
Controlled A/B on the paths where they run, `-count=6`:

| | before | after | |
|---|---:|---:|---:|
| `FlatI8.Query` (k=50) | 41.60 µs | 40.40 µs | −2.9% |
| `bm25.TopK` (k=50) | 28.77 µs | 29.47 µs | **+2.5%** |

Opposite directions, both a hair outside the ~5% drift floor. **The handoff's
"3.3–4.6× each" does not survive contact with the end-to-end path**, and the
honest reading is that these eight measured at zero. Bytes per query fell ~2%
(p=0.002), which is the only consistent signal.

The mechanism is nonetheless real, which is worth separating from the outcome.
`BenchmarkSortSites` isolates it on the same element type and comparator:

| n | `sort.Slice` | `slices.SortFunc` | |
|---:|---:|---:|---:|
| 10 | 822 ns | 86 ns | 9.52× |
| 50 | 3369 ns | 1143 ns | 2.95× |
| 1000 | 109.4 µs | 55.6 µs | 1.97× |

So why does 2.2 µs of isolated saving at n=50 not show in a 41 µs stage? Because
the fixture sorts RANDOM data with many ties, and the real sites sort the array
`topk.Selector.Result` returns — a heap, already partially ordered. pdqsort and
reflect-based quicksort respond very differently to partially-ordered input. A
microbenchmark on random data does not predict either one here.

**They were kept anyway, and not as a performance claim.** They delete eight
duplicated comparator closures in favour of three shared functions, they finish
the change audit #24 made in `hnsw.go` and never propagated, and no site got
slower beyond noise. The commit says they measured at zero.

---

## 6 · `bm25.Build` single-map — the item A1 promoted, 1.27×

Not a Phase A item at all: [`task-perf-lens-scans.md`](task-perf-lens-scans.md)
§3.7, taken next because A1 made `bm25.Build` the largest remaining stage of an
index run (27.6%).

| | before | after | |
|---|---:|---:|---:|
| `Build` (200k docs × 120 tokens) | 80.76 ms | 62.21 ms | **1.30×** |
| `BuildReal` | 1214.3 µs | 954.2 µs | **1.27×** |
| `W1/bm25Build` | 31.33 ms | 24.69 ms | **1.27×** |
| geomean | | | **−22.2%** |

Predicted 1.33×; measured in band. Allocations are unchanged (−0.3%) and bytes
fall 3–16% on the isolated benchmarks.

`m[k] = append(m[k], v)` is a mapaccess *plus* a mapassign — two independent
hashes of the same string — and `df[k]++` was a third and fourth in a second
map, with the intern check a fifth. Five hashes of the same key per (document,
term), 23.9 M times on the campaign's corpus. Now one map from term to an index
into a `[]termEntry`, so a single probe reaches the posting list, the document
frequency, and both extrema.

Two things beyond what the lens doc described. The `termStat` map item 39 added
is **gone**, absorbed into `termEntry`: `maxTf` and `minLen` are tracked as
`Build` goes, which also deletes the second pass over every posting list that
built them, and removes a third map probe per query term. And the indirection is
a slice index rather than a `map[string]*termEntry` — the lens doc's shape added
~30k allocations for a 30k-term corpus, which is why its allocation count went
up; this one's does not.

Allocation note: the W1 stage's bytes rose 2.8% (the `entries` slice), against a
21% time win.

---

## 7 · The chunker — 1.76×, the other item A1 promoted

[`task-perf-lens-scans.md`](task-perf-lens-scans.md) §3.1 + §3.2, taken because
A1 left `chunk.ChunkFile` at 18.7% of a batched index run.

| | before | after | |
|---|---:|---:|---:|
| `Chunker_Go` | 510.8 µs | 304.1 µs | **1.68×** |
| `Chunker_TypeScript` | 1.252 ms | 1.098 ms | 1.14× |
| `Chunker_Python` | 17.82 µs | 16.01 µs | 1.11× |
| **`W1/chunk`** | **20.87 ms** | **11.84 ms** | **1.76×** |

Predicted 1.62× combined; measured 1.76× without the presized `lineStart` the
lens doc also suggested.

**§3.1, `scanDepth`.** The per-byte closure became a scalar compare against the
next line start, and each `hasPrefixAt` is gated on its first byte. `cmtMark` is
a string *variable*, so `string(src[i:i+len(s)]) == s` cannot be specialized into
byte compares — it lowers to a `runtime.memequal` CALL, made once per byte of
source. Gating cannot change the outcome, since `hasPrefixAt` already requires
`src[i] == s[0]`.

**§3.2, the literal-prefix prescreen — and the lens doc is wrong about how.**
It says to use `regexp.LiteralPrefix()` and lists `^func\b`→`func`. It does not:
`LiteralPrefix` reads the *compiled program's* extracted prefix, and a following
`\b` or `\s` blocks that extraction. Measured, it returns `""` for `^func\b`,
for `^class\s+\w+`, and for **all four of Go's definition patterns** — only the
pure-literal comment rules got anything, 3 of 7 for Go.

The syntax tree has what the compiled program lost: `syntax.Parse("^func\b")`
simplifies to a concatenation of `\A`, the literal `func`, and a word boundary,
so walking its leading literals gives `func` directly. That lifts coverage from
3/4/1/6/4 patterns (go/java/python/rust/typescript) to **7/5/3/8/6**.

Soundness is by anchoring: every pattern here starts with `^`, asserted at
registration with a panic, so "the match begins with p" and "the line begins
with p" are the same statement. Gated over **5.9 M (pattern, line) pairs** —
every rule of every language against every line of the package fixtures *and*
every `.go` file in the repository, checking that a rejected line never matches.
The prescreen fires on 55.8% of pairs. A mutant that descends into an optional
group dies; the case-folding guard needed its own unit test, since no rule today
uses `(?i)` and the corpus-wide gate cannot see it.

TypeScript gains least because its patterns lead with `(export\s+)?`, an optional
group, which admits no literal prefix at all. Bounding those needs a first-byte
SET computed from the syntax tree rather than a prefix — ~~left undone, and the
reason TypeScript is 1.14× where Go is 1.68×~~.

**DONE (2026-07-31, on the M1 Pro but architecture-neutral): `anchoredFirstByteSet`.**
It computes FIRST(pattern) — the bytes a match can begin with — over the syntax
tree with nullability, so an optional leading group contributes its first byte
*and* lets the next element contribute (`(export\s+)?…class` → {e,d,a,c,…}). Used
as the fallback screen where the literal prefix is empty; a match is ^-anchored so
a first byte outside the set provably can't match. Bail-safe (non-ASCII / `.` /
case-fold → no screen), and the set is always a SUPERSET so an imprecise walk only
screens less. Soundness gated the same way as the prefix — **6.02 M (pattern,
line) pairs, the FB screen firing on 2.11 M of them, none a real match** — and
mutation-checked (an under-approximating FIRST is caught). **TypeScript −28.4%
(1.40×), Python −12.7%**; Go unchanged (all-literal prefixes, never reaches the
set). It also caught Rust/Java, whose modifier-led patterns have the same shape.
The one pattern still unscreened is TS's method-modifier rule
(`^(public|private|…|\*|\s)*[A-Za-z_$]…`), whose set is near-universal — correctly
returned nil rather than shipped as a no-op screen.

---

## 8 · Item B — already done

`dotI8AVX2` was rewritten before this handoff was picked up:
`1b9c88a linalg: rewrite dotI8AVX2 — 2.10x kernel, 2.02x scan (perf item 12 /
finding B)`, in the predicted 2–3× band. 64 int8 per iteration, four independent
accumulators, bottom-tested, and `VPMOVSXBW` taking its m128 source straight from
memory so the explicit loads are gone.

Its comment already records the analysis the handoff asks for: `VPMADDUBSW`
would remove the widening entirely, but u8×i8 pair sums can exceed int16 and it
**saturates**, so that route needs range-limited codes and belongs with the VNNI
work. (The way past that is `VPABSB`/`VPSIGNB` — |a| as the unsigned operand and
the sign folded into b keeps every pair sum under 2·127² = 32258 — which is worth
recording as the shape a future attempt should take.)

**This is the second Phase-A item found already landed**, after A3. Same lesson,
now twice: a handoff describes the tree as of when it was written.

---

## 9 · A4 — `preTokenize` byte-slicing, 1.95×

| | rebuild | sliced | |
|---|---:|---:|---:|
| `preTokenize` over the corpus | 71.63 ms | 36.76 ms | **1.95×** |
| allocations | 663,488 | 15,436 | **43× fewer** |

Predicted 2.65×; measured 1.95×. On the stage around it: `Tokenizer.Encode`
181.01 → 163.29 ms (**1.11×**) with allocations 684,645 → 53,290 (**−92%**), and
the whole `Encode` 297.37 → 281.13 ms (−5.5%).

Every token preTokenize emits is already a contiguous byte range of its input, so
the `strings.Builder` rebuild was pure copying — plus a `string(r)` heap
allocation per punctuation character, which is most characters in source code.

**The two paths differ on invalid UTF-8**, and only there: ranging a string
yields U+FFFD for a bad byte, so `WriteRune` rebuilt it as the three-byte
replacement character while slicing preserves the raw byte. In practice
`normalize` → `cleanText` removes those before `preTokenize` sees them, but
`cleanText` is a config flag. Rather than gate on the flag, the sliced path runs
`utf8.ValidString` and falls back to the rebuild — measured at **1% of the sliced
path**, which buys exactness for every input instead of for the configurations
someone checked.

**A4 also created a hazard that did not exist before.** Tokens are now views into
the normalized chunk, and `wpCache.put` stores the token as a map key — so a
cached word would pin its whole ~1.5 KB chunk, up to 8192 entries per shard. The
key is cloned now, which is the same fix `bm25.Build` documents, arriving here
the moment `preTokenize` stopped materializing its tokens.

A4's second half — dropping the per-word `var out []int32` — **no longer
applies**: `wordPieceCompute`'s slice is retained by A3's memo, so it cannot be a
shared scratch.

---

## 10 · A2 — the added-token carve-out, 1.22× on the stage

| | before | after | |
|---|---:|---:|---:|
| `Tokenizer.Encode` (stage) | 163.29 ms | 133.40 ms | **1.22×** |
| allocations | 53,290 | 38,371 | −28% |
| whole `StaticModel.Encode` | 281.13 ms | 253.46 ms | −9.8% |

The scan tested every added key at every BYTE of the document and rebuilt the
document through a `strings.Builder` on the way past. `addedKeys` holds
variable-length strings, so `strings.HasPrefix` lowers to a `runtime.memequal`
CALL — five per byte — and the segments it built were already contiguous ranges
of the input. Now a `[256]bool` of possible first bytes gates the scan, and
because this checkpoint's keys (`[PAD]`, `[UNK]`) share one first byte it
collapses to `strings.IndexByte`.

**Predicted 26.5×; the carve-out itself measured roughly 4× in situ.** The gap is
what the prediction isolated: 48.42 → 1.83 ms was the loop alone, and in place
the loop still hands its segments to `encodeSegment`, which is most of what
remains. Measured against Step 0's profile the carve-out was 10.23% of an index
run and is now roughly 2–3% of one.

Same UTF-8 story as A4, and the same resolution: the rebuild's U+FFFD
re-encoding differs from slicing on invalid input, so `utf8.ValidString` routes
those to a verbatim copy of the old path. Scanning by BYTE rather than by rune is
what that gate buys — in valid UTF-8 an ASCII byte never appears inside a
multi-byte rune, so a byte equal to a key's first byte is necessarily at a rune
boundary, which is the only place the rune-stepping original would have tried.

---

## 11 · A6 — `NewFlatI8` row-streaming: the memory prediction was exact, the time one was not

| n (d=256) | staged | streamed | time | bytes |
|---:|---:|---:|---:|---:|
| 2,000 | 7.576 ms | 5.978 ms | 1.27× | 2512 KiB → 512 KiB |
| 20,000 | 50.21 ms | 39.41 ms | 1.27× | 24.49 MiB → 4.96 MiB |
| 200,000 | 446.2 ms | 380.8 ms | 1.17× | 244.9 MiB → 49.6 MiB |

Predicted **2.02× and 25.7 MB → 5.2 MB at n=20,000**. The memory figure is right
to three digits — **−79.7% at every scale**, allocations 4 → 3 — and the time
figure is 1.6× optimistic. That matches how this was classified at Step 0: 1.10%
of an index run, a footprint item rather than a latency one.

The staged shape allocated an n·d float32 block, copied every vector in,
quantized it and dropped it — 387 MB discarded at the 378k×256 the package doc
cites, to produce a 97 MB index. `QuantizeRowsInt8` is a loop over
`QuantizeRowInt8`, and `quant.go` says as much ("exposed so a loader can quantize
each row as it is dequantized, without buffering the whole f32 matrix"), so
streaming is bit-identical rather than merely equivalent — asserted against the
staged path including the ragged cases.

The one-row scratch that remains exists for the ragged-row contract, not for
speed: a short vector is zero-padded and a long one truncated, and doing that in
place would mutate the caller's slice. A row already exactly d long skips it. A
mutant that drops the `clear()` between ragged rows dies; one that quantizes
short rows without padding survives and is *accidentally* equivalent, since
`QuantizeRowInt8` leaves the destination tail untouched and it was already zero —
undocumented behaviour the explicit padding does not rely on.

---

---

## W3 — cold start, and it overturns the lens doc's split

Lens §5.6 N9: "cold-start warm-up is real and nothing measures it." Now it is
measured — `bench/coldstart_bench_test.go`, min of 6, over
`examples/embedded-corpus`'s own assets (1,747 chunks, real potion-code-16M).

| stage | ms | % of run | peak MiB | MB allocated |
|---|---:|---:|---:|---:|
| **`embed.LoadFromFS`** | **62.51** | **76.3%** | **142.4** | 73.9 |
| `ann.LoadFlatI8` | 0.24 | 0.3% | 73.0 | 0.5 |
| `json.Unmarshal(corpus)` | 5.05 | 6.2% | 74.5 | 1.7 |
| `bm25.TokenizePlain` ×N | 8.83 | 10.8% | 79.9 | 3.3 |
| `bm25.Build` | 8.78 | 10.7% | 82.0 | 2.9 |
| **to first result** | **81.97** | 100% | **148.6** | 82.3 |

**The prior W3 has the split backwards.** Lens §5.4 put `embed.LoadFromFS` at
**11.1%** and `TokenizePlain + bm25.Build` at **66.9%**. Measured here they are
**76.3%** and **21.5%** — inverted. That table was built with a hand-written
`model.safetensors`; this one loads the real 64 MB checkpoint, and loading it is
what cold start *is*.

**This demotes N4.** Adding a serialization surface to `bm25.Index` is justified
in the lens doc as "67% of the flagship example's cold start". It is **21.5%**,
and eliminating it entirely would take 82.0 ms to 64.4 ms. Still real, no longer
the headline, and still an API design decision rather than a perf one.

Peak heap tells the same story: the model load alone reaches 142.4 MiB of the
run's 148.6, so everything after it adds ~6 MiB.

### `LoadMmap` — the escape hatch that existed and was unreachable

`OpenSafetensorsMmap` has been in `embed` the whole time and no load path reached
it: `LoadFromFS` goes through `fs.ReadFile`, which by definition cannot mmap, and
`Load` is a one-line wrapper around `LoadFromFS(os.DirFS(dir), ".")` that throws
away the one thing mmap needs — a real path. Every caller heap-read the whole
checkpoint. (Third time this pattern has appeared: A6's `QuantizeRowInt8` and
Phase 4's `Backend.MatmulBT` were the others.)

| | time | peak MiB | MB alloc |
|---|---:|---:|---:|
| `loadModel` (heap) | 59.5 ms | 142.5 | 73.9 |
| `loadModelMmap` | **48.8 ms** | **19.5** | **9.6** |
| cold start, heap | **84.8 ms** | 75.8 | 82.4 |
| cold start, mmap | 99.2 ms | **13.0** | **18.1** |

**Both directions are real and the item is a footprint one.** Peak heap falls
5.8× end to end and allocation 4.6×, while time-to-first-result rises **17%**:
the load itself is faster — no 64 MB read — but mmap defers the page faults to
the first `Encode`, and faulting 64 MB in costs more than reading it sequentially
from a warm page cache. Shipped as an additive `LoadMmap`, so no existing caller
changes behaviour and the trade is the caller's to make.

**Caveat on `peakMiB` comparisons.** The same `sum` arm measured 148.6 MiB in one
run and 75.8 in another, because the sampler starts from whatever heap state the
preceding sub-benchmark left. Adjacent arms within one run are comparable; the
same arm across runs is not. Read the pairs above, not the absolutes.

### The peak-heap instrument, and why it exists

`B/op` cannot arbitrate any of the remaining footprint findings — it counts bytes
ALLOCATED over a run, and a doubling of peak RSS can happen with no change in
that total while a pool can change the total with no change in peak. The
benchmarks report a `peakMiB` metric instead: a sampled maximum of
`HeapInuse`, which is a lower bound on peak heap and a loose proxy for peak RSS.
Approximate, but an approximation of the right quantity. The findings it exists
for predict doublings, so a sampler that might miss a microsecond spike is
adequate; where one turns out to hinge on a few percent, it is not, and that
should be said.

### Recorded negative: the first-query penalty is not measurable in-process

78.7 µs on a freshly loaded index against 82.7 µs warm — the cold arm is 5%
*faster*. Reloading the index leaves the code, allocator and model warm, so only
the data is cold, and 443 KB of it does not miss enough to show. A real
first-query penalty needs a genuinely cold process, which a Go benchmark cannot
be. The harness's warm-up pass (Step 0b) stands on its own measurement; this just
cannot reproduce it, and the query is 0.2% of cold start regardless.

---

## 12 · Where an index run stands

| | ms |
|---|---:|
| Step 0 baseline (serial, before any of this) | 382.73 |
| today, batched | **98.08** |

**3.90×** on "index a repository", from A1 + `bm25.Build` + the chunker + A4.
That is a cross-invocation comparison, so read it with the ~5% drift floor
attached; the per-item ratios above are all same-session A/Bs.

---

## 13 · Step 0 status

| | state |
|---|---|
| 0a · dead benchmarks revived | **already done** (campaign item 1) — verified live: `bm25`, `chunk/regex`, `chunk/treesitter` all produce numbers, none skip |
| 0b · harness warm-up + allocs + QPS | **already done** — `warmupQueries`, `AllocsPerQuery`/`BytesPerQuery`, `QPS`/`Concurrency` |
| 0c · W1/W2 workload benchmarks | **new** — `bench/workload_bench_test.go`, `embed/encode_split_bench_test.go` |
| 0c · real-hardware Amdahl table | **this document** |

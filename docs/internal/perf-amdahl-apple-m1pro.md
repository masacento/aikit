# Amdahl table — real hardware, real checkpoint (Apple M1 Pro)

> **Phase A hand-back arbiter table** ([`macbook-phase-a-handback.md`](../prompts/macbook-phase-a-handback.md)).
> The companion to [`perf-amdahl-linux-amd64.md`](perf-amdahl-linux-amd64.md),
> measured on the M1 Pro with the identical benchmarks and corpus. **This box is
> the arbiter of record**; the amd64 figures ranked the work, these are the ones
> to quote. Two comparable tables is the deliverable — read them side by side.
>
> **Box:** Apple M1 Pro (**6 P-cores + 2 E-cores**, no SMT), macOS, Go 1.26.5,
> `CGO_ENABLED=0`. **Method:** `-benchtime 2s -count=6`, **min of 6** reported;
> observed spread ≤ 3% on the large stages, consistent with a quieter thermal
> floor than the 3700X's ~5%.
>
> **Corpus:** aikit's own tree — 375 `.go` files, 2,009,777 bytes → **1,905
> chunks** at chunkSize 1500 via the `regex` chunker; `testdata/model` (Model2Vec,
> vocab 61,826, dim 256). Identical to the amd64 run. `benchmarks/` excluded.

Reproduce with:

```sh
go test ./bench/  -run XXX -bench 'BenchmarkW1|BenchmarkW2' -benchtime 2s -count=6
go test ./embed/  -run XXX -bench BenchmarkEncodeSplit    -benchtime 2s -count=6
go test ./bench/  -run XXX -bench 'BenchmarkW1/sum$' -benchtime 8x -cpuprofile w1.prof
```

---

## 1 · W1 — index a repository

Serially, one chunk at a time, as both shipped examples do it.

| stage | ms | % of run | amd64 % |
|---|---:|---:|---:|
| `chunk.ChunkFile("regex", …)` ×375 | 9.63 | 3.72% | 5.36% |
| `bm25.Tokenize` ×1905 | 13.64 | 5.26% | 4.75% |
| `bm25.Build` | 21.95 | 8.47% | 7.90% |
| **`embed.StaticModel.Encode` ×1905** | **212.78** | **82.12%** | 77.82% |
| ↳ `Tokenizer.Encode` | 103.38 | 39.90% | 47.30% |
| ↳ `encodeIDs` (gather + pool + L2) | 100.86 | 38.92% | 27.88% |
| ↳ seam (the `ids` slice handoff) | 3.83 | 1.48% | 2.53% |
| `ann.NewFlatI8` | 2.65 | 1.02% | 1.10% |
| `MarshalBinary` | 0.019 | 0.007% | 0.018% |
| stage sum | 260.66 | 100.6% | 96.94% |
| **end-to-end (`sum`)** | **259.11** | **100%** | — |

Decomposition is complete to within noise (stage sum 100.6% of `sum` — the small
overshoot is the per-stage benchmarks not paying the shared `[][]float32`
accumulation). `↳` rows use `BenchmarkEncodeSplit`'s absolute numbers (its
line-splitter yields 1,551 chunks vs W1's 1,905; its `whole` 208.08 ms is within
2.3% of W1's `embedEncode` 212.78 ms — embedding cost tracks total text, not
chunk boundaries).

### The tokenize / pool split — 50.6 / 49.4 (amd64: 62.9 / 37.1)

**This is the sharpest box disagreement, and it re-ranks the embed stage.**
Measured twice, agreeing to ~1 pp:

| method | Tokenizer.Encode | encodeIDs | split |
|---|---:|---:|---|
| `BenchmarkEncodeSplit` (direct, min of 6) | 103.38 ms | 100.86 ms | **50.6 / 49.4** |
| `pprof` cum on W1 | 32.77% | 30.67% | **51.7 / 48.3** |

On amd64 the split is 62.9/37.1; here the f64 **pooling gather** (`encodeIDs` —
per-token embedding-row gather, f64 MAC, `wsum`, final narrow + L2) is a co-equal
half of `Encode`, **30.7% cum / 29.4% flat** of the whole W1 run. It is the single
hottest leaf on this box. Two structural reasons this lands harder on arm64 than
on the AVX2 part: the gather+MAC is memory-and-scalar-f64 bound rather than
vectorised, and the tokenizer's byte-scanning half (which A2/A4 already trimmed)
is comparatively *cheaper* on the M1's wide front end. The consequence for
sequencing is in §3: **`encodeIDs` is now as large a target as the tokenizer**.
This promoted the memoization doc's §5 `StaticModel` presum (cache word→pooled
f64 sum, collapse the repeated-word subword gather), which it flagged as worth
more here than on amd64. **Built and measured 2026-07-31 — dead**: bit-exact
(0/386 real docs) but a serial wash (the per-word FNV-hash + RWMutex + map probe
costs as much as the ~0.4 gathers it collapses — the f64 gather is too cheap to
beat with a keyed lookup) and **+30.9% under EncodeBatch** (shared-cache reader-
atom contention across cores). See `task-perf-memoization.md` §5.

And a **vectorised f64 gather does not help either — measured, not assumed.** The
MAC loop (`sum[j] += f64(row[j])·ww`) runs at **0.7 ns/MAC on warm rows**, where a
4-wide unroll buys **+7%** and f32-accumulate **+5%** — it is *not* compute-
throughput bound, so NEON f64x2 (only 2-wide) would win single digits at best. On
the real corpus it is **1.42 ns/MAC**, 2× the warm rate: that gap is the cold
embedding-row gathers from the 63 MB table — **memory-bound**, which no MAC kernel
fixes. The pool is 49% of `Encode` because it is *memory*- and *latency*-bound, not
because the arithmetic is slow. There is no addressable compute lever in it, and
cutting its memory traffic is what §5 tried and lost. **Like the f32 kernel, the
pool is at its floor on this box.**

### Where the time goes inside the tokenizer (pprof cum, % of W1)

| | cum % |
|---|---:|
| `StaticModel.Encode` | ~63.4% |
| ↳ `Tokenizer.Encode` / `encodeSegment` | 32.77% |
| ↳↳ `normalize` (of which `cleanText` 6.30%) | 10.92% |
| ↳↳ `preTokenize` | 8.82% |
| ↳↳ `wordPiece` (**already memoized**) | 7.56% |
| ↳ `encodeIDs` | 30.67% (29.4% flat) |
| `bm25.Tokenize` | 6.30% |
| `bm25.Build` | 5.88% |
| `chunk` (regex) | ~3.7% |

`wordPiece` at 7.56% is the post-memoization residual (A3/`6b69133` is in the
tree), matching amd64's 7.59% — the one figure that transferred nearly exactly,
because it is a hash-map probe, not arithmetic.

---

## 2 · W2 — hybrid query, retrieval only (n=1905, k=50)

Measured on the **current** tree (post-Phase-A: A5's `fuse` presize and item 39's
WAND are both landed), so the amd64 Step-0 W2 table is not directly comparable —
its `fuse.RRF` 23.3% predates A5.

| stage | ns | % |
|---|---:|---:|
| `bm25.Tokenize(query)` | 235 | 0.36% |
| `embed.Encode(query)` | 4,446 | 6.79% |
| `bm25.TopK` | 21,455 | 32.77% |
| **`ann.FlatI8.Query`** | **32,831** | **50.15%** |
| `fuse.RRF` | 3,191 | 4.87% |
| stage sum | 62,158 | 94.94% |
| **end-to-end (`sum`)** | **65,470** | **100%** |

`FlatI8.Query` dominates the retrieval query at 50%, `bm25.TopK` at 33%; `fuse.RRF`
is 4.87% (post-A5). As on amd64, once a rerank is in the pipeline the whole
retrieval stack is a small fraction of a query — every row here is a
sub-millisecond concern.

---

## 3 · A1 — the concurrency curve, and what the 6P+2E asymmetry does

`StaticModel.EncodeBatch`, landed. The amd64 box got **8.21× at NumCPU** off 8
homogeneous cores + SMT. This box's answer is different, and the *shape* is the
finding. Sweep over the real corpus, min of 6:

| workers | ms | speedup | % of linear |
|---|---:|---:|---:|
| serial `Encode` loop | 212.52 | 1.00× | — |
| 1 | 212.47 | 1.00× | 100% |
| 2 | 106.57 | 1.99× | 99.7% |
| 3 | 73.10 | 2.91× | 96.9% |
| 4 | 56.86 | 3.74× | 93.4% |
| 5 | 47.76 | 4.45× | 89.0% |
| **6** (P-cores) | 42.37 | **5.02×** | 83.6% |
| 7 | 42.09 | 5.05× | 72.1% |
| **8** (NumCPU) | 40.63 | **5.23×** | 65.4% |

**Scaling is near-linear to 6 workers — the physical P-core count — then it bends
hard.** Worker 7 adds **0.7%** and worker 8 adds 3.5%: the two E-cores together
buy **1.04×** over the six P-cores, and per-linear efficiency falls off a cliff
from 84% to 65%. The mechanism is the straggler the `EncodeBatch` comment already
anticipates: a contiguous 1/8 share handed to an E-core finishes well after a
P-core's, and the fork/join barrier waits on the slowest shard.

**`concurrency = NumCPU` is still the fastest absolute setting (40.63 ms), so the
default is correct — but the last two workers are near-worthless and the curve's
knee is at the P-core count.** This belongs in `EncodeBatch`'s doc comment: on
Apple big.LITTLE, scaling is linear to the performance-core count and the
efficiency-cores contribute single digits; do not expect NumCPU-linear speedup,
and a caller optimising for throughput-per-watt should prefer `concurrency = 6`
(5.02× at 84% efficiency) over 8 (5.23× at 65%). Compared to amd64: 8 threads
here give 5.23× where 8 amd64 cores gave 6.26× (2 of ours are E-cores) and the
M1 tops out at 5.23× where amd64 reached 8.21× with SMT.

End to end on W1:

| | serial | batched | |
|---|---:|---:|---:|
| embed stage | 212.78 ms | 40.73 ms | **5.22×** |
| **whole index run** | **259.11 ms** | **89.43 ms** | **2.90×** |

### What A1 does to the ranking — and where the boxes diverge

Because A1 delivers **5.22×** here vs **8.60×** on amd64, the embed stage does not
collapse as far, and the post-A1 ranking is **different**:

| stage | ms | % of batched run | amd64 batched % |
|---|---:|---:|---:|
| **`embed.EncodeBatch`** | **40.73** | **45.5%** | 33.0% |
| `bm25.Build` | 21.95 | 24.5% | **27.6%** |
| `chunk.ChunkFile` | 9.63 | 10.8% | 18.7% |
| `bm25.Tokenize` | 13.64 | 15.3% | 16.5% |
| `ann.NewFlatI8` | 2.65 | 3.0% | 3.8% |
| `MarshalBinary` | 0.019 | 0.02% | 0.06% |

**On amd64, A1's 8.6× pushed `bm25.Build` to #1 (27.6%) and the embed stage to
33%. Here A1's E-core-limited 5.22× leaves the embed stage still #1 at 45.5%** —
so the tokenizer items (A2/A4) *and* the `encodeIDs` pool remain the top targets
on this box, where on amd64 they were promoted below `bm25.Build` and the chunker.
The arbiter re-derives its own ranking, and it does not match the amd64 one: the
single most valuable remaining index-time work on the M1 Pro is inside
`StaticModel.Encode`, split ~50/50 between the tokenizer and the pool.

---

## 4 · The three hand-back checks

- **A1 concurrency curve** (§3): the headline. Near-linear to 6, +1.04× from the
  two E-cores. `concurrency = NumCPU` stays optimal for latency; recorded for the
  `EncodeBatch` doc.
- **A2/A4's `utf8.ValidString` scan is negligible here — 0.16%**, vs amd64's 1%.
  Over the full corpus it is 0.168 ms against the 103.4 ms tokenize stage; arm64's
  SIMD `ValidString` is fast enough that the exact-on-invalid-UTF-8 fallback costs
  even less than on the AVX2 box. The design survives comfortably.
- **A5's sort mechanism holds on arm64 but the end-to-end sites can't be re-A/B'd
  (landed).** `BenchmarkSortSites`: `sort.Slice`→`slices.SortFunc` is 2104→832 ns
  at n=50 (**2.53×**, cf. amd64 2.95×), 324→72 at n=10 (4.5×), 1.34ms→848µs at
  n=10000 (1.58×) — the isolated win is real and comparable to amd64. The amd64
  finding that the eight non-`fuse` sites measured **zero** end-to-end (they sort
  partially-ordered heaps, which pdqsort handles differently than the random-data
  microbench) cannot be reproduced without reverting landed code, but the same
  structural argument applies here; nobody should quote "3.3–4.6× each" on either
  box.

---

## 5 · What transferred, and what did not

| quantity | amd64 (3700X) | M1 Pro | transferred? |
|---|---:|---:|---|
| tokenize / pool split | 62.9 / 37.1 | **50.6 / 49.4** | **no** — pool much heavier on arm64 |
| A1 speedup at NumCPU | 8.21× | **5.23×** | **no** — 6P+2E vs 8C+SMT |
| A1 knee | ~6 (then SMT) | **6 (P-cores)** | shape yes, ceiling no |
| embed % of serial W1 | 77.8% | 82.1% | close |
| embed rank after A1 | #2 (behind `bm25.Build`) | **#1** | **no** — A1 shrank it less here |
| `wordPiece` residual | 7.59% | 7.56% | **yes** (map probe, not arithmetic) |
| `utf8.ValidString` cost | 1% | 0.16% | cheaper here |
| A5 sort mechanism (n=50) | 2.95× | 2.53× | yes |
| §4.2 gate col-block | latency-neutral (2126→2123 ms) | **+3.5–6.9% (swigluMLP L=3584)** | **no** — 6P+2E fork/join tax on I/jb barriers |
| 22b Q8 widen fusion | (amd64: fix a only) | **−28%/−10%/−4.8% L=8/64/256** | n/a — arm64-only (packed path gated on `has2x8Kernel`) |

The scoreboard pattern from Phase A holds in reverse here: the quantities set by
*structure* (the `wordPiece` residual, the sort mechanism) transferred; the ones
set by *microarchitecture* (the f64 pool's weight, the parallel ceiling, the
post-A1 ranking) did not — and the ranking flip is the one that changes what to
work on next. **On the M1 Pro the next index-time win is inside
`StaticModel.Encode`, and the pool half is now as large as the tokenizer half.**

# aikit performance campaign — ranked opportunities (2026-07-28)

A prioritized menu, not a commitment. Every item cites `file:line`, states the
**mechanism** (why the hardware makes it slow), and declares whether the fix is
**bit-identical** or needs a golden re-pin.

**Constraints honoured throughout:** core stays pure-Go, no cgo. New hand-written
`.s` kernels are in bounds and several items depend on them.

---

## 0. How this was produced, and what to distrust

Four independent passes over the repo (linalg/quant · encoder/embed/vision ·
ann/bm25/sparse/topk/mmap · adversarial verification), followed by a refutation
round that re-read every load-bearing claim against the source. Four claims came
back overstated; they are corrected in place and listed in §7.

**Two caveats that matter for reading the numbers:**

1. **Measurements were taken on an x86 Xeon @2.1 GHz, 2 cores, AVX2 (no VNNI),
   L2 2 MiB, L3 260 MiB — not on your M1 Pro.** Ratios between two scalar Go
   loops transfer well. Absolute throughput, parallel scaling (capped at 2×
   here), and anything cache-capacity-dependent do not. The huge L3 in
   particular makes corpora up to ~250 MB cache-resident, which flatters every
   scan number.
2. **The repo does not build in the analysis sandbox** (`go.mod` pins Go 1.26.5;
   toolchain download blocked). Benchmarks were run against verbatim package
   copies in a `go 1.24` scratch module, or against standalone reproductions of
   the exact loop bodies. Nothing was measured through your real build.

So: treat every number as a **hypothesis with a mechanism attached**, and let
your own harness arbitrate — which is exactly the posture
`task-perf-linalg.md` already established when goinfer's end-to-end sweep
overturned the microbench-driven pool hypothesis.

**Ground truth first.** Items 1, 2, 5, 12, 22 and 26 are cheap and would each
either validate or kill a whole branch of this doc. Do those before the big ones.

---

## 1. The five headline findings

| # | Finding | Where | Why it's interesting |
|---|---|---|---|
| **A** | BM25 and SPLADE queries cost **O(corpus), not O(postings)** — measured **>99% waste** on a selective query | `bm25/query.go:45`, `sparse/sparse.go:99` | The dense score array's *allocation alone* costs more than the entire posting traversal. A 3-term query over 200k docs: 412 µs, of which <1% is scoring. |
| **B** | `dotI8AVX2` is **slower per MAC than the f32 blocked kernel** — so the int8 index buys a 4× memory cut and converts ~none of it into speed | `linalg/dot_amd64.s:311-340` | 7.9 MAC/cycle (int8) vs 14.9 (f32 `Dot8x4`), L1-resident. The arm64 SDOT path is correct; **this is an Apple-silicon-tuned project's amd64 blind spot.** |
| **C** | Every transcendental in the forward pass is a **scalar `math.*` f64 call**; softmax `exp` costs **3.6–9× the attention GEMMs it sits between**, independent of L and head count | `encoder/linalg.go:147`, `bert.go:382`, `vision/encoder.go:361,383` | SigLIP-so400m at np=4096 issues **7.25 billion `math.Exp` calls per image**. |
| ~~**D**~~ | ~~`annmetal`'s `gemm_w8a8` is the untuned Phase-1 correctness kernel~~ — **ALREADY FIXED before this doc was committed; see §7.5.** Both platforms now run a tiled GEMM + on-device top-k (`gpu/v0.15.0` Metal, `gpu/v0.16.0` CUDA). | `gpu/annmetal/backend.go`, `gpu/anncuda/backend.go` | The mechanism was real; only the "still unused" status was stale. Measured result now in `docs/BENCH-gpu-results.md`. |
| **E** | `EncodeBatch` pads to `Lmax` **inside index-ordered chunks**; ~92% of FLOPs run over padding-inflated M | `encoder/model.go:177-211`, `forward_batch.go:132` | 50 docs uniform on [20,512] ⇒ **48% of all linear-layer FLOPs computed on pad rows**. The fix (length bucketing) is provably bit-identical. |

---

## 2. Full ranked list

Legend — **Num:** `=` bit-identical · `~` ULP/reassociation, needs golden re-pin ·
`≠` changes numerics materially.

### Tier 0 — do first: cheap, and they unblock every measurement below

| # | Item | Area | Win | Num | Effort |
|---|---|---|---|---|---|
| 1 | Revive the 3 dead benchmark files; add warm-up + alloc accounting + a concurrent-QPS mode to `bench/harness.go` | bench | unblocks everything | — | S |
| ~~2~~ | ~~SPLADE: hoist `log1p` outside the L×V max-reduce~~ — **DONE** (§7.10); **measured 1.28–1.47×** on the pooling step (§7.12) | encoder | pooling is ~0.5% of `Expand`, so invisible end-to-end | **=** | — |
| ~~3~~ | ~~`topk.Push`: hoist the threshold compare~~ — **DONE** (§7.8) | topk, ann | **1.05× end-to-end** on `Flat.Query` (the 1.43× was the selection step alone) | = | — |
| 4 | `FlatI8.Query`: pool the score buffer; stop allocating a `Workspace` per query | ann | 10–25% now, large at N≥1M | = | S |
| ~~5~~ | ~~Index (de)serialization: bulk `copy`~~ — **DONE, 5.14×** (§7.8) | ann | 15.6 → 3.04 ms on a 50k×384 index; the 20–30× estimate was optimistic | = | — |
| ~~6~~ | ~~BERT/GTE forwards never call `enterForward()`~~ — **DONE** (§7.12) | encoder | removes oversubscription | — | — |
| ~~7~~ | ~~SPLADE vocab matmul runs **serial**~~ — **DONE, 1.05–1.19× on `Expand`** (§7.12); needed a trunk column fan-out too, since the trunk is NOT parallel at short L | encoder | 2.3× was optimistic | = | — |
| ~~8~~ | ~~`gte.go:230` allocates 12.6 MB/`Encode` outside the scratch arena~~ — **DONE, −12.7 MiB/call (−43%)** (§7.13) | encoder | exactly as estimated; **no latency change** | = | — |
| 9 | `SpanCache` LRU → MRU/scan-resistant; add `MADV_WILLNEED` on map; pipeline `Touch(b+1)` | mmap | 0% → max hit rate | — | S |
| 10 | `bm25`: hoist `1/avgdl`; precompute per-posting impact at build time | bm25 | 1.5–2× scoring | ~ | S–M |
| ~~40~~ | ~~**(NEW)** intra-op matmul fans across ROWS, replicating the weights per worker~~ — **DONE, −3.9% geomean / −15.8% on `SPLADE.Expand` L=91** (§7.14) | encoder | columns beat rows at every trunk shape, up to 3.33× | = | — |
| 41 | **(NEW)** `matmulBTInto`'s 4 MFLOP naive/blocked split is not reduction-order-consistent; `linalg` deleted its own for that reason | encoder | correctness-of-contract, not speed | — | S |
| ~~42~~ | ~~**(NEW)** attention softmax + GELU/GeGLU run SERIAL and are O(L²)/O(L·I)~~ — **DONE, −31% geomean, `GTE.Encode` L=690 3.28×** (§7.15) | encoder | the elementwise stages ARE the forward once the matmuls are parallel | = | — |

### Tier 1 — the big measured wins

| # | Item | Area | Win | Num | Effort |
|---|---|---|---|---|---|
| ~~11~~ | ~~**(A)** touched-set selection + pooled accumulator~~ — **DONE, bm25 + sparse** (§7.9) | bm25, sparse | 1.3–218× on bm25, selectivity-dependent | = | — |
| ~~12~~ | ~~**(B)** Rewrite `dotI8AVX2`~~ — **DONE, 2.10× kernel / 2.02× scan** (§7.6) | linalg | measured on Zen 2; int8 now at f32 MAC-parity | = (integer) | — |
| ~~13~~ | ~~**(C)** SIMD `expF32`/`erfF32`/`tanhF32` + `SoftmaxRowsInto`/`GELUInto`~~ — **DONE, all sites** (§7.16 text, §7.17 vision + cross-encoder); pure Go, no assembly | linalg→encoder, vision | text −20.1%; **cross-encoder −33.8%**; SigLIP −17.9%/−30.6% | ~ (contracts stated + gated) | — |
| 14 | **(E)** Length-bucketed `EncodeBatch` under a token budget | encoder | 1.3–2× on ragged batches | **=** | M |
| 15 | HNSW: batch neighbour scoring through `Dot8x4` | ann | **1.36–1.40× measured** end-to-end | ~ (1 ULP) | M |
| 16 | `Flat.Query` is single-threaded; shard it + per-shard selector | ann | 1.74–2.08× on 2 cores; 4–8× typical | = | M |
| 17 | HNSW build: batch `prune`/`selectHeuristic`; kill 225 allocs/insert | ann | 1.5–2.5× build | ~ | M |
| 18 | `vision/qwen_encoder.go` has no scratch arena — **~15 GB alloc+memset per image** | vision | removes ~1.5 s of memset | **=** | M |
| ~~19~~ | ~~`DequantizeRowInt4`: hardware integer divide per element~~ — **DONE, 4.93×** (§7.11) | linalg | above the 2–4× estimate | = | — |
| 20 | Int8 register blocking (1×4 / 4×1) — the arm64 kernel has none | linalg | 1.2–1.6× GEMV, more at prefill | = (integer) | M |
| ~~21~~ | ~~**(D)** `annmetal`: adopt the tiled/simdgroup kernel + on-GPU top-K~~ — **DONE** (§7.5) | gpu | measured: Metal 1.99×, CUDA 15.25× vs each box's own CPU @ N=100k/batch=256 | = (int32) | — |

### Tier 2 — structural

| # | Item | Area | Win | Num | Effort |
|---|---|---|---|---|---|
| ~~22~~ | ~~Q8 encoder path re-widens the whole int8 weight matrix to f32 **per matmul**~~ — **DONE via fix (a), −58.2% at short input** (§7.18); fix (b) (fuse into `packedFill`) still open | encoder | cost model exactly right; ~190 ms/forward measured | = | — |
| 23 | `packedFill` lost `blockedFill`'s m-blocking; a-panel re-read per 8-column group | linalg | plausibly large at prefill | = | M |
| 24 | `matmul_blocked.go` packed stride is a 4096 B power-of-two — recreates the aliasing packing exists to remove | linalg | free | = | S |
| 25 | arm64 `Dot2x8` has the wrong MR×NR: 4×4 needs 8 loads per 16 FMLAs vs today's 10 | linalg | 1.1–1.25× arm64 f32 GEMM | **=** | S–M |
| 26 | `math.Round` is **not** an amd64 intrinsic (`math.Trunc` is) — verified in disassembly | linalg, encoder | ~12 int ops → 1 `ROUNDSD` | = | S |
| 27 | Encoder's 4-MFLOP naive threshold sends **every** attention matmul at L<250 to a scalar triple loop | encoder | 3–9% for L<250 | ~ | S |
| 28 | `CrossEncoder` has **no batch API** and re-tokenizes the query per pair | encoder | unlocks item 14 for rerank | = | M |
| 29 | `bm25` posting is 16 B (`sparse`'s is 8 B); build allocates 3.1× the payload | bm25 | 1.5–2× scoring, 3× build alloc | = | M |
| 30 | `bm25.Tokenize` allocates a string per mixed-case token — 787 allocs for one 20 KB Go file | bm25 | 787 → ~10 | = | S |
| 31 | QKV split + per-head V transpose: 3 full `[L,D]` copies/layer, with a power-of-two stride pathology at exactly `DefaultMaxSeqLength=512` | encoder | 3.3× on the transpose (small absolute) | = | M |
| 32 | Vision preprocess: `draw.Draw` costs more than the JPEG decode; resize recomputes the x-map per row | vision | **2.3× measured** | = (tested) | S–M |
| 33 | `moeMLP` is entirely M=1 — 2048 single-row GEMM calls per MoE layer at L=512 | encoder | group tokens by expert | = | M |
| 34 | `forwardBatch` silently falls back for MoE / dense-GELU / qkv-bias models, **and double-counts `enterForward`** | encoder | correctness-ish perf bug | — | S |

### Tier 3 — the swings (bigger, riskier, or hardware-gated)

| # | Item | Area | Win | Num | Effort |
|---|---|---|---|---|---|
| 35 | AVX2/VNNI int8: the unsigned-offset correction is **free here** because weights are static | linalg | 1.3–1.8× on VNNI boxes | = | M |
| 36 | arm64 **i8mm (SMMLA)** — 2× int8, but **only for prefill/batch, and your M1 Pro can't run it** | linalg | ≤2× prefill on M2+/Graviton3+ | = | M–L |
| 37 | Outer-product f32 microkernel via by-element `FMLA` (raw `WORD` encoding verified) — 4.0 FMLA/load vs today's 1.6 | linalg | up to ~1.45× arm64 f32 | **≠** | L |
| 38 | Binary/Hamming prefilter + exact rerank (composes with `embed.Truncate`) | ann | ~10× first stage | ≠ (recall) | L |
| 39 | WAND / block-max WAND / MaxScore dynamic pruning | bm25, sparse | 2–10× **on top of** item 11 | = | L |
| 40 | Flash-attention / online-softmax tiling | encoder, vision | 1.05–1.15× alone; 67 MB → 1 MB scratch | ≠ | L |

---

## 3. The top items in detail

### 1 · The benchmarks that would have caught most of this are dead

Three benchmark files reference `../search/index.go`. **There is no `search/`
directory in aikit** — these are leftover `ken` paths, and `b.Skipf` makes them
pass green:

- `bm25/bm25_bench_test.go:19` → skips at `:22`
- `chunk/regex/chunker_bench_test.go:43`
- `chunk/treesitter/cast_bench_test.go:47`

So `bm25.Tokenize` (the hottest indexing function), `bm25.TopK` (finding A), and
the default regex chunker have **zero live benchmark coverage**. Separately, the
chunk benches' testdata paths are `../../../testdata/repo/…`, which from
`<root>/chunk/regex` resolves to the **parent of the repo root** — they'd skip
even in a full checkout. And the live `bm25` benchmark tops out at 1,000
documents, where an 8 KB score array is L1-resident and finding A is invisible
by construction.

`bench/harness.go:68-94` measures exactly one thing: sequential single-query
latency. No warm-up (so `p99` over a small query set is just the cold first
query), no allocation accounting (item 4's 40 MB/query at scale is invisible),
no concurrent-QPS mode (so `FlatI8`'s internal goroutine fan-out looks free when
under real load it oversubscribes). `MemMB` for `Flat` is computed as `n*d*4`
(`:62`), ignoring per-row slice headers, allocator rounding, and GC mark cost of
N separate objects.

**This is the highest-leverage small item in the doc** — it is what makes every
other number arguable instead of assertable.

---

### 2 · SPLADE applies `log1p` inside the max-reduce

`encoder/splade.go:102-112`:

```go
pooled := make([]float32, V)              // zeros
for i := range L {
    for v, x := range logits[i*V:(i+1)*V] {
        if x > 0 {
            if w := float32(math.Log1p(float64(x))); w > pooled[v] { pooled[v] = w }
        }
    }
}
```

`float32 ∘ Log1p ∘ relu` is monotone non-decreasing and maps 0→0, so
`max_i f(x_i) = f(max_i x_i)` **exactly** — including the f32 narrowing, the zero
initialization, and the negative-only-column case (stays 0, excluded by the
`w > 0` filter at `:115` either way). NaN is skipped by `x > 0` in both forms.

Take the max of raw logits first (plain float compares), then apply `log1p`
**once per vocab entry**: `V` calls instead of one per positive element of an
`[L,V]` matrix.

**Honest sizing.** A synthetic benchmark at 50% positive density measured
**17.8× (143.6 ms → 8.1 ms)**. Trained SPLADE logits are mostly negative by
construction, so at ~10% density the reduction is ~25× in call count and the
loop win is smaller in absolute terms. Still one of the best effort:payoff ratios
in the doc. Bit-identical; an equivalence test over 64×30522 normal-random
logits passes bit-for-bit.

**Two more in the same function.** `splade.go:98`'s vocab projection
(`L·768·30522` ≈ 12 G MACs at L=512, ~22–30% of `Expand`) routes through
`matmulBT` → `matmulBTBlocked` → `linalg.MatmulBTInto`, documented **"SERIALLY
(no goroutines)"** (`linalg/matmul_blocked.go:33`) — while the trunk goes through
`s.mm` → `wantParallelMatmul` → the parallel path. Amdahl on 8 cores: `43.5/8 + 12`
vs an ideal `55/8`, ~2.3× left on the table (item 7). And it allocates `L*V`
floats = **62.5 MB at L=512**; tiling the vocab into column blocks drops peak to
~1 MB, bit-identically, since `blockedFill` already takes an `[nStart,nEnd)` range.

---

### 6 · `enterForward` gap — the defect is backwards from how it reads

`grep enterForward` hits only `forward.go:38`, `forward_batch.go:43`,
`forward_q8.go:27,72`, `forward_tokens.go:19`. **`bert.go:230` and `gte.go:194`
have none** — despite `parallel.go:37-40` stating the contract: *"Every forward
variant (f32/q8, single/batch) must pair these."*

`wantParallelMatmul` (`parallel.go:72-80`) gates on `inflightForwards.Load() > 1`.
For the entire BERT family — `BERT.Embed`, `CrossEncoder.Score`, `SPLADE.Expand`,
`GTE.Encode` — that counter is permanently **0**, so the gate returns *true*
every time. The natural reranker loop (N goroutines each calling `Score`) becomes
**NumCPU × NumCPU** runnable goroutines contending at the join barrier — precisely
the oversubscription `parallel.go:12-16` was written to prevent. Symmetrically, a
`Model.Encode` running alongside a `CrossEncoder.Score` sees count=1 and
parallelizes, unaware of the BERT forward.

Two lines each in `bert.go` and `gte.go`. Also fix `forward_batch.go:58`'s
fallback, which calls `enterForward` twice (once in `forwardBatch:43`, again in
`forward:38`) — so `EncodeBatch(texts, …, 1)` on a MoE checkpoint is strictly
slower than a bare `Encode` per text (item 34).

---

### 11 · (A) BM25/SPLADE queries are O(corpus)

`bm25/query.go:45` allocates `make([]float64, ix.N())` per query, zeroes it
(large-object path ⇒ a fresh `memclr` of the whole span), writes into a handful
of slots, then **re-reads all N slots** for selection (`:90`, `:108`). Both the
zeroing and the selection sweep scale with corpus size and are independent of
query selectivity. `sparse/sparse.go:99,144,159` is identical.

**Measured** — 200k docs × 120 tokens, 30k vocab, 3-term query touching **2,335**
postings:

```
Scores            267.6 µs   1,605,632 B/op
AllocScoreArray   323.4 µs   1,605,632 B/op   ← make([]float64, 200k) alone
SelectOnly        144.3 µs                    ← the full-corpus range + heap
TopK              412.3 µs
```

`Scores` ≈ the allocation alone: **the 2,335 posting updates that are the actual
query are lost in measurement noise.** ~65% allocate+zero, ~35% selection sweep,
**<1% scoring**. SPLADE the same (30-term query, 22,185 postings: 654 µs).

**Fix.** Keep the dense accumulator (random-access `+=` beats a map), but (i) take
it from a `sync.Pool` and clear only touched slots, (ii) record touched doc ids in
a `[]int32` during accumulation and select over the touched set. `Scores()` is
public and must keep returning a dense slice — put this in an internal path that
`TopK`/`Query` use, keeping `Scores` as the compatibility wrapper. Visiting the
touched set in ascending doc order preserves `topk`'s first-seen-wins tie-break,
so output is identical.

**412 µs → roughly 5–15 µs**, and the win grows linearly with corpus size.

Second-order in the same functions: `bm25/query.go:46` and `sparse/sparse.go:112-113`
build a per-query `seen`/dedupe map (plus an `order` slice) for what is typically
tens of terms — sort/scan a small slice instead.

---

### 12 · (B) `dotI8AVX2` is port-bound at 8 MAC/cycle

`linalg/dot_amd64.s:311-340` handles **16 int8 per iteration** with 128-bit
`VMOVDQU` loads on a 256-bit ISA, feeds **two** `VPMOVSXBW ymm←xmm` widenings into
one `VPMADDWD`, accumulates into a **single** register, and is top-tested with an
unconditional `JMP` back (two branches/iteration).

`VPMOVSXBW` is a shuffle-domain op on one port on mainstream Intel; two per 16
MACs is a hard ceiling of **8 MAC/cycle** regardless of caching.

**Measured, L1-resident (so memory is provably not the limit), d=768:**

```
DotI8    46.32 ns/op   16.58 GMAC/s  =  7.9 MAC/cycle   ← at the VPMOVSXBW ceiling
DotF32   37.02 ns/op   20.75 GMAC/s  =  9.9 MAC/cycle
Dot8x4  196.7  ns/op   31.23 GMAC/s  = 14.9 MAC/cycle
```

**The int8 kernel is slower per MAC than the f32 register-blocked kernel.**
Scan-level, through `MatmulBTW8A8` at K=768, serial:

```
N=4096   (3 MB,  L3)   65.5 ns/row   11.7 GB/s
N=100k   (77 MB, L3)   67.2 ns/row   11.4 GB/s   ← flat in N ⇒ compute-bound
N=400k   (307 MB, RAM) 120.5 ns/row   6.4 GB/s
```

`Flat` (4 B/element) streams at ~25 GB/s; `FlatI8` (1 B/element) is stuck at
~11 GB/s. Per row at d=768: f32 ~123 ns, int8 ~115 ns. **So on amd64 the int8
index buys a 4× memory cut and converts almost none of it into speed.**
`FlatI8`'s apparent edge in the end-to-end bench (6.3 ms vs 21.0 ms at N=100k) is
mostly item 16 — it's parallel and `Flat` isn't — not the codes.

**Fix.** 32 B/iteration with full `VMOVDQU ymm`; `VPMADDUBSW`-based products
(cleanest: store codes as `uint8 = code+128` so `VPMADDUBSW(row_u8, query_i8)` is
directly usable and the `128·Σq` correction is one scalar per query); 4
independent accumulators; bottom-tested loop. Integer arithmetic ⇒ any
reassociation is bit-exact, and the existing differential tests against
`dotI8Scalar` gate it.

**The arm64 side is already correct** (`dot_i8dp_arm64.s`: 4 accumulators,
64-wide `SDOT` loop). This gap is exactly what you'd predict from a project tuned
on Apple silicon — and it is **not** blocked on the roadmap's "Zen 4+ access"
trigger. Only the VNNI variant (item 35) needs special hardware.

---

### 13 · (C) Scalar `math.*` transcendentals dominate softmax and the FFN activation

Sites: `encoder/linalg.go:147` (`math.Exp` in `softmaxRow`), `:168` (`silu`),
`bert.go:382` (`math.Erf`), `gte.go:291`, `crossencoder.go:155` (`math.Tanh`),
`vision/encoder.go:361,383`, `vision/qwen_encoder.go:627,636`.

**Measured, per element (N=65536):**

| kernel | ns/elem | vs. a plain elementwise f32 pass |
|---|---:|---:|
| GELU-erf (`math.Erf`) | 28.9 | 51× |
| GELU-tanh (`math.Tanh`) | 38.8 | 68× |
| SiLU (`math.Exp`) | 36.1 | 63× |
| `math.Exp` (softmax body) | 34.3 | 60× |
| scalar f32 minimax `expf` (reference) | 4.7 | 8× |
| `x[i] = v*1.0000001` (baseline) | 0.57 | 1× |

**Mechanism.** Go's `math.Exp`/`Erf`/`Tanh` are branchy pure-Go scalar f64
routines (`Ldexp`, range reduction, rational-segment selection). They cannot
inline into a SIMD loop and the compiler does not auto-vectorize.

**The closed form is the interesting part:**

> softmax ÷ attention-matmul = `t_exp / (2·headDim·t_mac)` — **independent of L
> and of head count**. At headDim=64 that's **3.6–9×**. The softmax costs several
> times what QKᵀ + scores·V cost.
>
> FFN activation ÷ FFN matmul = `t_act / (k·D·t_mac)` — **grows as D shrinks**:
> 11% at D=768, **22% at D=384 (the MiniLM cross-encoder)**.

| workload | transcendental calls/forward | derived share |
|---|---:|---|
| CodeRankEmbed, L=512 | 37.7 M `exp` + 18.9 M `silu` | ~20–38% |
| MiniLM-L6 cross-encoder, L≈200 | 1.8 M `erf` + 1.4 M `exp` | ~15% GELU + ~14% softmax |
| SigLIP-so400m, np=4096, 27 layers | **7.25 G `exp`** + 476 M `tanh` | ~40–70% |

**Fix.** A SIMD `expF32`/`erfF32`/`tanhF32` in `linalg` (AVX2/NEON `.s`) plus
`SoftmaxRowsInto`/`SiLUInto`/`GELUInto` wrappers so all eight call sites share
one implementation. 10–30× on those loops.

**Numerics — a bit-identical variant exists but is expensive.** Go's `math.Exp`
is a fixed sequence of IEEE-754 f64 ops; a vector transcription emitting the same
operations in the same order, with branches replaced by blends and FMA
contraction matching the compiler's (Go contracts on arm64, not amd64), is
bit-identical for all finite inputs. `Erf` is harder (multi-segment). A faster
minimax polynomial is **not** bit-identical and needs a cosine-parity re-pin.

**Before writing any assembly:** you already have the pprof that found scores·V
at ⅓ of `Encode` (`cpu-acceleration.md:153-160`). `math.Exp` under `softmaxRow`
should be the next line down that same profile. Confirm there first — it costs
nothing and it either validates or kills this entire branch.

---

### 14 · (E) `EncodeBatch` pads inside index-ordered chunks

`selfAttentionBatched` correctly skips pad positions *inside attention*
(`attention_batch.go:68-70`, `realLen`-bounded). But the projections and MLP do
not: `s.mm(h, Wqkv, qkv, BL, D, 3*D)` (`:31`), `s.mm(ctx, OutProj, out, BL, D, D)`
(`:108`), `swigluMLP(h, …, B*Lmax, s)` (`forward_batch.go:132`) all run over
`M = B*Lmax`. Attention is only `L/(8D)` of a SwiGLU layer — **8% at L=512,
D=768** — so ~92% of FLOPs are exposed, and the wasted fraction is exactly
`Lmax/mean(L) − 1`.

`model.go:178` partitions by **input order** (`chunkSize := (len(texts)+concurrency-1)/concurrency`),
so one long document inflates `Lmax` for its whole chunk *and* imbalances workers.
The justifying comment (`forward_batch.go:14-18`, "rerank batches are typically
uniform") holds for fixed-size chunk rerank — not for `EncodeBatch(texts…)` over
a real corpus, which is the exported general API.

**50 documents uniform on [20,512]: `Lmax`=512, mean=266 ⇒ 48% of all
linear-layer FLOPs computed on padding.** Length-sorting into buckets of ~8 drops
that below 5%. It also shrinks the scratch footprint, currently
`concurrency × B × Lmax × (3D + 2I + ~6D)` floats ≈ **190 MB per worker** at
B=7/Lmax=512/D=768/I=3072, held live forever (`scratch.go:103-108` never shrinks).

**Bit-identical, provably.** Each output of `blockedFill` reduces over k-tiles
determined solely by `kBlock` and `K` (`linalg/matmul_blocked.go:68-70`); `M` only
selects the m-block boundary, and `linalg` states M-invariance as a contract and
gates it (`TestMatmulBT_MConsistent`). Pad rows are never read. So changing which
sequences share a batch changes no real output bit.

*(Corollary: `forward_batch.go:35`'s comment — "the matmul tile sizes change with
M, so the f32 accumulation differs by ULPs" — is now wrong. The only remaining
source of batch-vs-single divergence is item 27's naive threshold.)*

**Fix.** In `encodeBatch` (`model.go:159`): tokenize all inputs, sort indices by
`len(ids)`, bucket under a **token budget** (`B*Lmax ≤ budget`) not a fixed count,
dispatch, scatter back by original index.

---

### 15 · HNSW scores one neighbour at a time

`ann/hnsw.go:386` calls `h.sim(qv, n)` per neighbour inside the traversal loop;
`sim` (`:234-239`) is a single `linalg.Dot`. Meanwhile `ann/flat.go:172` already
uses `linalg.Dot8x4(&q[0], &v0[0], …, &v7[0], n4, &sums)` — **flat got the 8-row
kernel; hnsw did not**, even though `task-ann-simd-dots.md` listed `hnsw.go`'s
`sim()` as the second site.

`Dot8x4` holds the query strip in registers and amortizes it across 8 candidate
rows instead of re-streaming it per neighbour.

**Measured.** Kernel level, 32 gathered rows: d=256 **2.05×**, d=768 **1.82×**.
End-to-end prototype (n=50k, d=256, k=10):

```
baseline           ef64  474.7 µs / 19 allocs   ef200 1137.8 µs / 22 allocs
pooled heaps only  ef64  519.4 µs /  1 alloc    ef200 1170.3 µs /  1 alloc
pooled + batched   ef64  338.0 µs /  1 alloc    ef200  838.1 µs /  1 alloc
                         └ 1.40×                       └ 1.36×
```

The transformation is order-preserving: collect *unseen* neighbours into a buffer
(marking visited as you go, unchanged), score in groups of 8, then run the
existing push/threshold logic **in the original order** — so the evolving
`results.items[0].sim` threshold sees the identical sequence.

**Correctness:** differential test vs the pristine implementation over
d∈{64,256,768} × n∈{500,5000} × ef∈{16,64,200} × 25 queries — **4,500 ranked
hits, 0 index differences, max |Δscore| 5.96e-08** (one f32 ULP). Same
reassociation tradeoff `Flat` already documents, and HNSW is approximate by
contract.

Same treatment applies to `greedyClosest` (`:337-341`) and `selectHeuristic`
(`:508`) — the latter is item 17.

---

### 18 · Qwen ViT allocates every working buffer per layer

`vision/encoder.go:258-278` documents the SigLIP fix — *"the old code re-make'd
n1/att/o/n2/mid/mlp plus attention's q/k/v/qh/kh/vt/scores/oh every layer — at
SigLIP-so400m ≈290 MB allocated and discarded PER LAYER, ~7.9 GB per image"*
(audit #5). **`vision/qwen_encoder.go` has the identical structure and never got
the treatment**: allocations at `:335, 337, 343, 404, 408-410, 427, 434-438, 472,
475, 482, 593`, all inside the per-layer loop at `:329-348`.

`make([]float32, n)` **zeroes**, so this isn't just GC pressure — it's a mandatory
memset of every byte plus first-touch page faults.

At Qwen2.5-VL ViT dims (hidden=1280, inter=3420, heads=16, depth=32,
nPatches≈5184): ~540 MB/layer × 32 = **≈15 GB allocated and zeroed per image**,
≈1.5 s of pure memset at ~10 GB/s before GC does anything.

Fix: a `qwenScratch` mirroring `encScratch`, sized once in `forwardViT`
(`maxSeg` is already computed at `:428-433`), one scratch per `Forward`.
Bit-identical — same ops, distinct buffers.

---

### 19 · `DequantizeRowInt4` executes a hardware integer divide per element

`linalg/quant.go:384-397`:

```go
for k := range cols {
    b := packed[k/2]
    ...
    dst[k] = float32(int(nib)-8) * scales[k/group]   // group is a runtime int
}
```

`group` is a function parameter, so the compiler cannot strength-reduce
`k/group`. Compiled for arm64, the loop body contains a per-element `SDIV` plus a
divide-by-zero guard.

`SDIV` is the only non-pipelined integer unit on Firestorm/Zen (~2–8 cycle
throughput, 7–12 cycle latency; `IDIVQ` on x86 is worse), against a body of ~6
cheap ops — so a nominally ~1 cycle/element loop runs at ~4–8.

`q4Span` (`:505-512`) calls this once per weight row, so `MatmulBTQ4` pays it N×K
times. Your CHANGELOG already records `MatmulBTQ4` as "~72% of decode in the f32
dequant" — a large share of that is this one instruction. It's also
`WeightMat.Row`'s int4 path (`weightmat.go:151-163`), i.e. the tied-embedding
lookup on every generated token.

**Fix:** group-outer / element-inner, hoisting `s := scales[g]`; process two
elements per packed byte to kill the `k&1` branch and the double load. Same
products, same order ⇒ **bit-identical**.

> **Correction to an earlier draft of this finding:** `packed[k/2]` is *not* a
> division — a constant power-of-two divisor is strength-reduced to a shift by Go's
> SSA rules. Only `k/group` costs an SDIV.

---

### 21 · (D) The `annmetal` GPU kernel is the untuned Phase-1 one — and the tuned kernel already exists in the repo

Your finding, and the code fully supports it. `gpu/annmetal/backend.go:29-64`
compiles its **own private MSL source string** containing only correctness-first
kernels, exactly as its header comment admits (`:20-21`): *"The kernels here are
intentionally minimal and correctness-only; the tuned decode kernel is Phase
1b/2."*

Four separate things are wrong with `gemm_w8a8` as written (`:47-64`):

1. **Uncoalesced global loads — the dominant cost.** With `g = m*N + j`, adjacent
   threads have adjacent `j` and read `codes + j*K`. Adjacent lanes therefore
   touch addresses **K bytes apart**. Every lane pulls a full cache line and uses
   one byte of it. At K=768 that is ~1/64 of achieved bandwidth. This alone
   explains ~8 GOP/s.
2. **Scalar `char` loads.** `acc += int(qrow[k]) * int(crow[k])` loads one byte
   per iteration; `char4`/`packed_char4` vector loads are the standard fix.
3. **Integer div/mod per thread.** `uint m = g / N, j = g % N` — `N` is a runtime
   `constant`, so no strength reduction. GPUs are notoriously bad at integer
   division. A 2-D dispatch removes it entirely.
4. **The full M×N score matrix is copied back to host every call**
   (`:209 copy(dst, outBuf.Floats())`), plus four device buffers allocated and
   released per call (`:194-202`). At N=100k, M=256 that's 25.6 M floats ≈
   **102 MB per call** — to answer a query that needs M×K = 2,560 floats.

**The fix is largely already written, in the same module.** `gpu/metal_vit.go:377`
has `gemm_w8a8_tiled` — threadgroup-staged, `TILE=16`, documented and tested as
**bit-identical** to the untiled kernel (int32 accumulate, so re-chunking K cannot
change the sum) — and `gemm_f32_sg` demonstrates the `simdgroup_matrix`
cooperative-matrix path on the same device layer. `annmetal` uses neither. The
CUDA mirror (`gpu/anncuda/backend.go:64`) has the same shape and the same
available upgrade (`KernelGEMMW8A8Tiled`, `cuda_vit.go:105`).

**Recommended sequence:**

1. Point `annmetal` at the existing tiled kernel instead of its private source
   string — the single highest-value line-count-to-speedup ratio in this doc, and
   bit-identical by the int32 argument the tiled kernel already documents.
2. **Fuse top-K into the kernel** so the device returns `M×K` scores, not `M×N`.
   This is the bigger structural win at ANN shapes: a per-threadgroup partial
   top-K plus a small host merge cuts the transfer by ~4 orders of magnitude at
   N=100k, K=10. (`gemv` has the same issue — it copies all N.)
3. Hoist the per-call buffer allocation into an M-sized arena; the "batch queries
   are the infrequent path" justification at `:172-174` stops being true the
   moment the crossover sweep is the thing you're measuring.
4. Only then re-run the crossover sweep. **The current "GPU loses ~5× on Apple"
   result is a statement about an untuned kernel, not about the device** — and
   your methodology is what surfaced that. Worth stating explicitly in
   `docs/BENCH-gpu.md` so the number isn't later cited as "Metal doesn't win for
   ANN."

---

### 22 · The Q8 encoder path re-widens the entire weight matrix per matmul

`encoder/linalg_q8.go:52-60`:

```go
w := deqW[:N*K]
for n := range N {
    sc := bScales[n]
    for k := range row { row[k] = float32(bq[k]) * sc }   // scalar, N*K per call
}
matmulBTInto(a, w, dst, M, K, N)
```

One full `[N,K]` f32 materialization **per matmul call**. Per Nomic layer:
Wqkv 1.77 M + OutProj 0.59 M + fc11/fc12/fc2 7.08 M = **9.44 M elements**, ×12
layers = **113 M per forward, independent of L**. Measured widen throughput
1.0 ns/elem ⇒ **~113 ms/forward**, plus 453 MB of stores + 453 MB of reads. The
largest buffer (fc11, 9.4 MB) far exceeds L2, so it round-trips through DRAM:
written once, read once, discarded. Redone per batch and per worker — 8 workers ×
12 layers × 5 matmuls = **480 widens of the same immutable weights per rerank
query**. Amortization is `1/M`: ~10% at L=80, ~1.5% at L=512, **~50% at MoE M=1
shapes**.

Two fixes, both bit-identical (`float32(int8) * scale` is a single rounding
either way):

- **(a)** SIMD widen in `linalg` (`VPMOVSXBD`+`VCVTDQ2PS`+`VMULPS`; NEON
  `SXTL`+`SCVTF`+`FMUL`) — 6–8×, effort S.
- **(b)** Fuse the widen into `blockedFill`'s b-panel pack so only an
  8×kBlock tile (≤24 KB, L1-resident) is ever materialized. `packedFill`
  (`matmul_blocked.go:211-259`) is the exact hook — same order, same bits. Kills
  the 0.9 GB round-trip *and* removes `scratch.deqW` (up to 9.4 MB pinned per
  pooled scratch). Effort M.

---

### 26 · `math.Round` is not an amd64 intrinsic — verified

Disassembled on this box (Go 1.24.7, amd64):

```
main.g → math.Trunc        ROUNDSD $0x3, X0, X1        ← one instruction
main.h → math.RoundToEven  ROUNDSD $0x0, X0, X1        ← one instruction
main.f → math.Round        SHRQ/ANDL/CMPQ/ORQ/…        ← inlined floor.go body, ~12+ int ops
```

`Round`'s half-away-from-zero semantics aren't a `ROUNDSD` mode, so Go's
intrinsic table registers it for **arm64 only** (→ `FRINTA`). Call sites:
`linalg/quant.go:58,157,364`, `linalg/kquant.go:123`, `encoder/quant.go:70` — all
on the per-call quantization path of `MatmulBTW8A8Into`/`MatmulBTW4A8Into`, i.e.
O(M·K) per matmul. At prefill (M=512, K=2048) that's ~1 M elements of scalar work,
~5% of the matmul on amd64 and ~0 on arm64.

**Bit-identical amd64 rewrite:** for this domain `|x| ≤ 127` and `x` is an
exactly-representable f32 widened to f64, so `x + 0.5` is exact and
`math.Trunc(x + math.Copysign(0.5, x)) == math.Round(x)`.

Better still, vectorize the quantizers outright — NEON does it bit-identically
(`FMUL`, `FRINTA`, `FCVTZS`, `SQXTN`×2; `FRINTA` *is* round-half-away-from-zero).

---

## 4. `mmap` — three separate problems (item 9)

**(a) No residency hints on the fast path.** `mmap/mmap_unix.go:28-51` maps and
returns; `mapAndAliasFlatI8` (`ann/flat_i8_mmap.go:150-172`) never calls `Advise`
(grep for `Advise` in `ann/` returns nothing). So the first query on a freshly
loaded index demand-faults every 4 KB with default readahead and no hint that the
pattern is a linear scan.

**(b) The LRU is the pathological policy for the pattern it serves.**
`mmap/spancache.go:83,94` is textbook LRU; `scorePaged` (`ann/flat_i8_mmap.go:116-118`)
touches blocks strictly ascending 0..B−1 **every query**. When the corpus exceeds
budget, LRU evicts block 0 first — precisely the block the next query reads first.
**Hit rate for a repeated cyclic scan under LRU is exactly 0%.** MRU or any
scan-resistant policy keeps `budget/blockBytes` blocks permanently resident. The
existing gate test only asserts `evictions > 0`, so it passes either way — add a
hit-rate assertion.

**(c) `Touch` then immediately fault.** `Touch(b)` issues `MADV_WILLNEED` — an
*asynchronous* readahead hint — and the code then immediately runs the matmul over
those very pages, stalling on the faults it just requested. Touch `b+1` before
scoring `b` to pipeline I/O behind compute.

Also hot-path: `Touch` does three map lookups plus `container/list` pointer
surgery plus an interface assertion per block — ~7,300 calls/query at 1 MiB blocks
on a 10M×768 index. Block ids are dense integers; a slice-indexed ring removes all
of it.

**Speculative:** the 67.2 → 120.5 ns/row degradation between a 77 MB and 307 MB
corpus at constant per-row work is partly cache (L3 is 260 MB here) but 4 KB pages
at ~5 rows/page also means a dTLB entry per 5 rows. `MADV_HUGEPAGE` is a two-line
experiment with a clean falsification. (Note: on Linux this only helps
heap-backed/prequant weights — THP on file-backed mappings needs large-folio
support — and macOS has no THP equivalent for file mappings.)

---

## 5. Additional findings surfaced during verification

These came out of the refutation pass and aren't in the tables above:

- **`FlatI8`'s fast path allocates a `Workspace` per query; the slow path
  doesn't.** `flat_i8.go:127` calls the allocating `linalg.MatmulBTW8A8` wrapper
  (`quant.go:174-177` does `var ws Workspace`), so `int8Buf(K)` + `f32Buf(1)` are
  fresh every query. `flat_i8_mmap.go:119-122` **already fixed exactly this** for
  the paged path, with a comment naming the problem — the common in-memory path
  was left behind. (Folded into item 4.)
- **`packedFill`'s packed stride can itself be a conflict stride.**
  `packKBlockFor` (`matmul_blocked.go:194-199`) returns 1024 whenever
  `K%768 != 0`, giving a **4096-byte power-of-two gap** between the 8 packed
  b-rows — the classic aliasing stride packing exists to avoid. It escapes only
  on Apple P-cores (128 KB L1D ⇒ 256 sets). One `+4` float pad makes it
  conflict-free by construction. (Item 24.)
- ~~**`gte.go:209`** builds `newRopeTable(L, headDim, RopeTheta)` per `Encode`~~ —
  **DONE** with item 8 (§7.13); `GTE` now holds a `ropeCache` that grows to the
  longest sequence seen and returns bit-identical views for shorter ones. The
  five Nomic/`forward*.go` sites still rebuild per call and were left alone —
  their shapes (L≤512, headDim 64) are where `newRopeTable`'s "cheap" comment is
  actually true, and changing them needs its own measurement.
- **`bm25/query.go:58-59`** divides by the loop-invariant `ix.avgdl` per posting;
  hoisting `invAvgdl` halves the divisions from two to one before you even get to
  precomputed impacts.
- **`topk.Push`'s `s.seq++` can move below the reject test** without changing
  semantics — `seq` only needs monotonicity among *retained* items.
- **`fuse/rsf.go:41-42,64-65,84`** has the identical two-parallel-maps shape as
  `fuse.go`; item 22's earlier framing named only `fuse.go`.
- **Doc drift:** `docs/internal/cpu-acceleration.md:163` says `forward_q8.go`
  still has the old scalar scores·V loop. **It doesn't** — `forward_q8.go:170,180`
  and `:238,250` both route through `s.mm` with the folded `vHT` transpose. That
  follow-up is done; the doc should say so.

---

## 6. Measured dead ends — recorded so nobody re-chases them

- **Contiguous vs scattered vector storage does not speed up the scan.** Expected
  `Flat`'s `[][]float32` (`flat.go:56`) to lose to a flat backing array. It
  doesn't: scattered 6.057 ms / 25,357 MB/s vs contiguous 6.173 ms / 24,884 MB/s —
  identical within noise, scattered marginally ahead. Go's size-class allocator
  already places same-size rows near-contiguously. The remaining arguments for a
  flat layout are **GC mark cost** (N separately-marked objects — real at 1M+
  vectors) and footprint (24 B headers + size-class rounding), **not** bandwidth.
  Don't sell it as a speed change.
- **Pooling HNSW's search heaps, alone, is not a latency win.** Allocations
  19–22/query → 1/query, garbage 14–42 KB → ~0.2–1 KB, and the query got
  *slightly slower* (519 vs 475 µs at ef=64). Worth doing for GC pressure under
  high QPS, and free once you're batching (item 15 needs the buffers anyway) —
  but don't book it as a speedup.
- **`layerNorm` has no bit-identical vectorization.** `encoder/layernorm.go:18-39`
  measures 2.98 ns/elem; a fused-moment f32 variant runs at 1.42 ns (2.1×) — but a
  SIMD reduction needs multiple partial accumulators, which changes the f64
  summation order by construction, and `layernorm.go:8-10` calls this *"the single
  most parity-sensitive op outside the GEMMs."* Total cost is ~0.5% of a forward.
  **Leave it alone.**
- **Q6_K native K-quant** — already proven below the gate in
  `task-q8k-integer-accum.md` by the byte-ratio ceiling argument. Nothing here
  changes that.
- **The spin-park worker pool** — built, measured end-to-end by goinfer, pulled.
  Item 11 in the linalg pass is a *different* thing (dynamic work-stealing over
  column chunks to fix E-core straggler imbalance, not warm workers to fix wake
  latency), but the bar for re-entering this area should stay high.
- **PQ / IVF are not performance items for this codebase today.** IVF trades
  recall for a fraction of the scan — which is what HNSW already does here; PQ's
  win is memory, and int8 already took the easy 4×. Binary+rerank (item 38) is the
  one worth measuring, and only after item 12 tells you how much gap is left.
- **AVX-512 for f32:** your deprioritization still holds, but for a *different*
  reason than the doc gives. The modern blocker isn't downclocking (Zen 4/5 have
  none) — it's that Intel client parts ship AVX-512 fused off, so the f32 win only
  reaches servers and AMD. **The int8 VNNI subset is the part worth taking, and it
  does not need 512-bit vectors** (AVX-VNNI is VEX-encoded, no AVX-512 required).

---

## 7. Claims that came back overstated — corrected

Four things an earlier draft asserted that the refutation pass knocked down:

1. **`DequantizeRowInt4`'s `packed[k/2]` is not a division** — constant
   power-of-two divisors are strength-reduced to shifts. Only `k/group` costs an
   SDIV. (The rest of item 19 stands exactly.)
2. **`fuse`'s comparator does not do a map lookup per comparison** — the score
   inequality short-circuits first (`fuse.go:101-106`), so the lookup runs only on
   exact ties. Still worth cleaning up; the win is small.
3. **SPLADE's `log1p` saving is density-dependent.** The 17.8× was measured at 50%
   positive density; trained SPLADE logits are mostly negative, so the real
   reduction is smaller. The mechanism and bit-identity are unaffected.
4. **`FlatI8`'s parallelism is threshold-gated.** `M*N*K >= 1<<24` means an
   M=1/dim=768 index below ~21.8k vectors runs fully serial, same as `Flat`. So
   item 16's framing ("FlatI8 is parallel, Flat isn't") holds only above that size.

5. **Finding D / item 21 was already fixed before this doc was committed.** The
   analysis ran against an older tree. `gpu/annmetal/backend.go` now carries
   `gemm_w8a8_tiled` **and** `topk_rows` (`gpu/v0.15.0`), and the CUDA mirror landed
   at 16:43 (`gpu/v0.16.0`) — 19 minutes before this doc's commit at 17:02.
   Measured, vs each box's own CPU at N=100k / batch=256: **CUDA 15.25×, Metal
   1.99×**. The mechanism the finding identified was real; only its "still unused"
   status was stale.

6. **Finding B is CONFIRMED on real amd64 hardware — and understated.** The doc's
   numbers came from an x86 Xeon in a scratch module, never through the real build.
   Re-measured on a Ryzen 7 3700X (Zen 2, AVX2, no VNNI) through the real toolchain
   via the newly added `BenchmarkDotI8VsF32_K768`, K=768, L1-resident:

   ```
   int8  DotI8    41.4 ns/op   18.6 GMAC/s
   f32   Dot8x4  142.7 ns/op   43.1 GMAC/s     <- 2.28x faster PER MAC
   ```

   The doc claimed 1.89× (7.9 vs 14.9 MAC/cycle); the real gap on Zen 2 is larger.
   Finding B stands, and item 12 is the best-evidenced entry in this document. Note
   there was **no int8 dot benchmark at all** to run — it had to be written first,
   which is item 1's argument in miniature.

7. **Item 12 is DONE, and it landed inside the predicted band.** `dotI8AVX2` rewritten
   to 64 int8/iteration with four independent accumulators and a bottom-tested loop,
   on a Ryzen 7 3700X:

   ```
   kernel (BenchmarkDotI8VsF32_K768)   18.6 -> 39.0 GMAC/s    2.10x
   scan   (BenchmarkMatmulBTW8A8)     232.3 -> 115.2 us       2.02x
   ```

   The microbenchmark proposed and the scan-level measurement disposed — they agree,
   so this is a real win rather than a harness artifact.

   **One correction to the item's diagnosis.** It attributed the ceiling to
   `VPMOVSXBW` port pressure and prescribed "32 B/iteration". Byte width is not what
   mattered: `VPMOVSXBW` takes an **m128 source operand**, so the old kernel's separate
   `VMOVDQU` loads were pure overhead — 6 uops per 16 MACs became 4. The win is a uop
   reduction plus a broken accumulator dependency chain, not wider loads. Had the
   ceiling really been `VPMOVSXBW` throughput, widening could not have helped at all,
   because the widen:MAC ratio is invariant.

   **`VPMADDUBSW` was considered and rejected** (the item's "cleanest" suggestion):
   u8xi8 pair sums can exceed int16 and the instruction SATURATES, so it needs
   range-limited codes. That belongs with the VNNI work (item 35), not here.

   **Finding B's headline consequence is now retired**: int8 was 2.28x slower per MAC
   than f32; it is now 1.10x — MAC-parity. The int8 index finally converts its 4x
   memory cut into comparable throughput on amd64. arm64 was already correct and is
   untouched.

8. **Items 5 and 3 are DONE — and both under-delivered against their estimates.**
   Recorded here because a missed estimate is as useful as a hit.

   **Item 5 (serialization): 5.14×, not 20–30×.** `MarshalBinary` appended the code
   block one byte at a time and pushed every scale through a `put32` closure that
   captured `b` by reference — a non-inlinable indirect call per scale. Replaced with
   one `memmove` (via the same `int8`/`byte` aliasing `LoadFlatI8Mmap` already uses)
   plus inlined `AppendUint32`. On a 50k×384 index: **15.6 ms → 3.04 ms**, 1243 →
   6386 MB/s. Removing the redundant zeroing pass (`make([]byte, total)` →
   `make([]byte, 16, total)`) changed nothing measurable, which locates the remaining
   cost in the 19 MB allocation's page faults rather than in the copy. Getting past
   5× needs an API that writes into a caller-supplied buffer, not a faster loop.

   **Item 3 (topk threshold hoist): 1.05× end-to-end, not 1.43×.** The mechanism is
   real — `Push` can't inline (siftUp/siftDown), so every rejected candidate paid a
   call to fail one comparison — and the new `topk.Selector.Threshold()` is inlinable,
   so the scan now only calls `Push` for candidates that can actually be retained.
   But `Flat.Query` is dominated by the SIMD dot product, not by selection:
   N=50k goes 3868 → 3666 µs. The doc's **1.43× was the selection step measured
   alone**, which is a small fraction of query time. Worth keeping (free and
   bit-identical), worth not expecting 1.43× from.

   Both are bit-identical: the hoisted guard uses `>`, matching `Push`'s own strict
   comparison, so a tied newcomer is rejected either way.

9. **Item 11 is DONE for `bm25` (the `sparse` mirror remains), and it OVERSHOT its
   estimate — but only at one end.** The estimate of "10–50×" is really a range that
   depends entirely on query selectivity, which the mechanism makes inevitable: the fix
   replaces O(corpus) with O(postings touched), so the win *is* the selectivity ratio.
   200k docs × 120 tokens, 30k vocab, k=10:

   ```
   selective (tail-of-vocab, few postings)   262,412 -> 1,203 ns    218x
   common    (14,243 of 200,000 docs)        599,886 -> 461,166 ns  1.30x
   allocations, both cases               1,606,520 -> 888 B/op    ~1800x
   ```

   The doc's own scenario (2,335 postings of 200k) sits near the selective end, so its
   "412 µs → 5–15 µs" was about right for that query. But quoting a single ratio for
   this item is misleading in either direction — it should always be stated with the
   selectivity it was measured at.

   **Two implementation notes.** The touched set uses GENERATION STAMPS rather than a
   "score != 0 means touched" test. Today every BM25 contribution is strictly positive
   (the Lucene idf is > 0 for any known term), so the zero test would work — and would
   break silently the day a scorer admits a zero or negative contribution, dropping the
   doc from results entirely. And `accum.sortTouched` is load-bearing, not cosmetic:
   `topk` keeps the FIRST-SEEN member of a tie, so reproducing the dense scan's exact
   output requires visiting the touched set in ascending doc order.
   `TestTopK_touchedSetMatchesDenseScan` gates that against a reference implementation
   and asserts the fixture actually contains ties, so the tie-break path is not
   silently untested.

   Also folded in the item's second-order note: the per-query `seen` map is now a
   linear scan over the preceding terms (queries are tens of terms, where the map's
   allocation and hashing cost more than the scan).

10. **Item 11's `sparse` half and item 2 are DONE — and finishing item 3 exposed a bug
    I had introduced.**

    **`sparse`** now mirrors bm25's pooled touched-set accumulator. One difference is
    recorded in the code: bm25 could have used a "score != 0 means touched" test
    (its contributions are strictly positive), but sparse genuinely cannot — a sparse
    dot can produce a zero contribution from a zero-weight posting or from
    cancellation, so there the zero test would be wrong *today*, not merely fragile.
    Generation stamps are used in both for one shape. The existing first-appearance
    dedupe order (audit #16, which exists so identical queries don't produce
    0.6 vs 0.6000000000000001) is preserved: the `seen` map became a linear scan over
    the accumulated terms, which preserves that order by construction.

    **Item 2** is in, and its exactness claim is now gated directly rather than
    inferred: `TestSpladePooling_hoistedLog1pIsExact` asserts bit-identity over a
    64×3000 fixture and *checks the fixture contains both regimes* — my first version
    had 3000 positive columns and **zero** all-negative ones, so the `f(0)=0` boundary
    the hoist depends on was untested until the vacuity check caught it.

    **A bug found by finishing the job.** Item 3's `topk.Threshold()` indexed
    `heap[0]` when `k == 0`, where the heap is permanently empty. `ann` never hit it
    (its callers guard `k <= 0` earlier), and bm25 has no k=0 test — sparse's
    `TestQuery_kSemantics` panicked and found it. Fixed (k==0 → +Inf, "nothing can be
    retained"), plus `TestThreshold_matchesPush`, which cross-checks the hoisted guard
    against `Push`'s own decision over a random stream at k ∈ {0,1,3,10}. Worth noting
    the shape: an optimisation that is correct in the package it was measured in, and
    wrong in the next package to adopt it.

11. **Item 19 is DONE — 4.93×, above its estimate — and getting there produced the
    sharpest methodology lesson of the campaign.**

    ```
    DequantizeRowInt4, 4096 cols, group 32   15,765 -> 3,198 ns   4.93x
    ```

    **The first three measurements said the opposite.** A benchmark written as
    `const cols, group = 4096, 32` reports the ORIGINAL at 957 ns and the optimised
    form at 3,210 ns — i.e. the fix looks like a 3.35× REGRESSION. It isn't: with a
    constant divisor the compiler folds `group`, strength-reduces `k/group` to a shift,
    and the benchmark measures a loop that never executes the divide at all. Changing
    the constants to `:=` locals does **not** fix it — SSA constant-propagates those
    too. Only obtaining them through a `//go:noinline` call defeats the folding, and
    then the true figures appear.

    So the doc's own item-1 thesis has now bitten three times, and this is the worst
    case: not a benchmark that *misses* an effect, but one that **inverts** it. A
    benchmark for an optimisation whose whole premise is "this divisor is not a
    compile-time constant" must ensure the compiler agrees. The note is in the
    function's doc comment so the next person does not re-derive it.

    The rewrite is group-outer with the nibble pair unrolled. The case that needed
    gating is an ODD `group`, where a packed byte spans two groups and the pair loop
    would otherwise apply one group's scale to the next group's element — the
    equivalence test covers 10 group sizes × 13 column counts against the original,
    and caught exactly that bug on the first attempt.

12. **Items 2, 6 and 7 are DONE — and item 7's mechanism was right but incomplete,
   in a way only an end-to-end measurement could show.** All three now measured on
   a real `naver/splade-cocondenser-ensembledistil` checkpoint (fetch recipe in
   `scripts/README.md`; the repo publishes only `pytorch_model.bin`, so it needs a
   safetensors conversion). `SPLADE.Expand`, 3700X/16-thread, n=8, benchstat:

    | L (wordpieces) | before | after | |
    |---|--:|--:|--:|
    | 22 | 154.1 ms | 146.2 ms | −5.1% (p=0.000) |
    | 91 | 771.6 ms | 680.8 ms | −11.8% (p=0.000) |
    | 357 | 2.424 s | 2.036 s | −16.0% (p=0.000) |

    **Item 2 (hoisted `log1p`): 1.47× / 1.38× / 1.28×** on the pooling step at
    L=22/91/357 — real, and the density caveat in §7.3 above is confirmed: nothing
    like the synthetic 17.8×. Pooling is ~0.4–0.5% of `Expand`, so this is
    invisible end-to-end. It is now also validated the strong way: the HF parity
    golden passes at cosine 1.000000, not just the synthetic bit-identity test.

    **Item 7 (serial vocab projection): the projection itself got 5.1×/6.3×/6.3×
    faster — and short-query `Expand` got 6.7% SLOWER (p=0.005).** The doc's
    estimate of "up to ~2.3× on `Expand`" assumed the premise on line 188, *"the
    trunk goes through `s.mm` → `wantParallelMatmul` → the parallel path."* That
    premise is false at short L. `wantParallelMatmul` requires `M ≥ 64`, and M is
    the token count: at L=22 **every** projection in the trunk fails it, so the
    whole forward ran on one core of sixteen. Parallelizing only the vocabulary
    projection therefore dropped a single all-core burst into an otherwise serial
    forward, and that costs the serial part real time. Measured directly with a
    memory-free all-core spin burst as a control:

    ```
    trunk alone                              97.8 ms
    trunk after a memory-free all-core burst 110.3 ms   (+12.8%, boost clock)
    trunk after the real parallel projection 129.2 ms   (+32%,   boost + cache)
    ```

    So ~13 points of the interference is the package dropping out of single-core
    boost and ~19 is cache/memory thrash. The fix is not to keep the projection
    serial — it is for the trunk to be parallel too. `wantParallelCols` now picks
    up exactly the shapes the row split structurally cannot serve (`M <
    2*minRowsPerWorker`) and fans them across output columns instead;
    `linalg.MatmulBT` already provided that, bit-identically, so no new linalg API
    was needed. Alternating A/B in one process, L=22 trunk: **176.0 ms serial →
    134.8 ms (1.31×)**. With both changes the short-query regression inverts to a
    5.1% win.

    Two lessons, both instances of patterns already in this doc. First, magnitude
    again: mechanism right, size optimistic (2.3× estimated, 1.19× measured at the
    most favourable L). Second, and new: **a per-kernel benchmark cannot see
    interference between kernels.** The projection benchmark showed a clean 5–6×
    at every L and was completely honest about the kernel; it simply could not
    observe that making one stage parallel taxes every serial stage around it. Only
    the end-to-end number showed the regression, and only a phase-attributed
    measurement (a temporary `BenchmarkSpladePhases` splitting trunk/proj/tail)
    located it in the trunk — a stage whose code had not changed at all.

    **Item 6 (`enterForward` gap)** landed as specified for `bert.go` and `gte.go`
    and is gated by `TestBERTFamily_bracketsInFlight`, which observes the counter
    from inside a forward and fails (peak=0) without the brackets. It was a
    prerequisite, not a bonus: extending intra-op parallelism to a family whose
    in-flight counter is permanently 0 would have deepened exactly the
    oversubscription the gate exists to prevent. The `forward_batch.go:58` double
    count is left alone — this doc numbers it item 34.

    **One unrelated finding, since chased down.** In the same A/B at L=91, where
    the row split *does* engage, the parallel trunk measured **636.8 ms vs 592.4 ms
    serial** — the row-split path is a net loss at BERT-base trunk shapes on this
    box. That became §7.14, which found the cause (a row split replicates the
    weights across workers; a column split partitions them) and retuned the axis.

13. **Item 8 is DONE — the estimate was exact, and the item was honest about what
    it was buying.** Measured on a real `Snowflake/snowflake-arctic-embed-m-v2.0`
    checkpoint (fetch recipe in `scripts/README.md`), `GTE.Encode`, benchstat n=4:

    | L (wordpieces) | B/op before | after | | sec/op |
    |---|--:|--:|--:|--:|
    | 22 | 987.6 KiB | 568.5 KiB | −42.4% | ~ (p=1.000) |
    | 175 | 6.023 MiB | 3.984 MiB | −33.9% | ~ (p=0.057) |
    | 690 | 29.45 MiB | 16.75 MiB | −43.1% | **+1.38%** (p=0.029) |

    The table's Win column said "12.6 MB/call" and the measurement is 12.7 MiB at
    L=690 — the closest an estimate in this doc has come. Two allocations moved:
    the fused up/gate buffer (`L*2*intermediate`, 17 MB at L=690) into a new
    `scratch.upGate` field sized by `ensureFusedMLP`, kept out of `ensureLayer` so
    BERT/Nomic scratches do not carry it; and the per-`Encode` `newRopeTable` into
    a `ropeCache` on the `GTE` (the companion finding at §"more findings").

    **Latency did not improve, and at the longest shape it got slightly worse.**
    That is not a failure of the item — the item never claimed latency — but it is
    worth writing down that removing 12.7 MiB of allocation from a 4.2 s call
    buys nothing at single-call latency. The value is GC pressure under
    concurrency, which this benchmark does not exercise. `allocs/op` is unchanged
    (12.67k at L=690): these were 2 allocations, just enormous ones.

    The RoPE cache rests on a bit-identity claim — row m is cos/sin of
    m·invFreq[d] with no dependence on the table's own length, so a view over a
    longer table's first L rows equals a table built at L. That is gated
    (`TestRopeCache_viewIsBitIdentical` over 3 headDims × 3 bases × 8 lengths,
    plus a grow/shrink/switch sequence against fresh tables) and the gate was
    mutation-checked: a `view` that forgets to narrow fails both tests. Pooling
    `upGate` also introduces a genuine hazard — the buffer now arrives holding a
    previous forward's values — so `TestScratchUpGate_fullyOverwritten` poisons
    every pooled scratch and re-encodes; a mutant whose fused matmul leaves tail
    rows unwritten fails it.

    **Unrelated, and larger than this item:** `GTE.Encode` at L=690 takes **4.2
    seconds** on a 16-thread 3700X — roughly 37 GFLOP/s against ~156 GFLOP of
    work, where the row-split path is supposed to be engaged. That is the same
    smell as §7.12's closing note, now on a second model. Both are §7.14, which
    confirmed the axis was wrong and fixed it. (L=690 itself sits above the new
    crossover and keeps the row split; the win landed at the shorter lengths.)

14. **The row-split path was the wrong parallel axis on amd64 — a NEW item, found
    by §7.12 and §7.13 independently, now measured and fixed.** `BenchmarkParallelAxis`
    sweeps serial/rows/cols over the real trunk shapes with a 12-deep weight bank
    (so no variant streams the same `b` out of L3 repeatedly). On a 16-thread
    3700X, columns win at **every** shape:

    | shape (M,K,N) | serial | rows | cols | cols/rows |
    |---|--:|--:|--:|--:|
    | fc11 L22 · 22,768,3072 | 2.32 ms | 2.29 ms | 0.69 ms | **3.33×** |
    | qkv L91 · 91,768,2304 | 7.18 ms | 2.87 ms | 1.50 ms | **1.91×** |
    | fc11 L91 · 91,768,3072 | 9.80 ms | 3.81 ms | 1.90 ms | **2.01×** |
    | fc2 L91 · 91,3072,768 | 13.1 ms | 4.89 ms | 2.56 ms | **1.91×** |
    | fc11 L357 · 357,768,3072 | 33.1 ms | 7.12 ms | 5.62 ms | 1.27× |
    | upgate L175 · 175,768,6144 | 36.0 ms | 7.53 ms | 5.46 ms | 1.38× |
    | upgate L690 · 690,768,6144 | 140 ms | 20.8 ms | 20.1 ms | 1.03× |

    Two causes, both structural rather than mis-tuned constants:

    1. **A row split replicates `b`; a column split partitions it.** Every row
       worker reads the whole weight matrix (`parallel.go`'s own comment says so:
       *"reads a[iStart\*K:iEnd\*K] + the shared read-only b"*), so a 9.4 MB fc11
       weight is streamed `workers` times. In a transformer linear `a` is [L,K]
       activations and `b` is [N,K] weights — the row split multiplies the
       dominant memory traffic by the worker count. The gap closing as M grows is
       this effect fading as arithmetic intensity rises.
    2. **The row split is worker-starved at exactly the sizes that matter.**
       `matmulBTBlockedIntoParallel` caps workers at `(M+31)/32`, so M=91 gets
       **three** workers on a 16-thread box and M=22 gets one (i.e. serial).

    Neither is arm64-specific, but the *cost* of (1) is: replicating `b` across an
    M1 Pro's large unified cache is far cheaper than across a desktop part's
    split-CCX L3, which is why the original tuning table never saw it.

    **The fix is a crossover, not a global flip — and the micro-benchmark alone
    would have got it wrong.** Making columns unconditional won SPLADE L=91 by
    17.4% but LOST GTE L=690 by 2.32% (p=0.008), even though the micro predicted a
    3% column win at that shape. Near a crossover the per-kernel number stops
    being predictive. The boundary is therefore taken from the end-to-end
    measurement and expressed as the condition that actually matters — *is the
    row split able to fill the machine?* — i.e. columns below
    `minRowsPerWorker·NumCPU`, rows at or above. Derived from `NumCPU` rather than
    pinned, so it stays honest on other core counts.

    Final, benchstat n=5, versus the row-first dispatch:

    | | before | after | |
    |---|--:|--:|--:|
    | `SPLADE.Expand` L=91 | 688.0 ms | 579.2 ms | **−15.8%** (p=0.008) |
    | `SPLADE.Expand` L=357 | 2.052 s | 1.957 s | −4.6% (p=0.008) |
    | `GTE.Encode` L=175 | 1.265 s | 1.250 s | −1.2% (p=0.008) |
    | `GTE.Encode` L=690 | 4.203 s | 4.189 s | ~ (p=0.841) |
    | geomean | | | **−3.9%** |

    All paths stay bit-identical, gated by
    `TestMatmulBTInto_dispatchIsNumericallyInert` across the crossover and under a
    raised in-flight counter. That matters more than usual here: the dispatch now
    depends on machine load, so a non-inert path would mean a golden that passes
    alone and fails under a concurrent batch.

    **Found while writing that gate, not acted on:** `matmulBTInto` keeps a
    4 MFLOP naive/blocked split, and the two are NOT bit-identical to each other
    (different reduction order). `linalg` deleted its own such threshold for
    exactly this reason — see `MatmulBT`'s M-INVARIANT note, which several
    consumers depend on. The encoder's copy survives. Every shape is
    self-consistent so nothing is wrong today, but attention QKᵀ is `L·headDim·L`
    and crosses 4 MFLOP around L=256, meaning the same model uses different
    reduction orders for short and long inputs. Worth its own item.

15. **Once the matmuls were parallel, the ELEMENTWISE stages were the forward —
    another NEW item.** §7.14 ended with `GTE.Encode` at L=690 still taking 4.2 s
    (~37 GFLOP/s of ~156 GFLOP). Profiling it, rather than assuming the matmuls
    were at fault, gave the answer immediately:

    ```
    softmaxRow   2.11 s/call  (50% of a 4.19 s call)  ENTIRELY SERIAL
      of which math.archExp        1.63 s
    linears      ~1.4 s            (already parallel)
    gelu         ~0.50 s           (27% after the softmax fix)  SERIAL
    ```

    The reason this hides is a difference in growth rate. The linear projections
    are **O(L)** work and the attention score matrix is **O(L²)**, and every
    element of that matrix goes through `math.Exp`. At L=22 softmax is 3% of a
    forward and invisible; at L=690 it is half the call. Parallelizing the
    matmuls does not touch it, so the more the matmul work is fixed the more
    completely the elementwise stages own the profile.

    Three splits, all across independent rows and therefore bit-identical (a row
    softmax uses only its own elements; GeGLU and GELU are elementwise):
    `softmaxRows` over the score matrix (6 call sites), `gelu`'s elementwise pass
    in chunks, and GTE's fused GeGLU row loop. Factored into one `parallelRows`
    helper that runs inline below threshold or when another forward is in flight
    — the same in-flight contract as the matmul gates.

    Measured against §7.14's state, benchstat n=5+6:

    | | before | after | |
    |---|--:|--:|--:|
    | `GTE.Encode` L=690 | 4.189 s | **1.279 s** | **−69.5%** (3.28×) |
    | `GTE.Encode` L=175 | 1.250 s | 956.7 ms | −23.5% |
    | `GTE.Encode` L=22 | 99.8 ms | 86.6 ms | −13.2% |
    | `SPLADE.Expand` L=357 | 1.957 s | 1.435 s | −26.7% |
    | `SPLADE.Expand` L=91 | 579.2 ms | 443.0 ms | −23.5% |
    | `SPLADE.Expand` L=22 | 148.0 ms | 140.1 ms | −5.4% |
    | geomean | | | **−31.0%** |

    Average parallelism across `GTE.Encode` went from 4.2× to 10.2×.

    **A measurement caveat that cost a wrong conclusion, and is worth adopting.**
    The `gelu` split appeared to REGRESS GTE by 4.5% (p=0.008, n=5) — on a code
    path that provably does not exist in GTE, which has no `gelu()` call at all.
    Re-running showed the whole GTE benchmark had drifted ~4.5% between
    invocations as the box heated over a long session. **A provably-unchanged
    code path moving 4.5% bounds the noise floor for cross-invocation benchstat
    on this machine at ~5%**, which retroactively means §7.14's smaller GTE
    numbers (−1.2%, and the L=690 "~") should be read as "within noise", not as
    signal. Everything reported above is far outside it. The durable fix is
    §7.12's lesson again in a new form: compare variants inside ONE process where
    possible, and treat a sub-5% cross-run delta as unmeasured.

    **Now load-dependent, so gated end to end.** A forward now picks among
    parallel and serial implementations for its matmuls, its softmax AND its
    activation, every choice made by reading the in-flight counter.
    `TestGTEEncode_loadIndependent` compares a fully-parallel forward against a
    fully-serial one bit for bit, because the failure this guards is not a crash
    but an encoder whose output depends on how busy the machine is — passing in
    CI and drifting in production under a concurrent batch.

    **What is left at L=690:** `dotFMA8` 62% and `math.archExp` 21%, both now
    parallel. The remaining transcendental cost is item 13 (SIMD
    `expF32`/`erfF32`), which is orthogonal to this and multiplies with it.

16. **Item 13 is DONE for the text encoders — and it is the first item in this
    campaign to land INSIDE its predicted band.** The doc estimated 1.25–1.5× on
    text; measured geomean is **1.25×** (−20.06%). benchstat n=6, against §7.15:

    | | before | after | |
    |---|--:|--:|--:|
    | `SPLADE.Expand` L=357 | 1.435 s | 1.016 s | **−29.2%** |
    | `GTE.Encode` L=690 | 1.279 s | 911.5 ms | **−28.7%** |
    | `SPLADE.Expand` L=22 | 140.1 ms | 104.2 ms | −25.6% |
    | `GTE.Encode` L=22 | 86.6 ms | 71.7 ms | −17.3% |
    | `SPLADE.Expand` L=91 | 443.0 ms | 390.1 ms | −11.9% |
    | `GTE.Encode` L=175 | 956.7 ms | 912.2 ms | −4.7% |
    | geomean | | | **−20.1%** |

    **Its own measurement table did not transfer, and that changed the plan.**
    Re-measured on the 3700X: GELU-erf **29.4 ns/element**, matching the doc's
    28.9 — but `math.Exp` is **15 ns**, less than half the 34.3 quoted. So the
    two halves of this item have very different ratios on amd64, and the "10–30×
    on those loops" figure is unreachable without real SIMD. What was built
    instead is PURE GO, portable to arm64 unchanged, and still delivered the
    predicted end-to-end band:

    | kernel | before | after | |
    |---|--:|--:|--:|
    | full softmax row (n=691) | 15.5–17.9 ns/elem | 8.4–8.6 | ~1.9× |
    | GELU | 29.2–30.7 ns/elem | 15.2–17.5 | ~1.9× |

    `linalg` gains `ExpF32`/`ErfF32`/`GELUF32`/`SiLUF32` plus `SoftmaxRowInto`/
    `GELUInto`/`SiLUInto`; `encoder`'s `softmaxRow`, `geluScalar` and `silu` are
    now three-line delegations.

    **This is the one deliberately NON-bit-identical change in the campaign,** so
    each kernel carries a stated, measured contract instead:

    | kernel | contract | measured |
    |---|---|---|
    | `ExpF32` | ≤1 ULP relative | 0.68 ULP max, 0.19 mean (3.8M pts) |
    | `ErfF32` | ≤2.5e-07 absolute | 1.92e-07 |
    | `GELUF32` | ≤1e-06 absolute | 4.67e-07 |
    | `SiLUF32` | ≤4 ULP relative | 1.21 ULP |

    **The contracts are derived, not tuned to pass.** GELU's is absolute because
    that is what propagates: it feeds an FFN matmul over I≈3072 with |W|~0.02, so
    a per-element error ε contributes ~1.1ε to a hidden state, against goldens
    already at maxΔ≈7.9e-06. GELU cannot be given a relative contract on x<0 at
    all — there it evaluates erfc via (1+erf), which cancels as erf→−1, so no
    absolutely-accurate erf yields relative accuracy; SiLU can, because
    x/(1+e^−x) divides rather than subtracts. Getting this wrong sent me round
    two false iterations of "loosen the threshold until it passes" before the
    metric, not the kernel, turned out to be the problem.

    **Two real bugs the gates caught, worth recording:**

    - **A&S 7.1.26 is unusable near zero.** The textbook erf form computes
      1 − P(t)·e^(−x²); as x→0 both terms tend to 1 and it loses nearly all
      significance — measured 5.5e-07 absolute at x≈0.024, i.e. 2e-05 relative,
      170 ULP, exactly where erf is at its most linear. Its quoted 1.5e-07 bound
      is on the mathematical truncation, not on the floating-point evaluation.
      Fixed with a Maclaurin branch for |x|<1, which shares the factor x and so
      cannot cancel.
    - **Rounding the series coefficients cost 4× the entire error budget.**
      Writing 1/42 as `2.3810e-2` looks harmless; at |x|=1 every power of x is 1,
      so a coefficient's own error lands undiminished in the result — that one
      constant contributed 5.4e-07 against a 1.5e-07 target.

    Also, at the top of `ExpF32`'s range k reaches 128 while the true result is
    still finite, so building 2^128 in one step encodes +Inf and poisons an
    answer that should be ≈2.4e38. Scaled in two steps instead.

    **The goldens moved CLOSER to HF, not further.** GTE worst hidden maxΔ went
    8.11e-06 → 5.96e-06 and cosine stayed 1.000000; SPLADE stayed at cosine
    1.000000 / term-jaccard 1.0. That is the expected direction on reflection:
    PyTorch computes these activations in float32 too, so Go's float64
    `math.Erf` was the outlier, not the f32 kernel replacing it.

    **Vision and the cross-encoder followed (§7.17).** The "no vision checkpoint
    on this box" that deferred them was wrong — see §7.17.

    SIMD assembly for these kernels remains available and would multiply with
    what landed.

17. **Item 13 finished: vision, the cross-encoder, and a blocker that was not
    real.** The remaining sites are done, and the model that gained most is the
    one the doc predicted would.

    | workload | before | after | |
    |---|--:|--:|--:|
    | `CrossEncoder.Score` L≈200 (D=384) | 560.3 ms | 370.9 ms | **−33.8%** (p=0.002) |
    | SigLIP tower, 576 patches, h768 | 3.048 s | 2.115 s | **−30.6%** (p=0.008) |
    | SigLIP tower, 196 patches, h512 | 478.3 ms | 392.9 ms | −17.9% (p=0.008) |

    **The cross-encoder prediction was right, quantitatively.** §13's closed form
    said the FFN activation share GROWS as D shrinks — 11% at D=768, ~22% at
    D=384 — and put MiniLM-L6 at ~22% GELU + ~14% softmax. Measured: **33.8%**.
    That is the doc's most precise correct call.

    **The vision prediction was not.** "Up to 2× vision" against a measured
    1.24× geomean (1.44× at the larger tower). But the *shape* is right and
    visible: the win grows with patch count (17.9% at 196 → 30.6% at 576)
    exactly as an O(patches²) softmax share should, and both towers here are far
    below the so400m/np=4096/27-layer case the 40–70% estimate was for. The
    trend supports the estimate; this hardware simply cannot reach the size
    where it would be demonstrated.

    `linalg` gains `TanhF32` (Cephes `tanhf`, ≤2 ULP measured 1.10) and
    `GELUTanhF32` (≤1e-06 absolute, measured 4.33e-07). `TanhF32` needs the same
    two-branch treatment `ErfF32` did and for the same reason: the closed form
    (e^2x−1)/(e^2x+1) cancels as x→0, where tanh is most linear. A test asserts
    the two GELUs stay *different functions* — SigLIP is trained on
    `gelu_pytorch_tanh`, so a future "simplification" that aliases one to the
    other would silently change every vision output, and now fails instead.

    All four vision sites plus `crossencoder.go:155` now route through `linalg`.
    The cross-encoder tanh is still measurement noise on its own (384 elements
    per `Score`); it moved for consistency, so there is one tanh in the kit.

    **The blocker was not real, and that is the lesson.** §7.16 deferred this
    work saying "no vision checkpoint on this box to re-verify parity against".
    Both vision fixtures were already present, and neither is a download —
    `scripts/pin_siglip_vision.py` *generates* a tiny random `SiglipVisionModel`
    locally, and `scripts/gen_siglip_bench.py` generates the real-sized towers.
    The parity gates had been passing all along. Cost: one deferred item.
    Recorded as a measurement-discipline entry in
    [`measuring-performance.md`](measuring-performance.md) §1.13 — *verify a
    blocker before reporting it*.

    Also fetched `cross-encoder/ms-marco-MiniLM-L-6-v2`, so
    `TestCrossEncoder_parity` now runs here rather than skipping: worst forward
    Δ 4.29e-06 against the Python reference, scores matching to 4 decimal places.
    That gate did not exist on this box when the tanh was changed, which is
    exactly when a gate matters most.

18. **Item 22 is DONE via fix (a) — and its cost model was exactly right.** The
    doc predicted the widen is O(N·K) and INDEPENDENT of L, so it amortizes as
    1/M. Measured on a 3700X with `nomic-ai/CodeRankEmbed`:

    | shape | forward | widen (flat, per forward) | widen share |
    |---|--:|--:|--:|
    | ~10 tokens | 258 ms | **183 ms** | 35% flat / 42% cum |
    | ~350 tokens | 1540 ms | **195 ms** | 2.2% |

    Same absolute cost at both lengths — the prediction holds quantitatively.
    (The doc estimated ~113 ms/forward from a 1.0 ns/elem widen; this box
    measures ~190 ms, i.e. ~1.7 ns/elem in situ.)

    Fix (a), the SIMD widen, landed as `linalg.DequantizeRowsInt8Into` with an
    AVX2 kernel (`VPMOVSXBD` → `VCVTDQ2PS` → `VMULPS`, 32 elements per
    iteration) and a portable fallback. **5.7× on the kernel**, and end-to-end,
    benchstat n=6:

    | input | before | after | |
    |---|--:|--:|--:|
    | ~10 tokens | 258.7 ms | **108.1 ms** | **−58.2%** (2.39×) |
    | ~85 tokens | 666.1 ms | 506.4 ms | −24.0% |
    | ~350 tokens | 1.537 s | 1.387 s | −9.7% |
    | geomean | | | **−34.1%** |

    Bit-identical, and not merely by argument: `float32(int8)` is exact and the
    scale is one f32 multiply either way, so the gate asserts EXACT equality
    against the scalar loop over lengths straddling both the 32- and 8-wide
    loops, plus the sign-extension extremes (−128, −1, 0, 127) and a
    write-past-the-end guard. `TestModelQ8_cosineMatchesF32` returns the same
    cosines to six decimals as before the change.

    **A §1.3 moment worth recording.** The kernel benchmark measures the scalar
    widen at 0.50 ns/elem, but in situ it costs ~1.7. The benchmark rotates one
    weight matrix that stays L3-resident; a real forward streams twelve
    different cold ones. The kernel ratio (5.7×) therefore *understates* the
    end-to-end effect at short L, which is the opposite of the usual direction
    and the reason the arbiter is the only number quoted above.

    **Fix (b) is still available and still worth doing.** Fusing the widen into
    `packedFill`'s b-panel pack would eliminate the remaining ~33 ms, the 0.9 GB
    DRAM round-trip, and `scratch.deqW` (up to 9.4 MB pinned per pooled
    scratch). What landed is the S-effort half of the item.

    **arm64 is untouched:** `dequantRowInt8` falls back to scalar there. The NEON
    version (`SXTL`+`SCVTF`+`FMUL`) is straightforward but unmeasurable on this
    box, and §7.16 is the standing reminder that per-kernel ratios do not
    transfer between architectures.

---

## 8. Suggested sequencing

**Phase 0 — make it measurable (a few days).**
Item 1 (revive benchmarks + harness: warm-up, allocs, QPS mode). Then re-run your
existing `Model.Encode` pprof and check whether `math.Exp` under `softmaxRow` is
where item 13 predicts. Two cheap gates that decide the shape of everything after.

**Phase 1 — the free wins (a week).**
Items 2, 3, 4, 5, 6, 7, 8, 19, 26. All S-effort, all bit-identical or
scheduling-only, no golden re-pins. Item 5 alone is ~20–30× on index load.

**Phase 2 — the measured structural wins (2–4 weeks).**
Item 11 (BM25/SPLADE touched-set) → item 12 (`dotI8AVX2`) → item 15 (HNSW
batching) → item 16 (parallel `Flat`) → item 14 (length bucketing) → item 18
(Qwen scratch). Item 11 first, because it's the largest measured ratio in the doc
and it re-baselines where retrieval time actually goes — which changes the
priority of items 29, 38 and 39.

**Phase 3 — kernels and layout (a month).**
Items 20, 22, 23, 24, 25, 35. These want the Phase-0 harness as arbiter and
several want an arm64 box for the real numbers.

**Phase 4 — GPU. ✅ ALREADY DONE, ahead of this doc.**
Item 21 landed on both platforms before this was committed (§7.5). The crossover
has been re-run and `docs/BENCH-gpu-results.md` carries both machines, so there is
no "untuned Phase-1 kernel" caveat left to add — those numbers are the tuned ones.

**Gate on evidence:** items 36 (needs an M2+/Graviton3 box), 37 (needs a golden
re-baseline), 38 and 39 (only after 11 and 12 re-baseline the retrieval profile).

---

## 9. One methodological note

> **See also: [`measuring-performance.md`](measuring-performance.md)** — the
> operational companion to this section. Every way a measurement misled us
> during this campaign is catalogued there with its numbers, along with the
> predicted-vs-measured scoreboard. Read it before designing a benchmark for
> any remaining item.

The pattern that produced the best findings here is the one already in your docs:
**`task-perf-linalg.md`'s arbiter discipline** — a microbenchmark proposes, an
end-to-end sweep disposes, and a negative result gets written down as
prominently as a positive one. Three of the six most interesting things in this
doc (item 12's amd64 gap, item 21's GPU kernel, item 6's inverted
`enterForward` defect) are cases where a *correctness-complete* component was
never measured under the load it actually serves. The harness work in item 1 is
what turns that from luck into process.

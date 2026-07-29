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
| **D** | `annmetal`'s `gemm_w8a8` is the untuned Phase-1 correctness kernel — **~8 GOP/s, ~5× slower than your SIMD CPU** — and the *tiled and simdgroup versions already exist in the same repo*, unused | `gpu/annmetal/backend.go:47-64` vs `gpu/metal_vit.go:377,~430` | The ANN-GPU crossover is never reached on Apple. Not a GPU-is-wrong result — a kernel-was-never-tuned result. |
| **E** | `EncodeBatch` pads to `Lmax` **inside index-ordered chunks**; ~92% of FLOPs run over padding-inflated M | `encoder/model.go:177-211`, `forward_batch.go:132` | 50 docs uniform on [20,512] ⇒ **48% of all linear-layer FLOPs computed on pad rows**. The fix (length bucketing) is provably bit-identical. |

---

## 2. Full ranked list

Legend — **Num:** `=` bit-identical · `~` ULP/reassociation, needs golden re-pin ·
`≠` changes numerics materially.

### Tier 0 — do first: cheap, and they unblock every measurement below

| # | Item | Area | Win | Num | Effort |
|---|---|---|---|---|---|
| 1 | Revive the 3 dead benchmark files; add warm-up + alloc accounting + a concurrent-QPS mode to `bench/harness.go` | bench | unblocks everything | — | S |
| 2 | SPLADE: hoist `log1p` outside the L×V max-reduce | encoder | 5–25× on that loop | **=** | S |
| 3 | `topk.Push`: hoist the threshold compare above the non-inlinable call | topk | **1.43× measured** on selection | = | S |
| 4 | `FlatI8.Query`: pool the score buffer; stop allocating a `Workspace` per query | ann | 10–25% now, large at N≥1M | = | S |
| 5 | Index (de)serialization: bulk `copy` instead of byte-at-a-time | ann | **~20–30×** | = | S |
| 6 | BERT/GTE forwards never call `enterForward()` → uncontrolled NumCPU² fan-out | encoder | removes oversubscription | — | S |
| 7 | SPLADE vocab matmul runs **serial** while the trunk parallelizes | encoder | up to ~2.3× on `Expand` | = | S |
| 8 | `gte.go:230` allocates 12.6 MB/`Encode` outside the scratch arena | encoder | 12.6 MB/call | = | S |
| 9 | `SpanCache` LRU → MRU/scan-resistant; add `MADV_WILLNEED` on map; pipeline `Touch(b+1)` | mmap | 0% → max hit rate | — | S |
| 10 | `bm25`: hoist `1/avgdl`; precompute per-posting impact at build time | bm25 | 1.5–2× scoring | ~ | S–M |

### Tier 1 — the big measured wins

| # | Item | Area | Win | Num | Effort |
|---|---|---|---|---|---|
| 11 | **(A)** BM25/SPLADE: touched-set selection + pooled accumulator | bm25, sparse | **10–50×** on selective queries | = | M |
| 12 | **(B)** Rewrite `dotI8AVX2`: 32 B/iter, 4 accumulators, bottom-tested | linalg | 2–3× int8 scan (4× w/ VNNI) | = (integer) | M |
| 13 | **(C)** SIMD `expF32`/`erfF32`/`tanhF32` + `SoftmaxRowsInto`/`GELUInto` | linalg→encoder, vision | 1.25–1.5× text, up to 2× vision | ~ or = (see item) | M–L |
| 14 | **(E)** Length-bucketed `EncodeBatch` under a token budget | encoder | 1.3–2× on ragged batches | **=** | M |
| 15 | HNSW: batch neighbour scoring through `Dot8x4` | ann | **1.36–1.40× measured** end-to-end | ~ (1 ULP) | M |
| 16 | `Flat.Query` is single-threaded; shard it + per-shard selector | ann | 1.74–2.08× on 2 cores; 4–8× typical | = | M |
| 17 | HNSW build: batch `prune`/`selectHeuristic`; kill 225 allocs/insert | ann | 1.5–2.5× build | ~ | M |
| 18 | `vision/qwen_encoder.go` has no scratch arena — **~15 GB alloc+memset per image** | vision | removes ~1.5 s of memset | **=** | M |
| 19 | `DequantizeRowInt4`: hardware integer divide per element | linalg | 2–4× on the dequant loop | = | S |
| 20 | Int8 register blocking (1×4 / 4×1) — the arm64 kernel has none | linalg | 1.2–1.6× GEMV, more at prefill | = (integer) | M |
| 21 | **(D)** `annmetal`: adopt the tiled/simdgroup kernel + on-GPU top-K | gpu | 5–30× the GPU path | = (int32) | M |

### Tier 2 — structural

| # | Item | Area | Win | Num | Effort |
|---|---|---|---|---|---|
| 22 | Q8 encoder path re-widens the whole int8 weight matrix to f32 **per matmul** — 113 M converts + ~0.9 GB DRAM/forward | encoder | ~100 ms/forward | **=** | S (SIMD) / M (fuse into pack) |
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
- **`gte.go:209`** builds `newRopeTable(L, headDim, RopeTheta)` per `Encode`,
  recomputed from scratch rather than cached per (maxSeq, headDim) — a second
  per-call allocation alongside item 8's 12.6 MB.
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

**Phase 4 — GPU, separately.**
Item 21, on its own track. Step 1 (point `annmetal` at the existing tiled kernel)
is small enough to slot into Phase 1; steps 2–4 are their own project. Re-run the
crossover sweep only at the end, and annotate the current result in
`docs/BENCH-gpu.md` as "untuned Phase-1 kernel" so it isn't later cited as a
verdict on Metal.

**Gate on evidence:** items 36 (needs an M2+/Graviton3 box), 37 (needs a golden
re-baseline), 38 and 39 (only after 11 and 12 re-baseline the retrieval profile).

---

## 9. One methodological note

The pattern that produced the best findings here is the one already in your docs:
**`task-perf-linalg.md`'s arbiter discipline** — a microbenchmark proposes, an
end-to-end sweep disposes, and a negative result gets written down as
prominently as a positive one. Three of the six most interesting things in this
doc (item 12's amd64 gap, item 21's GPU kernel, item 6's inverted
`enterForward` defect) are cases where a *correctness-complete* component was
never measured under the load it actually serves. The harness work in item 1 is
what turns that from luck into process.

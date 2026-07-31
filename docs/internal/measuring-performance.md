# Measuring performance in aikit

> **Status:** living document. Started 2026-07-29 during the
> [2026-07-28 perf campaign](perf-campaign-2026-07-28.md); extended as that
> campaign proceeds. Every failure mode below is one that actually happened
> here, with the numbers it produced — none of it is hypothetical.

The governing rule already existed, in
[`task-perf-linalg.md`](task-perf-linalg.md): **a microbenchmark proposes, an
end-to-end sweep disposes**, and a negative result is written down as
prominently as a positive one. That doc has the case where a persistent worker
pool was built correctly, measured honestly, and pulled because the arbiter said
no.

This doc is the operational half: *how* to build a measurement that can actually
settle the question, and the specific ways ours have failed.

---

## 0 · The short version

1. **Profile before optimizing.** Not to confirm a hypothesis — to find out you
   had the wrong one.
2. **The arbiter is end-to-end**, on a real checkpoint, with `benchstat` and
   n ≥ 6.
3. **Compare variants inside one process** wherever possible. Cross-invocation
   deltas under ~5% are unmeasured on our boxes, not small.
4. **A per-kernel benchmark cannot see what a change does to the kernels around
   it.** It is evidence about the kernel, never about the program.
5. **Write the gate before you trust the number**, and mutation-check the gate.

---

## 1 · The failure catalogue

### 1.1 The benchmark measures something you did not change

The most embarrassing one, and it happened. Item 7 routed SPLADE's vocabulary
projection onto a new `matmulBTColsInto`, then measured
`BenchmarkSpladeVocabProj` — which still called the *old* `matmulBT`. It
reported "no change" through a 5× speedup.

**Guard:** after a change, confirm the benchmark reaches the new code before you
believe a null result. A counter, a `t.Log`, or a deliberate `panic()` in the new
path costs seconds. A null result is a claim and needs the same scrutiny as a
positive one.

It happened a second time, and the habit is what caught it. Item 24 pads
`packedFill`'s pack stride; a sweep over pad ∈ {0,4,16} at four shapes came back
uniformly inside noise. The reason was not that padding does not help — it is
that `blockedFill` gates `packedFill` on `has2x8Kernel`, which is **false off
arm64**. The whole function is dead code on this box and the benchmark measured
an unexecuted branch. The change was reverted rather than shipped on the strength
of a null result that meant nothing.

**Corollary worth internalizing: a uniformly flat sweep is itself a signal.**
Real tuning parameters produce *some* variation. When every setting ties exactly,
suspect that none of them ran.

### 1.2 The compiler deletes the thing under test

Item 19's benchmark reported the *unoptimised* form running **16× faster** than
the optimised one. `const group = 32` let the compiler strength-reduce the
integer divide the whole optimisation was about, so the "before" loop never
executed a divide at all. Changing the constants to `:=` locals did **not** fix
it — SSA constant-propagates those too.

The tell was arithmetic: the "before" case was running at **0.84 cycles per
element**, which is not possible for a loop containing an `IDIV`.

**Guard:** obtain any value whose *runtime variability* is the point through a
`//go:noinline` source. And sanity-check ns/op against a cycle budget — if a
number is physically impossible, it is measuring nothing.

### 1.3 Cache-warm reuse flatters whichever variant streams more memory

Comparing parallel matmul strategies with a single reused weight matrix lets `b`
sit in L3 across iterations. That erases the exact cost that distinguishes the
strategies — a row split streams the weights once *per worker*, a column split
partitions them.

**Guard:** rotate a **bank** of weights sized like the real model (we use 12, one
per transformer layer) so no iteration reuses the previous one's residency. See
`BenchmarkParallelAxis` in `encoder/parallel_sweep_test.go`.

It cuts the other way too, which is easy to miss. Item 22's int8→f32 widen
benchmarks at **0.50 ns/elem** on one L3-resident weight but costs **~1.7** in a
real forward, which streams twelve different cold ones. There the kernel ratio
(5.7×) *understated* the end-to-end effect (−58% at short input). A warm kernel
benchmark is not conservative — it is simply measuring a different workload, and
which direction the error goes depends on whether the change is bound by
compute or by memory.

### 1.4 Per-kernel benchmarks cannot see cross-kernel interference

The sharpest lesson of the campaign (§7.12). Parallelizing *only* SPLADE's
vocabulary projection made the projection **5× faster** and made short-query
`Expand` **6.7% slower** (p=0.005). The projection benchmark was completely
honest about the projection; it simply could not observe that dropping an
all-core burst into an otherwise serial forward taxes every serial stage around
it.

Isolated with a memory-free all-core spin burst as a control:

```
trunk alone                                97.8 ms
trunk after a memory-free all-core burst  110.3 ms   (+12.8%, boost clock)
trunk after the real parallel projection  129.2 ms   (+32%,   boost + cache)
```

So ~13 points are the package dropping out of single-core boost and ~19 are
cache/memory thrash. **Neither is visible in any per-kernel number.**

**Guard:** when a change alters *how much of the machine* a stage uses, the
arbiter must be end-to-end. And when you see a stage-local win with an
end-to-end loss, suspect interference before suspecting the measurement.

### 1.5 Near a crossover, the microbenchmark stops being predictive

Columns beat rows at **every** shape in the per-kernel sweep, including 1.03× at
GTE's L=690. Making columns unconditional won SPLADE L=91 by 17.4% and **lost**
GTE L=690 by 2.32% (p=0.008).

**Guard:** take crossover boundaries from the end-to-end number, not the kernel
number. Express the boundary as the condition that actually causes the effect
(*can the row split fill the machine?* → `M < minRowsPerWorker·NumCPU`) rather
than as a fitted constant, so it stays meaningful on other core counts.

### 1.6 Cross-invocation drift is larger than most wins

While measuring the GELU split, GTE appeared to **regress 4.5% (p=0.008, n=5)**
— on a code path that does not exist in GTE, which has no `gelu()` call at all.
The whole benchmark had drifted as the box heated over a long session.

**A provably-unchanged code path moving 4.5% bounds the cross-invocation noise
floor at ~5% on `nvidia-rtx2070s`.** A low p-value does not help: it measures
consistency within a run, not equivalence of conditions between runs.

**Guards:**

- Compare variants **in one process**, alternating, when the change can be put
  behind a switch. `BenchmarkSpladePhases` and the `TestTrunkParallelAB` probe
  both did this.
- When you must compare across invocations, **re-run the pair in the opposite
  order**. If the sign flips, it was thermal.
- Treat sub-5% cross-run deltas as *unmeasured*, and say so.

### 1.7 Synthetic fixtures with the wrong data distribution

Item 2 (hoisting `log1p` out of SPLADE's max-reduce) measured **17.8×** on a
synthetic fixture at 50% positive density. On a real trained SPLADE — whose
logits are mostly negative by construction — it is **1.28–1.47×**, and pooling
is ~0.5% of `Expand`, so it is invisible end-to-end.

**Guard:** if a win depends on data *distribution* (sparsity, density, sequence
length, vocabulary hit rate), a synthetic fixture cannot size it. Get the
checkpoint. See `scripts/README.md` for the SPLADE and GTE fetch recipes.

### 1.8 Allocation wins are not latency wins

Item 8 removed **12.7 MiB per `GTE.Encode`** (−43% B/op) and produced **no
latency change** — +1.38% (p=0.029) at the longest shape. The item never claimed
latency, and it delivered exactly what it promised. But it is worth stating
plainly: removing 12.7 MiB from a 4.2 s call buys nothing at single-call
latency. The value is GC pressure under concurrency, which the benchmark did not
exercise.

**Guard:** report `B/op` and `sec/op` as separate results with separate claims.
If the claim is about GC pressure, the benchmark has to have concurrency in it.

Item 18 is the same shape and shows how to state it honestly. Removing 573 MiB
of alloc+**memset** per forward is ~57 ms at ~10 GB/s, and 57 ms of a 1.88 s
forward is 3.0% — measured 2.8%. The memset arithmetic in the estimate was
right; what was oversold was the *share*. The real prize there is that
allocation became depth-independent (103 MiB at depth 4 and at depth 8), which
turns a ~15 GB peak into ~0.5 GB at production dims — a memory-ceiling result
that the latency number does not express at all.

### 1.9 Profile attribution lies in two specific ways

- **Goroutine samples do not appear in the spawning function's `cum`.** After
  parallelizing the vocabulary projection, `pprof -list expandIDs` showed the
  projection line as *free*. Its work had moved to worker goroutines, which
  `pprof` roots separately. Read total samples ÷ wall time to get average
  parallelism (ours went 4.2× → 10.2× over the campaign) and reconcile against
  it.
- **Growth rates hide stages until they dominate.** The linear projections are
  O(L); the attention score matrix is O(L²) and every element goes through
  `math.Exp`. At L=22 softmax is ~3% of a forward and invisible. At L=690 it was
  **half the call** and entirely serial. Profile at the length you care about.

### 1.10 Fixing the largest stage promotes the next one

Amdahl, but it keeps surprising: after the softmax split, `gelu` went from
"noise" to **27%** of `GTE.Encode`. After that, `dotFMA8` at 72% became the
floor.

**Guard:** re-profile after every accepted change. Budget for it — the sequence
softmax → GELU → GEMM was three re-profiles, and each one changed what to do
next.

### 1.11 Measurement tables do not transfer between machines

The campaign's item 13 table was measured on an M1 Pro. On the 3700X:

| kernel | doc (M1 Pro) | measured (3700X) | transferred? |
|---|--:|--:|:--:|
| GELU-erf (`math.Erf`) | 28.9 ns/elem | 29.4 | ✅ |
| `math.Exp` | 34.3 ns/elem | 15.0 | ❌ (2.3× off) |

Half the item's premise held and half did not, which changed what was worth
building (the "10–30× on those loops" figure was unreachable without SIMD; a
pure-Go f32 kernel still delivered the predicted *end-to-end* band).

**Guard:** re-measure the per-kernel table on the box you are optimising for
before designing against it. Cross-machine comparisons belong in the normalized
"speedup over each box's own CPU" form — see `docs/BENCH-gpu-results.md`.

### 1.12 Vacuous benchmarks and metrics

- `b.ReportMetric` called **before** `b.Loop()` is discarded — a custom metric
  that silently never prints is a metric you are not collecting.
- A benchmark whose two arms differ by a `copy()` the other arm does not do is
  biased by that `copy()`. Say so, or restructure.
- A gate that cannot fail proves nothing. **Mutation-check every gate**: break
  the thing deliberately and confirm the test goes red. This caught a
  non-discriminating bit-identity test and an off-by-one row range in the same
  session.

### 1.13 A blocker you did not verify is not a blocker

Item 13's vision half was deferred with "no vision checkpoint on this box to
re-verify parity against". Both fixtures were already present, and neither is a
download: `scripts/pin_siglip_vision.py` **generates** a tiny random
`SiglipVisionModel` locally, and `scripts/gen_siglip_bench.py` generates the
real-sized towers. `TestSiglipEncoder_parity` had been passing the whole time.

The same session then fetched a cross-encoder checkpoint and found the largest
single win of item 13 (−33.8%) sitting behind it.

**Guard:** before reporting work as blocked on a missing fixture, run the test
and read the skip. Check whether the fixture is *generated* rather than
downloaded — this repo has three generator scripts and two of them need no
network at all. `ls testdata/` costs nothing; a deferred item costs a session.

### 1.14 A gate that compares a run against itself cannot fail

Item 18 moved the Qwen ViT's per-layer buffers into one reused arena, whose
hazard is a buffer read before it is written — it would carry the previous
layer's values. The first gate ran the forward twice and compared the results.

It could never have worked. Stale-arena data is **deterministic**: both runs read
the same stale bytes and agree, wrongly. It passed a mutant that dropped an
entire attention segment.

The working version poisons every arena buffer with NaN and compares against a
run on a *fresh* arena — a known-good reference, not a second sample. Even that
missed a partially-written output on the first attempt, because the test
allocated its own destination buffer while production passes one from the arena.

**Guard:** ask what reference a gate compares against. "The same code, again" is
not one. And make the test's setup match production's aliasing exactly — for an
arena test that means the outputs come from the arena too.

### 1.15 A benchmark's name is not evidence of its shape

`BenchmarkCrossEncoderScore/L200` ran at **L=512**. Its document tokenized to 897
tokens and the tokenizer truncated to `maxSeq=512`, so every number attributed to
"L≈200" was really a 512-token forward.

It cost two things. Item 27 appeared to have **no effect** on the cross-encoder
(the threshold it removes only bites below L≈250, and 512 is above it) — the real
figure at L=200 is **−50.5%**. And item 13's cross-encoder result had been
compared against a prediction the doc made specifically for L≈200, which was not
the length measured; at the right length it is −29.4%, not −33.7%.

**Guard:** for any benchmark whose shape comes from *text*, print the resulting
token count once and check it against the name. Truncation to a model's `maxSeq`
is silent, and a subtest name is an assertion no one verifies. The fixed version
spans `L200` and `L512` deliberately, with a comment saying why.

---

## 2 · The recipe

```
1. Profile the real workload, at the size you care about, on a real checkpoint.
2. Attribute to a STAGE before attributing to a function.
      -> a temporary phase-split benchmark (trunk / projection / tail) is worth
         writing and throwing away; BenchmarkSpladePhases located a regression
         in a stage whose code had not changed.
3. Form the hypothesis. Predict a magnitude. Write the prediction down.
4. Build the kernel benchmark — with a weight bank, and a check that it reaches
   the new code.
5. Build the end-to-end arbiter. benchstat, n >= 6, same process if possible.
6. If |delta| < 5% cross-run: re-run in the opposite order before believing it.
7. Write the gate. Mutation-check it.
8. Record the measured number AND the prediction it missed.
```

### Commands

```sh
# End-to-end arbiter with statistics
go test ./encoder/ -run xxx -bench 'BenchmarkGTEEncode' -benchtime 5x -count=6 \
  | grep -E '^(goos|goarch|pkg|cpu|Benchmark)' > after.txt
go run golang.org/x/perf/cmd/benchstat@latest before.txt after.txt

# Profile, then attribute by line
go test ./encoder/ -run xxx -bench 'BenchmarkGTEEncode/L512' -benchtime 10x \
  -cpuprofile cpu.prof -o enc.test
go tool pprof -top -nodecount=12 enc.test cpu.prof
go tool pprof -list 'expandIDs' enc.test cpu.prof
```

`-benchtime Nx` (fixed iterations) rather than a duration keeps long benchmarks
bounded — `GTE.Encode` at L=690 was 4.2 s per iteration, so the default 1 s
budget silently becomes many minutes.

---

## 3 · Known machine facts

Numbers here are for calibration, not for citing as results.

### `nvidia-rtx2070s` — AMD Ryzen 7 3700X (8C/16T, split-CCX L3), Linux

| fact | value | source |
|---|--:|---|
| cross-invocation benchmark drift | **~5%** | §1.6 |
| serial stage slowdown after an all-core burst | +12.8% (clock) / +32% (clock+cache) | §1.4 |
| `math.Exp` (f64, scalar) | 15 ns/elem | item 13 |
| `math.Erf` inside GELU | 29.4 ns/elem | item 13 |
| `linalg.ExpF32` | 5.6–7.6 ns/elem | item 13 |
| `math.Tanh` inside SigLIP's GELU | ~30 ns/elem | item 13 |
| scalar int8→f32 widen, cold weights | ~1.7 ns/elem | item 22 |
| AVX2 int8→f32 widen | 0.089 ns/elem | item 22 |
| row-split matmul worker cap | `(M+31)/32` → **3 workers at M=91** | §7.14 |
| exact f32 dot scan, per candidate | 43.5 ns (d256) / 118.4 ns (d768) | item 38 |
| Hamming scan, per candidate (POPCNTQ) | 2.96 ns (d256) / 8.31 ns (d768) | item 38 |
| `math/bits.OnesCount64` without POPCNTQ | **1.55× slower** — not intrinsified at `GOAMD64=v1` | item 38 |
| streaming scan bandwidth ceiling | **~26 GB/s** (both scans hit it at N≥100k) | item 38 |

The split-CCX L3 is why the row-split matmul axis lost here and not on the M1
Pro: replicating the weight matrix per worker is far cheaper across a large
unified cache.

### `apple-m1pro` — Apple M1 Pro (8 core), macOS

`parallelThreshold`'s tuning table in `encoder/parallel.go` was measured here.
Treat every constant in it as unverified on amd64 until re-measured — §7.14 found
the *axis* itself was wrong on the 3700X.

| fact | value | source |
|---|--:|---|
| item 24 power-of-two packed stride | **REAL here** — 4096 B `kSpan`=1024 stride costs 8% in the kernel / **−9.8% on large-encoder fc2** (K=4096); 16-f32 pad fixes it | campaign item 24 |
| serial `packedFill` fraction of kernel peak | 75–81% of ~42 GMAC/s (fc2 M512/M690) — item 23's a-re-read gap | campaign item 23 |
| A1 `EncodeBatch` scaling knee | **near-linear to 6 workers (P-cores), then +1.04× from the 2 E-cores** | perf-amdahl-apple-m1pro §3 |
| A1 speedup at NumCPU=8 | **5.23×** (amd64's 8C+SMT gave 8.21×) | " |
| `StaticModel.Encode` tokenize/pool split | **50.6 / 49.4** (amd64 62.9/37.1 — pool much heavier here) | perf-amdahl-apple-m1pro §1 |
| `encodeIDs` pool MAC | **memory/latency-bound, not compute** — 0.7 ns/MAC warm (unroll +7%), 1.42 real (cold row gathers); SIMD won't help | 2026-07-31 |
| `utf8.ValidString` over corpus | 0.16% of the tokenize stage (amd64 1%) | " |
| scalar int8→f32 widen | **0.34 ns/elem** (5× faster than amd64's 1.7) | item 22b |
| NEON int8→f32 widen (`dequant_i8_arm64.s`) | **0.098 ns/elem** (3.43× over scalar) | item 22b |
| `GTE.Encode` L512 profile | **`dotNEON2x8` 52%**, `memmove` 17%, `packedFill` 20% | 2026-07-30 |
| `dotNEON2x8` f32 GEMM kernel | **~42 GMAC/s ≈ 95% of single-core FMLA peak** (compute-bound) | item 37 |
| outer-product 8×8 kernel (item 37) | 46 GMAC/s = **1.10×** raw; 4× slower with packing → reverted | item 37 |
| GTE L512 addressable breakdown | `dotNEON2x8` 51% (at peak), thread fork/join ~21%, `packedFill` `memmove` 13% (**not addressable — overlaps compute**) | 2026-07-30 |
| Q8 forward hot path | widen→**f32 `dotNEON2x8`** (no int8 GEMV kernel); item 20 is a *decode* (goinfer) lever, not encoder | 2026-07-30 |
| cols-first matmul axis | **correct here too** (not just amd64) | 2026-07-30 |
| cols→rows crossover, GTE end-to-end | **M ≈ 80–96** (rows wins from L96) | BenchmarkGTEAxisProbe |
| forced-rows penalty at small M | up to **+178%** (L32: row split → 1 worker = serial) | " |
| rows' mid-length edge (L96–128) | 3–8% (wide unified cache ⇒ cheap b-replication) | " |

**The f32 encoder is at its arm64 floor on the M1 Pro (2026-07-30).** `dotNEON2x8`
runs at ~95% FMLA peak (killing microkernel items 25/37), fork/join is already
minimized (the spin-park pool was built and pulled, §6), and the last candidate —
`packedFill`'s **13%** b-panel `memmove` — turned out **not addressable**. It runs
only for K≥`packKThreshold`=2048 (fc2/down) and re-packs the immutable weights every
forward, so pre-packing them once at load *looks* like a 13% win. Built it
(`PackWeightBT`/`MatmulBTPacked`, bit-identical to `MatmulBT`, mutation-checked) and
measured: **~0–3%, within noise.** The 13% is pprof *self*-time, not addressable
wall-clock — on the M1's out-of-order core the copy overlaps the compute-bound
`Dot2x8` of adjacent tiles (§1.4 again: a profile attributes cycles it cannot prove
are on the critical path). Reverted. **Measuring before the architectural
integration saved the whole change for a ~1% gain** — the point of measure-first.

**The 2026-07-30 axis re-check (Task 0 of the arm64 handoff): cols-first stays
the default.** The fear was that the amd64-driven cols-first flip would be
neutral-to-negative on the M1 Pro, whose original table showed row splitting
winning 2.6–4.5×. It is not: columns are *essential* at short input (the row
split collapses to one worker below M=64) and only lose 3–8% in the L96–128 GTE
band. Capturing that band needs the *narrow-N* shapes (fc11/fc2, N≤3072) on rows
too — a shape-aware upgate-only carve-out recovered none of it — so a crossover
change would land on exactly the shapes plain BERT depends on, for a sub-10%,
band-limited, thermally-hard-to-measure gain. Not worth the BERT regression risk;
left as-is. `wantParallelCols`'s `M < minRowsPerWorker·NumCPU` crossover (256 here)
is conservative for GTE but safe across models.

---

## 4 · Scoreboard: predicted vs measured

The campaign's recurring pattern, kept honest in one place. **Mechanisms have
been consistently right; magnitudes consistently optimistic.**

| item | predicted | measured | |
|---|---|---|:--:|
| 2 · SPLADE `log1p` hoist | 17.8× (synthetic) | 1.28–1.47× on pooling, ~0 end-to-end | ❌ |
| 3 · `topk.Push` threshold | 1.43× | 1.05× end-to-end | ❌ |
| 5 · index (de)serialization | 20–30× | 5.14× | ❌ |
| 7 · SPLADE vocab matmul | ~2.3× on `Expand` | 1.05–1.19× | ❌ |
| 8 · GTE allocations | 12.6 MB/call | 12.7 MiB/call | ✅ |
| 11 · touched-set selection | large | 1.3–218×, overshot | ✅ |
| 12 · `dotI8AVX2` | in-band | 2.10× kernel / 2.02× scan | ✅ |
| 13 · f32 transcendentals (text) | 1.25–1.5× text | 1.25× (−20.1%) | ✅ |
| 13 · …on the D=384 cross-encoder | ~29–36% at L≈200 | **−29.4%** at L=200 | ✅ |
| 13 · …on SigLIP | up to 2× | 1.24× geomean (1.44× at 576 patches) | ❌ |
| 22 · q8 weight widen | ~113 ms/fwd, L-independent | ~190 ms/fwd, L-independent | ✅ |
| 22b · arm64 NEON widen | 6–8× kernel | 3.43× (arm64 scalar already 5× amd64's) | ❌ |
| 22(b) · fuse widen into pack | remaining ~33 ms + 0.9 GB round-trip | **−28%/−10%/−4.8% fwd at L=8/64/256** | ✅ |
| §4.2 · gate col-block (arm64) | latency-neutral footprint win (amd64) | **+3.5–6.9% swigluMLP; reverted (6P+2E fork/join)** | ❌ |
| 37 · outer-product kernel (arm64) | ~1.45× (2× in the µkernel, amd64) | 1.10× raw / net-negative (dot2x8 at 95% peak) | ❌ |
| 24 · packed-stride pad (arm64) | "may be a no-op, measure first" | **−9.8% on large-encoder fc2 — real** | ✓ (rare: measured MORE than the guard expected) |
| §5 · StaticModel word presum | ~29% pool collapse (14% of Encode) | wash serial / **+31% batch** (cache overhead + contention) | ❌ |
| 39 · MaxScore long-query pruning | faster than exhaustive at >8 terms | **2.5–5.7× SLOWER** (TAAT accumulator beats DAAT when selectivity is uniform) | ❌ |
| 18 · Qwen ViT arena | ~15 GB/image, ~1.5 s memset | −70/−85% B/op; **~3%** of latency | ⚠️ |
| 27 · 4-MFLOP naive threshold | 3–9% for L<250 | **up to −50.5%** | ❌ (under) |
| 24 · packed-stride aliasing | "free" | unreachable on amd64; reverted | — |
| 32 · vision preprocess | 2.3× | 2.45× on convert+resize; **−31.4%** end-to-end | ✅ |
| 16 · shard `Flat.Query` | 4–8× typical | **1.73–2.26×** on 16 threads | ❌ |
| 30 · bm25 tokenizer allocs | 787 → ~10 | **983 → 2**, −44.7% time | ✅ |
| 29 · bm25 8-byte posting | 1.5–2× scoring | ~0% scoring; **−50% index memory** | ⚠️ |
| 10 · bm25 precomputed norm | 1.5–2× scoring | ~0% scoring | ❌ |
| 44 · bm25 touched-set order | ~18% (pprof share) | **−43.5%** on the query | ❌ (under) |
| 4 · FlatI8 query scratch | 10–25% time | −5.4% time; **−99.2% B/op** | ⚠️ |
| 34 · double `enterForward` | correctness-ish | unmeasurable on this box's checkpoints | — |
| 15 · HNSW batched scoring | 1.36–1.40× | **1.36×/1.33×** | ✅ |
| 17 · HNSW build batching | 1.5–2.5× build | 1.34×; allocs 225 → **89**/insert | ❌ (time) ✅ (allocs) |
| 14 · length-bucketed batch | 1.3–2× ragged | **1.15×** ragged, neutral uniform | ❌ |
| 28 · CrossEncoder batch API | "unlocks item 14 for rerank" | **7.56×**, but from parallelism not tokenization | ✅ (size) ❌ (cause) |
| 9 · SpanCache eviction | "0% → max hit rate" | **0% → 10.9/45.0/88.6%**; the 0% was literal | ✅ |
| 9(b) · Touch(b+1) prefetch (arm64) | overlap fault with compute | **+12%**; darwin DONTNEED no-op, no fault to hide — reverted | ❌ |
| §4.5 · Q8 release peak RSS (arbiter) | 727→242 MiB (amd64) | **726.2 MiB on M1 Pro — no change** (DONTNEED inert) | ❌ (does not transfer) |
| 31 · QKV transpose | 3.3× on the transpose | **0.06%** of the forward — closed unimplemented | ❌ |
| 1 · revive dead benchmarks | "unblocks everything" | 3 files revived; harness gained allocs + QPS | ✅ |
| 26 · `math.Round` not intrinsic | ~5% of the matmul | **0.61%**; the proposed fix is a wash — closed | ❌ |
| 33 · MoE expert grouping | "group tokens by expert" | **1.81×**; `moeMLP` was 48.9% of the encode | ✅ |
| 19 · `DequantizeRowInt4` | 2–4× | 4.93× | ✅ |
| 38 · binary Hamming prefilter | ~10× first stage | **13–26×** end-to-end, geomean 18.6× | ✅ |
| 39 · WAND dynamic pruning | 2–10× on bm25 **and** sparse | **3.88×** bm25 mixed query; **reverted** on sparse | ✅ (bm25) ❌ (sparse) |
| §4.3 · `HNSW.WriteTo` | full copy → 1 MB staging | **131.0 MB → 65.5 KB** transient; peak RSS 181.3 → 125.5 MiB | ✅ |
| §4.3 · HNSW arena `Load` | ">2.1 M allocs → 2" | **153,396 → 8** (2.164 → 0.004 per doc) | ✅ |
| §4.3 · the baseline itself | "MarshalBinary allocates the blob once" | it allocated it **twice** — 131.0 MB for a 58.3 MB blob (bad capacity estimate) | ✓ (worse than predicted) |
| 15b · HNSW search-heap pooling | campaign recorded it as "slightly SLOWER"; expect allocs only | **19 → 2 allocs, 13.4 KB → 1.2 KB, and −27.1% time** | ✅ (reverses a recorded dead end) |

Two entries were **found by measurement rather than predicted** and are the
largest encoder wins of the campaign: the parallel axis being wrong on amd64
(§7.14) and the elementwise stages dominating once the matmuls were parallel
(§7.15, `GTE.Encode` L=690 3.28×).

Two rows are worth singling out as counter-examples to the "magnitudes are
optimistic" pattern, and they have something in common. Item 13's cross-encoder
share came from a **closed form** (activation ÷ FFN-matmul = `t_act/(k·D·t_mac)`,
which grows as D shrinks). Item 22's widen cost came from a **structural
argument** (O(N·K) work independent of M ⇒ amortizes as 1/M), and its
L-independence held exactly, to the point where the measured per-forward cost was
the same at 10 tokens and at 350.

**Derived predictions have held; measured-then-scaled ones have not.** When an
estimate comes with a formula, trust its shape and re-measure only its constant.

Item 27 is the counter-case and the only large UNDER-estimate so far: "3–9% for
L<250" against a measured 50.5%. The formula there sized *one* diverted matmul
and forgot that the diverted shape is the per-head attention matmul, which recurs
heads × layers × 2 times per forward — 144× in a 12/12 model. **An estimate that
sizes a kernel and omits its multiplicity is wrong by the multiplicity**, and the
error is unbounded in the optimistic direction as easily as the pessimistic.

---

### 1.16 A ratio without a denominator is not a result

Item 32's Win column read "**2.3× measured**". End-to-end it delivered −31.4%.
Both are correct: `image.Decode` is 46% of `Preprocess` and the item does not
touch it, so 2.3× of the convert+resize work (measured 2.45×) becomes −31.4% of
the call. Nobody was wrong; the table just never said what the ratio was over,
and the obvious reading was the wrong one.

**Guard:** record ratios as "X× on <the thing measured>", and when the thing
measured is not the whole operation, give the end-to-end number beside it. The
same applies in reverse to an estimate you are about to act on — establish its
denominator before budgeting effort against it.

---

### 1.17 Convert a bandwidth-bound result to GB/s before believing a core-count estimate

Item 16 sharded `Flat.Query` across 16 threads and got **1.73–2.26×**, where the
estimate said "1.74–2.08× on 2 cores; 4–8× typical". One number explains it:

| index | working set | serial | sharded |
|---|--:|--:|--:|
| 10k×128 | 5.1 MB (L3) | 20.0 GB/s | **45.1 GB/s** |
| 200k×384 | 307 MB (DRAM) | 14.6 GB/s | 25.2 GB/s |

The DRAM cases sit at this box's practical dual-channel ceiling and do not move
with core count. A flat cosine scan reads each byte once and does two FLOPs per
float, so it is bandwidth-bound by construction — parallelism only closes the gap
between one core (~14 GB/s) and the memory system, then stops. The L3-resident
row is the control: same code, no DRAM wall, 2.26×.

**Guard:** for any scan-shaped workload, divide the working set by the time and
compare against the machine's memory bandwidth BEFORE estimating a speedup from
core count. If the serial number is already a large fraction of the ceiling, the
available win is `ceiling / serial`, not `cores`. This is cheap to compute and
would have predicted the result exactly.

---

### 1.18 An earlier item in the same campaign can spend a later one's win

Items 10 and 29 each predicted "1.5–2× scoring" for bm25. Both landed,
bit-identically, and scoring did not move at either corpus size (p=0.310 and
p=0.937).

Neither estimate was wrong when written. **Item 11 — in the same document —
invalidated them before they were attempted.** Its touched-set accumulator
replaced an O(corpus) scan with O(postings touched), which moved the bottleneck
off the posting walk that 10 and 29 both optimize. Profiling the query loop
afterwards shows the largest cost is now the touched-set *sort* at ~18%, with
the posting scan not visible above GC.

**Guard:** a campaign's estimates are a snapshot of one profile. After landing
anything that changes the *shape* of a hot path, re-profile before trusting any
remaining estimate that touches the same path — and re-read the sibling items for
ones that were sized against the old shape. The cheap version of this is to
re-run the profile that generated the estimate and check the line is still there.

The corollary is that items 10 and 29 were still worth landing — item 29 halves
the index's dominant storage (381.7 MB → 190.9 MB at 200k docs). Its Win column
just led with the effect that evaporated rather than the one that survived, which
is §1.16's lesson in a different costume: **state which quantity a claim is
about.**

---

### 1.19 A pprof percentage is a share of what pprof measured

Item 44 was sized at "~18% of a common query" from a `pprof -top` line. The
measured win on the query is **−43.5%**.

The profile was not wrong; it was answering a different question. The benchmark's
setup builds a 200k-document index, which takes seconds, while the measured loop
takes ~0.5 ms per iteration — so a large share of every sample belonged to
`Build`, not to the query. `sortTouched` was 18% *of the whole binary run* and a
far larger share of the thing being optimized.

**Guard:** before quoting a pprof share as an item's size, check that the profile
window is dominated by the code under test — raise `-benchtime` until the setup
is negligible, or profile the operation directly. Cross-check the share against
wall-clock: 18% of a 0.5 ms query is 90 µs, which was implausible next to a
sort of ~100k int32s.

---

### 1.20 A timing ratio is the wrong shape for a capability claim

`TestEncodeBatch_speedup` asserted that batched encoding is ≥2× a sequential
loop, from ONE sample of each. It was observed failing at 1.82× and 1.97× while
measuring 2.6–2.8× when run alone — because `go test ./...` runs packages
concurrently, so it routinely shares the machine with a dozen other packages'
tests.

Nothing was wrong with the threshold. The claim is that the machine *can* reach
2× through parallelism, and a single wall-clock ratio under unknown contention
cannot test that: it measures the worst moment it happened to sample.

**Guard:** when a test asserts a capability rather than an average, take
best-of-N. Genuinely broken parallelism still fails every attempt, so the
assertion keeps its strength; only the contention flake goes away. The same
applies to any test with a hard floor on a speedup, a throughput, or a latency
budget.

---

### 1.21 Some correctness properties are not observable at test scale

Item 15's rewrite is justified by being *order-preserving*: the push loop must
see neighbours in the same sequence, so the evolving top-k threshold behaves
identically. Two successive gates failed to demonstrate it. A top-1 comparison
passed a mutant that **reversed the push order entirely**; strengthening it to
full ranked lists over 4,500 hits did not catch that mutant either — reversing
within a neighbour group simply does not change the returned results at these
sizes.

The property is real and worth preserving; it is what makes the transformation
obviously correct instead of empirically correct. But **no test here can
demonstrate it**, and pretending otherwise would be worse than saying so.

**Guard:** when a mutation of the exact property you claim does not fail your
gate, do not conclude the gate is adequate — establish what it *does* test, and
say which part of the argument rests on reasoning rather than evidence. Here the
gate verifies the weaker property callers depend on (identical ranked results)
and does catch a genuinely wrong gathered row; the order argument stands on
reading the code.

---

### 1.22 A quality gate catches what a correctness gate cannot

Item 17 returned build scratch instead of copying it. Both callers appeared to
copy the ids out immediately — a reading taken from the first six lines of the
caller, and wrong. `Add` iterates the same slice AGAIN to add back-edges and
calls `prune` inside that loop, which re-enters `selectHeuristic` and overwrites
the scratch mid-iteration.

Nothing crashed. `-race` was clean. Every structural test passed. The graph was
simply, quietly worse: **recall@10 fell from 1.00 to 0.83**, and the only things
that noticed were the recall gates.

**Guards, two:**

- Verify an aliasing contract against *every* path out of the caller, including
  indirect re-entry. "The caller copies it immediately" is a claim about the
  whole function, not its first statement.
- Keep quality gates (recall, cosine, parity) running on work that looks purely
  structural. The instinct is that a refactor which preserves arithmetic needs
  only correctness tests; this one preserved arithmetic perfectly and still
  degraded the output, because it corrupted the data structure rather than the
  numbers.

---

### 1.23 Configure the benchmark so the defect can occur

Item 14 removes work wasted on padding inside a multi-sequence batch. The first
benchmark ran 16 texts at `concurrency=NumCPU`, which gives ONE sequence per
forward — B=1, no padding in either version — and reported no difference. It was
measuring two identical code paths.

The parameter that had to be pinned was not the corpus but the *concurrency*:
padding only exists when `len(texts) > concurrency`. Fixing it at 4 with 32 texts
forces B=8, and the effect appears immediately (−13.0%, p=0.008).

**Guard:** write down the condition under which the defect occurs, then check the
benchmark satisfies it — as a precondition, not an afterthought. Related to §1.1
(does the benchmark reach the code?) but distinct: here the code ran, it simply
had nothing to do. **Pair it with a control** that should NOT improve — a uniform
batch here — because a control that stays flat is what distinguishes a real
effect from a lucky configuration.

---

### 1.24 A skipping benchmark is a passing benchmark

Three benchmark files in this repo pointed at paths that do not exist — leftover
`ken` locations, one of which (`../../../testdata/repo/…`) resolved to the
*parent of the repo root* and so could never have matched in any checkout. Each
called `b.Skipf`. `go test ./...` was green throughout, and `bm25.Tokenize` — the
hottest indexing function in that package — had **no live benchmark coverage at
all** for as long as those files had been checked in.

**Guard:** a fixture checked into the repository is not an environment
condition. `b.Skipf` is for "this machine lacks a 4 GB checkpoint"; for
"this file is in git", use `b.Fatalf`, so a broken path fails loudly instead of
reporting success by omission. And periodically run `go test -bench . ./... 2>&1
| grep -c SKIP` — a skip you did not intend is a measurement you are not taking.

---

### 1.25 A guard against one measurement error can cause another

Item 26's first benchmark routed every value through a `//go:noinline` source,
on §1.2's reflex of defeating constant folding. But the values came from a
**slice**, which the compiler cannot constant-fold, so the guard defended
against nothing — and cost a function call per element, ~2 ns, against a ~1 ns
difference under test. It made `math.Round` appear 2× slower than it is and the
comparison unreadable.

**Guard:** `//go:noinline` sources are for values the compiler could otherwise
*prove* constant. Slice contents, function parameters and file input are already
opaque. Adding the guard anyway is not free and not neutral — it adds a cost to
both arms that can exceed the effect you are measuring, which is §1.2's failure
mode wearing §1.2's uniform.

---

### 1.26 "No checkpoint for that" is a claim to check

Twice this campaign an item was deferred as unmeasurable for want of a model,
and twice the model was available:

- Item 13's vision half was deferred with "no vision checkpoint on this box".
  Both fixtures were already present, and neither was a download — two scripts
  in `scripts/` *generate* them offline (§1.13).
- Item 33 was deferred as needing "a MoE checkpoint this box doesn't have". The
  config fields the loader reads name `nomic-ai/nomic-embed-text-v2-moe`, a
  1.8 GB download. It turned out to be worth **1.81×** on a path that was 48.9%
  of the encode.

**Guard:** before recording an item as blocked on a fixture, spend the two
minutes to find out which model it needs — read the config fields the loader
actually parses — and whether it is downloadable or generated. Both deferrals
cost more than the check would have.

---

### 1.27 An approximate index measured on structureless data measures its own noise floor

Item 38's first recall gate built 20k random unit vectors, ran the binary
prefilter against the exact scan, and got **recall@10 = 0.31**. That reads as a
broken prefilter. It was not:

- the angular gap between the 10th and the 160th nearest of a random dim-256
  corpus is ~0.051 rad;
- the Hamming estimator's own standard deviation at that dim is ~0.097 rad.

There is nothing there to resolve. Random high-dimensional vectors are all
nearly equidistant, so "the true top-10" is decided by differences smaller than
any approximate method's noise. **The corpus had no neighborhoods, so there were
no neighborhoods to find.**

The second version fixed the corpus and not the queries: it generated clustered
vectors, then drew the queries from a *second, independent* set of centers. Every
query was still a random direction with no near neighbors in the corpus, and
recall came back at 0.28 — the corpus's structure was there and entirely
unreachable. Only when the queries were drawn around the *same* centers did the
gate reach 0.94, and the real-corpus test 0.96.

**Guard:** an approximate-retrieval gate needs a corpus with neighborhoods AND
queries that land in them; the generator should produce both together so they
cannot drift apart. Sanity-check by computing what the exact top-k similarities
actually are — if the 1st and the 100th neighbor score about the same, the
benchmark cannot distinguish any two methods, including a correct one from a
broken one. Keep a real-checkpoint recall test alongside it: synthetic clusters
are isotropic and zero-mean, which happens to be the geometry binary
quantization likes best, and it hid the entire effect of mean-centering.

### 1.28 Benchmark the knob, not just its default

Item 38's quality knob (`overquery`, the prefilter's candidate multiplier)
looked free from the design: the scan reads the whole corpus either way, and
only the rerank grows. Sweeping it said otherwise — 594 µs at 4 and 2042 µs at
32, a **3.4× swing on a parameter that controls only recall**. The cost was not
in the rerank at all but in per-shard selection heaps sized by the candidate
count. Rewritten as a counting sort, the same sweep is 877 → 901 µs.

Two things follow. The obvious one: a knob users are told to raise for quality
must be measured across its range, not at its default, or the first person who
raises it pays a cost nobody priced. The less obvious one: **the sweep localized
the bug.** A flat-in-theory parameter that is steep in practice is pointing at
an implementation detail scaling with it, and that is a far sharper signal than
a single slow number.

It also inverts the usual order. The knob's price is what decides where the
default belongs — once raising it was nearly free, the default moved from where
the implementation stopped hurting (8) to where the recall curve actually
flattens (16).

---

### 1.29 A fixed low iteration count is not a short benchmark, it is a noisy one

Item 39's first A/B ran with `-benchtime 500x` and benchstat reported spreads of
**±44%, ±39% and ±32%** — on a box whose drift floor is ~5% (§1.6). The
exhaustive baseline, whose code had not changed at all, moved 75% between two
invocations. Nothing was wrong with the code; 500 iterations of a ~100 µs
operation is a 50 ms sample, short enough that GC timing and cache state
dominate it.

Re-run with `-benchtime 1s`, the same comparison came back at **±2–6%** and the
verdicts were stable.

**Guard:** use a time-based `-benchtime` unless there is a specific reason not
to. `Nx` is for making a slow benchmark finish, not for making a fast one
precise — and if the spread benchstat reports is far above §3's drift floor, do
not interpret the comparison, fix the sampling first.

### 1.30 The observable has to be the quantity you are saving

Item 39's pruning test measured how far each posting-list cursor had advanced,
which is the natural thing to reach for and is exactly wrong: a SKIP advances
the cursor index past the postings it declined to read. A query that pruned 95%
of its work left its cursors at the end of their lists, indistinguishable from
an exhaustive walk, and the test reported **0.0% skipped** on a change that was
in fact working. The real observable — documents evaluated — needed a counter,
and showed 2,171 of 47,732.

The same mistake has a general shape: the metric was a PROXY (cursor position)
for the quantity of interest (work done), and the optimization broke the
proportionality between them. That is not incidental — an optimization that
changes how work maps onto state is precisely the kind that invalidates proxy
metrics.

**Guard:** name the quantity the change is supposed to reduce, then check that
the instrument measures that quantity and not a correlate of it. If measuring it
honestly needs a counter in the shipping path, price the counter — one increment
per candidate against a candidate's own float work is free — rather than
accepting a proxy.

### 1.31 A stale baseline flatters the thing you are measuring

Item 39 measured WAND against `sparse`'s existing query path and found 37.5× on
one shape and 3.9× on another. Both were mirages: that path still carried a full
`slices.Sort` of the touched set, which **item 44 had removed from `bm25` two
items earlier** and which nobody had ported, because `sparse` had no benchmarks
to notice it. Fixing the baseline first (1.40–8.47×, and 1.70× on the shape that
matters) left WAND winning only on a query shape the package never sees, and the
decision flipped from ship to revert.

This is §1.18's rule seen from the other side. There the risk was an earlier item
spending a later one's win; here it was an *un*applied earlier item inflating a
later one's. Both come from the same place: a measurement is a comparison, and
half of it is the thing you are not thinking about.

**Guard:** before crediting a change with a large win, ask what the baseline is
carrying — particularly in a package that mirrors another one where a fix has
already landed. Sibling packages that were written as parallel implementations
drift apart exactly where one of them lacks the test or benchmark that would
have kept them in step.

### 1.32 A footprint claim names a quantity, and heap is often not it

Lens §4.5 predicted `LoadWeightsQ8` peaks near 690 MB to produce a 140 MB model.
Peak heap during the load measured **199.5 MiB — identical to its steady state**,
which reads as "the claim is wrong." Peak RSS in the same run measured **727.6
MiB**, which reads as "the claim is right." Both are correct: the checkpoint is
mmapped, so the f32 side is file-backed pages, and `HeapInuse` cannot see a page
that was never allocated by Go.

The instrument decided the verdict, and the two verdicts were opposite. Worse,
`B/op` would have shown nothing at all — no Go allocation happens for a
zero-copy tensor view — so the standard benchmark output is silent on the entire
finding.

The distinction is not academic. Clean file-backed pages are reclaimable under
pressure; anonymous heap is not. A 528 MiB spike of the first is a soft cost you
may choose to ignore, and of the second is an OOM. Reporting only "peak was 727
MiB" hides which one you have.

**Guard:** when a claim is about memory, say which memory before measuring —
allocated bytes, live heap, or resident set — and sample the one the mechanism
actually moves. If the mechanism is mmap, page cache or a syscall, heap
instruments are blind to it by construction and their silence is not evidence.
Report both when they disagree; the disagreement is the finding.

### 1.33 A fully-tested function can still allocate twice what it needs

`HNSW.MarshalBinary` had a round-trip test, a determinism test, a stability test,
a hostile-input test and an over-allocation guard test. It also allocated
**131.0 MB to produce a 58.3 MB blob** — the capacity estimate budgeted 8 bytes of
graph header per node where a node with L layers needs 4+4L, so `append` doubled
mid-write. That had been true since the function was written.

Every one of those tests asks *what bytes came out*. None asks *what it cost to
produce them*, and no benchmark existed for the function at all, so there was no
`B/op` column for anyone to notice. The defect was invisible to a green suite by
construction.

It surfaced only because lens §4.3 predicted a doubling and the A/B measured a
**tripling** — the "before" number was worse than the finding claimed. That is the
tell: when a baseline measures worse than a pessimistic prediction, the baseline
has a second bug in it, and it is worth finding before crediting the fix.

**Guard:** for any function whose cost is proportional to the data (serializers,
loaders, encoders), have at least one benchmark with `ReportAllocs`, and assert
the size estimate against the actual output where one exists —
`TestHNSW_WriteToMatchesMarshal` now checks `blobSize() == len(blob)`, which is
the single line that would have caught this years earlier. Correctness gates and
cost gates are different gates; passing the first says nothing about the second.

### 1.34 An absolute allocation bound is not portable across build modes

`TestQuery_scratchIsPooled` first asserted "≤6 allocations per `Query`", tuned to
an observed 2. It passed, and then **failed the `-race` suite**: under race
instrumentation the same pooled query allocates **10**, and the unpooled one 25.
Nothing was wrong with the code — the gate had baked in a constant that only holds
in one build mode.

Rewritten to measure *both* arms in the same process and compare them, it reads
9.5× normally and 2.8× under `-race`, and passes in both because the ratio is what
the change controls. The absolute count is a property of the toolchain; the ratio
is a property of the pool.

The same argument applies to GC settings, Go version and fixture size, all of
which move an absolute count and none of which move the ratio.

**Guard:** when gating a resource (allocations, bytes, syscalls), measure the
baseline in the same run rather than hard-coding it. If a second arm is genuinely
impossible, state the build mode the constant was measured in and expect to
revisit it. A gate that only holds under `go test` with no flags is a gate that
will fail someone else's CI for the wrong reason.

---

## 5 · Keeping this current

Add to this doc when:

- a measurement **misleads** you — that is a new §1 entry, with the numbers;
- a prediction lands or misses — that is a §4 row;
- a machine-level constant is measured — that is a §3 row;
- a technique **works** and is not obvious — put it in §2.

Prefer adding the specific number and the specific example over adding advice.
The entries above are useful because each one names what was believed, what was
measured, and by how much they differed.

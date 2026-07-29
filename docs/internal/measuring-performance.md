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
| row-split matmul worker cap | `(M+31)/32` → **3 workers at M=91** | §7.14 |

The split-CCX L3 is why the row-split matmul axis lost here and not on the M1
Pro: replicating the weight matrix per worker is far cheaper across a large
unified cache.

### `apple-m1pro` — Apple M1 Pro (8 core), macOS

`parallelThreshold`'s tuning table in `encoder/parallel.go` was measured here.
Treat every constant in it as unverified on amd64 until re-measured — §7.14 found
the *axis* itself was wrong on the 3700X.

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
| 13 · …on the D=384 cross-encoder | ~22% GELU + ~14% softmax | **−33.8%** | ✅ |
| 13 · …on SigLIP | up to 2× | 1.24× geomean (1.44× at 576 patches) | ❌ |
| 19 · `DequantizeRowInt4` | 2–4× | 4.93× | ✅ |

Two entries were **found by measurement rather than predicted** and are the
largest encoder wins of the campaign: the parallel axis being wrong on amd64
(§7.14) and the elementwise stages dominating once the matmuls were parallel
(§7.15, `GTE.Encode` L=690 3.28×).

Item 13's cross-encoder row is worth singling out as the counter-example to the
"magnitudes are optimistic" pattern: it came from a **closed form** (activation ÷
FFN-matmul = `t_act/(k·D·t_mac)`, which grows as D shrinks) rather than from a
microbenchmark extrapolation. Derived predictions have held; measured-then-scaled
ones have not.

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

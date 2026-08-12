# Roofline campaign, 2026-08 — linalg CPU and CUDA

> **Status:** RESULT of two sessions on `nvidia-rtx2070s` (Ryzen 7 3700X, Zen 2, 8C/16T,
> AVX2 no VNNI; RTX 2070 SUPER, driver 595.58.03). Every ratio below is against a
> ceiling **measured on the machine that ran it**, by a probe in the tree:
> `linalg/fmapeak_amd64.s` for the CPU roofs, `gpu/roofline.cu` for the device ones.
> Re-run the device three with `cd gpu && go test -run TestDeviceCeilings -v`.
>
> The companion docs are [`measuring-performance.md`](measuring-performance.md) (method
> and failure catalogue), [`perf-amdahl-linux-amd64.md`](perf-amdahl-linux-amd64.md)
> (stage weights) and [`cpu-acceleration.md`](cpu-acceleration.md) (the arm64 side).

## The one-line version

The f32 GEMM went from **39% to 52–64%** of this box's FMA peak, the CPU int8 W8A8
matmul gained **17.4%**, `FlatI8`'s CUDA `Query` got **3.3–6.0×** and its `QueryBatch`
**4.5–4.6×** end to end. Every win came from measuring a ceiling first; **every prediction made
from reading an instruction mix was wrong — four for four.**

## 0 · Why the ceilings had to be built first

`BenchmarkGEMMPeakFraction` has no build tag, so it has always run on amd64 — against
a hardcoded M1 Pro constant (3.2 GHz × 16 f32-FMA/cyc = 102.4 GFLOPS). On this box it
reported the GEMM at **"~50 %peak"** when the measured figure is **38%**. Nothing was
wrong with the benchmark's arithmetic; the denominator was from another machine.

That is the whole reason this campaign starts with probes rather than kernels. A wrong
denominator does not announce itself — it produces a plausible number, and a plausible
wrong number is harder to catch than an obviously wrong one.

| ceiling | measured | how |
|---|--:|---|
| f32 FMA peak (amd64) | **135 GFLOP/s** | `fmapeak_amd64.s`, 14 YMM accumulators, no memory |
| VPMADDWD throughput (int8) | **69.0 GMAC/s** | load-free probe, 1.03/cycle — single-pipe on Zen 2 |
| device streaming read | **412 GB/s** | `int4` grid-stride over 512 MiB, 92% of the 448 nameplate |
| device int32 mul-add | **4876 GMAC/s** | 16 independent `a=a*b+c` chains, no memory |
| device `__dp4a` | **19534 GMAC/s** | same loop, 4 int8 MACs per instruction |

Each probe cross-checks itself, which is the part that makes a denominator
trustworthy rather than merely plausible:

- 135 GFLOP/s ÷ 32 flops/cycle implies **4.2 GHz**, exactly where a 3700X boosts.
- 4876 GMAC/s ÷ (40 SM × 64 lanes) implies **1.90 GHz**, inside the 2070 SUPER's
  1.77–2.15 GHz boost range.
- `dp4a_peak` and `imad_peak` run the *same* loop with the *same* chain count and
  differ only in the instruction issued, so their **instruction** rates must match —
  4883 vs 4876 G/s. If they had not, dp4a's 4× MAC advantage would have been an
  artifact of loop overhead rather than a property of the ISA. `TestDeviceCeilings`
  asserts this ratio, so the probe fails loudly rather than reporting a pretty lie.

A figure implying 8 GHz, or a dp4a/imad ratio of 4, would mean the probe was measuring
something other than what it claims.

## 1 · CPU f32: 39% → 52–64% of peak

| shape | before | after | | % of 135 |
|---|--:|--:|--:|--:|
| M8_K768_N768 | 53.1 | 74.5 | +40% | 55% |
| M32_K768_N768 | 54.3 | 86.4 | +59% | 64% |
| M128_K768_N768 | 54.4 | 84.8 | +56% | 63% |
| M512_K384_N1536 | 53.9 | 69.8 | +30% | 52% |

Two commits: `a0e6299` removed a `[32]float32` shim, `555483c` added the 3×4 kernel.

**The shim first, because it is the smaller and stranger one.** On amd64 `dotFMA8`
already finishes its horizontal reduction in-register and writes eight final scalars —
then the wrapper zeroed a 128-byte `[32]float32`, scattered those eight to indices
0, 4, 8 … 28, and the caller re-read all 32 and added, **24 of those adds being
`x + 0.0`**. Pure interface tax from an arm64-shaped signature, where the four lanes
per column are genuine partial sums. Worth ~4%.

**Then the kernel, where the stated mechanism was wrong.** `dotFMA8` takes its `b`
operands from memory, so every FMA carries its own load: 0.89 FMA/load, which two load
ports should cap at half the FMA rate. So a 2×4 kernel was built holding `b` in a
register — 1.33 FMA/load, a third fewer loads, same 8 accumulators. **It gained 4%.**

Same chain count, a third fewer loads, no gain ⇒ **loads were never binding**. The
load-free probe settles it by varying only the accumulator count:

```
 8 chains → 108.5 GFLOPS   (exactly 8/10 of peak)
10 chains → 135.6          (full peak)
12 chains → 135.0
14 chains → 135.4
```

Zen 2 issues 2 FMAs/cycle at ~5 cycles latency, so **≥10 independent chains** are
needed before the pipes, rather than the dependency chain, set the pace. `dotFMA8` has
8. So did the 2×4. 3×4 is the first tile that clears 10 (12 chains) inside AVX2's 16
registers — 12 accumulators + 3 a-vectors + 1 rotating b.

**arm64's 2×8 shape cannot be ported.** 16 accumulators + 2 a + 1 b = 19 registers;
NEON has 32, AVX2 has 16. The two backends now differ in kernel *shape*, not just ISA.

Bit-identity is by construction — row blocking changes which dot products are computed
together, never the arithmetic within one — and gated on exact equality across odd `n4`
(where `n%8==4` exercises the 4-element tail, the path with a real historical
VEX-zeroing bug). Mutation-checked both ways.

## 2 · CPU int8: W8A8 matmul +17.4%

`w8a8Span` scored one (row, column) pair per call, re-widening the same a-row for every
column. `dotI8x8AVX2` widens it once per group of eight: 25 ops per 128 MACs against 32.

| | before | after | |
|---|--:|--:|--:|
| kernel, 8 cols @ K=768 | 39.1 | 53.9 GMAC/s | +37.8% |
| `MatmulBTW8A8` | 113.0 µs | 96.2 µs | +17.4% |

53.9 is **78%** of the 69.0 GMAC/s VPMADDWD ceiling; the 1×1 form was at 63%.

**`VPMADDUBSW` was built, verified exact, and rejected.** Half of `dotI8AVX2`'s vector
uops are `VPMOVSXBW` widening, and `VPMADDUBSW` with `VPABSB`/`VPSIGNB` sign-folding
removes them. It measured **+8.5%** — the third time instruction-mix reasoning
overpromised. It also carries a precondition: the bound needs |a|,|b| ≤ 127
(2·127² = 32258 < 32767), but **−128 gives 2·128² = 32768 and saturates silently by
one**. Every in-tree quantizer clamps to ±127 (`quant.go:59-62`, `158-161`), but
`DotI8` is exported and general, so relying on that needs an enforced invariant.
Register blocking was worth 4.4× more with no precondition at all.

That difference is visible in the gate: `TestDotI8x8_matchesScalar` sweeps the **full
int8 range including −128**, which the `VPMADDUBSW` route could not have survived.

**No bit-identity risk here, unlike f32.** Integer addition is associative and
int8×int8→int32 cannot overflow for any K this library sees (|Σ| ≤ K·127², so K would
need to exceed 133,000). So the gate compares against **scalar**, testing what is
actually true rather than an accident of matching instruction order.

## 3 · CUDA: `FlatI8.Query` 3.3–6.0×

| shape | before | after | % of 403 GB/s | |
|---|--:|--:|--:|--:|
| N=200k K=256 | 25.4 | 82.8 GB/s | 6% → 21% | 3.3× |
| N=200k K=768 | 26.7 | 148.6 | 7% → 37% | 5.6× |
| N=500k K=768 | 27.8 | 165.4 | 7% → 41% | 6.0× |

At N=500k/K=768 that is **13.8 ms → 2.32 ms per query**.

**The cause was in the kernel's first line.** One thread per output row means adjacent
threads read addresses K bytes apart — 768 at the real shape — so a 32-lane warp's load
touches 32 cache lines and uses **one byte from each**. 1/32 = 3%; caching lifted it to
the observed 7%. An access-pattern defect, not a tuning shortfall, which is why it was
worth multiples rather than percent.

Now one **warp** per row: the lanes walk the same row so a load covers 32 consecutive
addresses, then `__shfl_down_sync` folds them. Where `k%4==0` (256/384/768 — every real
shape) it loads 4 bytes per lane and uses `__dp4a`.

**The file said this was deliberate**, and that was worth taking seriously: *"no
`__dp4a`, no warp-per-row… that blob is goinfer's and stays there; this is the proving
kernel."* The scoping argument is sound. What retired it is that the kernel is wired to
`FlatI8.EnableGPU()`, a production entry point, where it was **slower than the CPU path
it exists to accelerate**. A proving kernel is fine; a proving kernel on a production
entry point is a pessimization.

Safe to reshape because it is `BIT-IDENTITY-EXEMPT` for a *structural* reason — `int`
accumulation, associative — so a warp reduction is exactly the sequential sum. Parity
confirms rather than assumes: GPU≡CPU top-10 over 200 queries at worst Δ **0.000e+00**.

`71269fe` then routed **M=1 batches** to the same kernel, since `gemmTileMinM = 2` means
the naive thread-per-(query,row) GEMM only ever ran at that one width.

## 3b · CUDA: the batch path, `QueryBatch` 2.2–5.4×

This was §6's open item — "the only unmeasured production path on this device, tuned
before any ceiling existed to tune against". It was worse than unmeasured.

`FlatI8.QueryBatch` ran `vit.cu`'s 16×16 `gemm_w8a8_tiled`. Kernel time at N=200k,
K=768:

| M | tiled GEMM | looped GEMV | **new batch GEMV** | vs tiled |
|--:|--:|--:|--:|--:|
| 2 | 6.79 ms | 0.96 | **0.55** | 12.4× |
| 8 | 6.82 | 3.75 | **0.66** | 10.3× |
| 16 | 6.88 | 7.48 | **1.08** | 6.4× |
| 64 | 27.36 | ~29.9 | **3.94** | 7.0× |
| 256 | 109.28 | ~119.5 | **15.51** | 7.0× |

**The tiled kernel's wall time did not depend on M at all below 16.** It tiles a 16×16
output block, so a batch of 2 pays for 16 rows and discards 14. `gemmTileMinM = 2`
routed everything from M=2 up into it.

That constant was not wrong when it was written — it was **invalidated by §3**. It was
calibrated against a thread-per-row GEMV that was 5.6× slower than the one that ships
now. Re-measured against the current baseline, the tiled kernel does not break even
until M ≈ 15 and never beats a loop over the single-query GEMV by more than 9%. A
tuning constant is a claim about two things, and fixing one of them silently retires it.

End-to-end through `QueryBatch`, which is what a caller feels:

| M | before | after | | per query |
|--:|--:|--:|--:|--:|
| 2 | 9.19 ms | 1.69 ms | 5.4× | 4.59 → 0.85 ms |
| 16 | 10.25 | 4.65 | 2.2× | 0.64 → 0.29 |
| 64 | 31.21 | 9.13 | 3.4× | 0.49 → 0.14 |
| 256 | 121.02 | 26.54 | 4.6× | 0.47 → 0.10 |

**Batching now pays at every width.** Before, M=2 cost 4.59 ms per query against M=1's
1.22 — asking for two answers at once was 3.8× *worse* per answer than asking twice.

The replacement is a lane-group GEMV: LANES threads cooperate on one corpus row (as
§3's kernel does), but QTILE queries are staged in shared memory so **one corpus load
feeds QTILE `__dp4a`s**. Corpus traffic falls from once-per-query to once-per-tile.

**The two parameters were swept, not reasoned — and reasoning would have got it wrong
for the fourth time.** The obvious port of §3's kernel keeps 32 lanes per row: that is
the slowest row of the sweep, 2.2× off the best, because the reduction cost scales with
QTILE × log2(LANES) and a wider query tile therefore wants *fewer* lanes. Predicted
~0.5 ms at M=16 from a bandwidth argument; measured 2.78 ms, because the kernel is not
bandwidth-bound at all — it reaches 110 GB/s of a 412 GB/s roof. The sweep table is in
`gemv_w8a8.cu`.

`gemm_w8a8_tiled` is no longer reachable from this backend at any M, and the naive
`gemm_w8a8` — dead since `71269fe` rerouted M=1 — is deleted.

## 3c · CUDA: `topk_rows` 3.2–3.4×, and flat in k

The device top-k made **k passes** over the score row. Its own comment said the k passes
were "cheap next to the GEMM that produced the row" — true when written, and retired by
§3b. Phase-split with production buffers at N=200k:

| | gemv | topk (k-pass) | |
|---|--:|--:|--:|
| M=8, k=10 | 0.88 ms | **2.57** | topk 2.9× the GEMV |
| M=256, k=10 | 18.2 | 7.46 | 0.4× |
| M=256, k=50 | 18.2 | **36.0** | 2.0× |

The GEMV's cost does not move with k; this one's was linear in it. Replaced with one
pass — per-thread candidates in registers, then a block merge over 256 heads per round
instead of over N. **0.77 / 0.92 / 2.16 ms** at k=10 (3.2–3.4×), and flat in k.

**The cost is linear in the per-thread width and flat in k — the reverse of what it
replaced, and not what I predicted.** The scan rejects nearly every element with one
compare, but an *accepted* candidate costs a width-long bubble, and accepts are not rare:
a thread scanning S elements accepts about `TKREG·ln(S)` of them. So four instantiations
routed by k: at M=256, width 8 runs at **233 GB/s (57% of the roof)**, 16 at 95, 32 at
42, 64 at 11. One widest kernel would have been 2.3× slower at k=10 — the repo's default.

`float4` loads help **only at width 8**, where the scan is near memory-bound; at 16 and
32 they cost 15–25%, adding pressure to a kernel already limited by registers.

## 3c (Metal) · `topk_rows` 2.1×, and where the M1 wall is different

The M1 mirror had the **same disease in a different organ**. It never made k passes — it
was already one-pass, each thread keeping a register top-k over its strip — but then a
single thread merged all `TOPK_TG·k` partial candidates alone, k times, while the other
127 idled. Isolated against this box's **187 GB/s** streaming ceiling (a new
`TestMetalTopKBandwidth`, GPU-timestamped, min-of-8) it ran at **1.5–8.3%** of it.

Two changes, both measured:

- **Parallel tree merge.** Each thread sorts its local top-k, then the `TOPK_TG` lists
  fold pairwise in log₂(`TOPK_TG`) O(k) two-pointer steps — a block merge over 128 heads
  per round, not one thread over all of them. Same total-order (score desc, index asc)
  top-k regardless of merge shape, so parity is by construction. ~1.5–2.2×.
- **Scalar worst-cache** in the strip-scan, so the reject path — nearly every element — is
  a register-only compare that never touches the dynamically-indexed, local-memory-backed
  candidate array. ~10% on top.

M=256/N=200k **24.2 → 11.5 ms**, N=500k **32.8 → 18.7 ms** (2.1× / 1.75×).

**But the M1's remaining wall is occupancy, not k.** The CUDA side's width-routing win does
not transfer at the shipped k=10: this kernel's accept loop is already k-bounded, not
width-bounded, so the register array is the minimal width-16 tier either way (routing would
only help k<8). What is left is different: at M=256 the kernel reaches **9.5% of the roof
against CUDA's 23% at width-16**, and a `TOPK_TG` sweep shows **64 beats 128 at large M**
(more threadgroups resident) **but loses up to 46% at small M** — the limiter is the 16 KB
of threadgroup memory the staged lists occupy, capping residency to ~2 groups/core. So 128
stays the balanced point; closing the gap is a staging redesign, and per-M dispatch (cf.
`batchPlan`) is the cheaper next lever. Same trap as §4's stale-gate entry surfaced here
too: `TestMetalGEMM_batchParityWithCPU` runs at n=3000 < `topkMinN`, so it exercises the
host fallback and never reached the device kernel — a new `TestMetalTopK_randomParityWithCPU`
at n≥`topkMinN` gates the merge on random scores, worst Δ **0.000e+00**.

## 3d · Crossover, regenerated on both boxes

Neither box's recorded crossover described its own code any more, and neither harness
timed single-query `Query` at all — only `QueryBatch`, a different kernel. Both now do.

| | CUDA (RTX 2070S) | Metal (M1 Pro) |
|---|--:|--:|
| Query(1), N=100k | **2.28×** | 0.65× |
| Query(1), N=10k | 1.02× | 0.42× |
| batch 8, N=100k | 6.44× | 0.56× |
| batch 256, N=100k | **36.50×** | 1.50× |

The Metal column is post-§3c(Metal): its `QueryBatch` numbers moved up with the topk_rows
rewrite (batch 256 at N=100k 1.36 → 1.50×), while the single-query rows sit on the
unchanged `gemv_w8a8` path and drift only with run-to-run variance.

**The two backends now disagree about when `EnableGPU()` pays.** On CUDA a single query
is worth sending to the GPU at N=100k; on the M1 Pro it is not worth sending at any
measured size, and batching only pays from 64. A single "use the GPU above X" rule would
be wrong on one of them.

One reading trap worth recording: at N=10k the CUDA batch speedup **peaks at 8 and
declines** (4.06 → 3.70 → 3.20), because the CPU baseline itself jumps at batch 64 while
the GPU is already near its floor. Reading only the largest batch would have suggested
the advantage grows monotonically with batch size. It does not, at small N.

## 4 · Negatives and process failures, in full

- **2×4 AVX2 kernel** — 1.33 FMA/load vs 0.89, +4%, not kept. Its disproof is recorded
  in `dot3x4_amd64.s` because "we tried fewer loads and it did nothing" is what makes
  the chain-count explanation credible rather than merely asserted.
- **`VPMADDUBSW` int8** — +8.5%, exact, rejected on the −128 precondition (§2).
- **A contention test that asserted nothing.** Guarding `measureFMAPeak`'s max-of-N
  estimator by measuring under load: 8 loader goroutines *did* discriminate and
  destabilised `embed`'s `TestEncodeBatch_speedup`, which runs concurrently in another
  package — a test that saturates the machine fails its neighbours. 2 loaders left
  neighbours alone and asserted **nothing**: mutating max→average still read 136.4
  GFLOPS. Both requirements conflict on a shared machine, so no contention test ships;
  the reasoning lives at `measureFMAPeak` and the implied-clock check is the real gate.
- **A "verified by mutation" comment that had not been mutation-checked.** Written into
  the 2-loader test before testing it. Caught, and the claim removed.
- **Three gates found exercising a different path than the change.**
  `TestCUDAGEMM_batchParityWithCPU` runs at M=16, which was ≥ `gemmTileMinM` and so
  exercised the *tiled* kernel — the one path the M=1 reroute did not touch.
- **A stale tuning constant that no test could ever have caught.** `gemmTileMinM = 2`
  selected a kernel that was 7–12× slower than the alternative, and every parity gate
  stayed green throughout, correctly: both kernels compute the same scores. Nothing in
  the suite measures *which* kernel is faster, so the only thing standing between a
  correct-but-slow route and production was someone re-measuring. Worth remembering
  before trusting any other constant of this shape.
- **A mutation that was RIGHT to survive.** Pairing the wide batch kernel with the
  narrow one's lane count passed every parity test — and should have: over-launching is
  absorbed by the row-bound check, at 2× the blocks for the same answer. The reverse
  pairing silently drops the tail of the corpus and still returns a full, plausible
  top-k. Neither is reachable by a correctness test, so the pairing was extracted into
  `batchPlan` and gated as a pure function.

## 5 · The through-line

**Every prediction from instruction mix was wrong; every ceiling measured in isolation
was right.**

- f32: "load ports bind" → built the fix → 4%. The binding constraint was accumulator
  latency, found by varying only chain count.
- int8: "half the uops are widening" → built the fix → 8.5%. The lever was register
  blocking, worth 4.4× more.
- CUDA `Query`: the roofline pointed straight at coalescing and was right first time —
  **because a bandwidth-bound kernel has exactly one thing to get right.**
- CUDA `QueryBatch`: "stage the queries and it becomes bandwidth-bound" → built it →
  2.78 ms against a predicted 0.5, at 27% of the bandwidth roof. The lever was the
  reduction width, found by sweeping two parameters and reading the table.

The corollary for whoever picks this up: the probes are cheap (each is a few dozen lines
and runs in under a second) and they are the only step that has never misled. Build the
denominator before the kernel.

## 6 · Open, and what needs the M1

**Unmeasured on this box:**

- **The top-k still runs ONE BLOCK PER QUERY.** At M=8 that leaves 8 of 40 SMs busy, and
  it is the remaining limit at small batches (§3c). Splitting N across several blocks per
  query needs a second merge pass; not attempted.
- **k > 64 still takes the k-pass kernel**, at 36 ms where the register form would be
  ~19. A 128-wide instantiation would spill; a radix-select would not, and is the
  principled answer if large-k retrieval ever matters.
- **The single-query GEMV has 22% left, already measured.** The §3b sweep covered
  QTILE=1, and 8 lanes per row beats the shipped 32 at M=1: 0.401 ms against 0.490 at
  N=200k. `gemv_w8a8` is untouched here because it is a separate path with its own
  gates; the change is a constant.
- CPU f32 remains at ~63% and int8 at 78% of their ceilings. The rest is loop overhead,
  the 12 horizontal reductions, and cache — diminishing returns.

**For the M1 Pro:**

- `gpu/annmetal`'s `gemv_w8a8` has the **identical** thread-per-row shape
  (`uint j [[thread_position_in_grid]]`). Prompt sent:
  `~/claude-mailbox/to-cowork/2026-08-12-linux-annmetal-gemv-coalescing.md`. It asks for
  the bandwidth ceiling **first**, because unified memory behaves nothing like GDDR6 and
  the CUDA result may not transfer.
- Note the asymmetry: Metal's `gemm_w8a8_tiled` *is* tiled where CUDA's `gemm_w8a8` is
  naive. **The two backends optimized opposite halves.**
- `BenchmarkGEMMPeakFraction` now measures its own denominator per arch, so the M1
  numbers in `cpu-acceleration.md` (~95 GFLOPS ceiling, 68–73%) can be re-derived rather
  than trusted.

**Crossover, one side done.** The M1 Pro's `crossover_test.go` was regenerated
(`ab803ab`) after the Metal fix, and gained a single-query row the old harness lacked —
it only ever timed `QueryBatch`, so it could not answer "is `EnableGPU()` worth it for
one query". Its finding: on the M1 Pro single-query GPU still LOSES to CPU (0.43–0.66×)
and `EnableGPU()` pays only at batch ≥ 8.

**Both are now current** (§3d). One inconsistency is left standing deliberately:
`crossover-metal.jsonl` labels the M1 Pro "apple" while `vit-metal.jsonl` labels the same
box "apple-m1pro", so the generated report renders one machine as two and says
"3 machine(s)". Those records belong to the other box, which was mid-flight; flagged
there rather than edited across.

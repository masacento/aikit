# Roofline campaign, 2026-08 — linalg CPU and CUDA

> **Status:** RESULT of one session on `nvidia-rtx2070s` (Ryzen 7 3700X, Zen 2, 8C/16T,
> AVX2 no VNNI; RTX 2070 SUPER, driver 595.58.03). Five commits, `8073a57`…`71269fe`.
> Every ratio below is against a ceiling **measured on the machine that ran it**, and
> the probes that produce those ceilings are now in the tree.
>
> The companion docs are [`measuring-performance.md`](measuring-performance.md) (method
> and failure catalogue), [`perf-amdahl-linux-amd64.md`](perf-amdahl-linux-amd64.md)
> (stage weights) and [`cpu-acceleration.md`](cpu-acceleration.md) (the arm64 side).

## The one-line version

The f32 GEMM went from **39% to 52–64%** of this box's FMA peak, the CPU int8 W8A8
matmul gained **17.4%**, and `FlatI8`'s CUDA `Query` got **3.3–6.0×**. Every win came
from measuring a ceiling first; **every prediction made from reading the instruction
mix was wrong.**

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
| device streaming read | **403 GB/s** | `int4` grid-stride over 512 MiB, 90% of the 448 nameplate |

The f32 probe cross-checks itself: 135 GFLOP/s ÷ 32 flops/cycle implies **4.2 GHz**,
exactly where a 3700X boosts. A figure implying 8 GHz or 1 GHz would mean it was
measuring something other than FMA throughput.

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
- **Three gates found exercising a different path than the change.** Most recently
  `TestCUDAGEMM_batchParityWithCPU` runs at M=16, which is ≥ `gemmTileMinM` and so
  exercises the *tiled* kernel — the one path the M=1 reroute does not touch.

## 5 · The through-line

**Every prediction from instruction mix was wrong; every ceiling measured in isolation
was right.**

- f32: "load ports bind" → built the fix → 4%. The binding constraint was accumulator
  latency, found by varying only chain count.
- int8: "half the uops are widening" → built the fix → 8.5%. The lever was register
  blocking, worth 4.4× more.
- CUDA: the roofline pointed straight at coalescing and was right first time —
  **because a bandwidth-bound kernel has exactly one thing to get right.**

The corollary for whoever picks this up: the probes are cheap (each is a few dozen lines
and runs in under a second) and they are the only step that has never misled. Build the
denominator before the kernel.

## 6 · Open, and what needs the M1

**Unmeasured on this box:**

- The **tiled batch GEMM** (`vit.cu`'s `gemm_w8a8`, used for `QueryBatch` M≥2) has never
  been measured against the 403 GB/s ceiling. It is now the only unmeasured production
  path on this device, and it was tuned before any ceiling existed to tune against.
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

**Stale on both sides:** the CPU int8 path gained 17–38% and the CUDA path 3.3–6.0×, so
`crossover_test.go`'s recorded CPU/GPU crossover points describe neither. Regenerating
them is arguably worth more than another kernel — it is what decides when
`EnableGPU()` is worth calling at all.

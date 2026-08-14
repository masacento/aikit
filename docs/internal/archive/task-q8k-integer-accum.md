# task: Q8_K integer-accumulation matmul kernel (Q4_K/Q6_K × Q8_K)

> Status: **CLOSED — negative result (2026-07-26).** The scalar + SDOT reference kernels were
> built, validated bit-exact, and benchmarked; the ≥1.3× gate (§5.4) is **provably unreachable for
> Q6_K** (byte-ratio ceiling 256/210 = 1.22× < 1.3×, regardless of unpack quality) and below
> cpubrrr's real-world Q4_K precedent (~1.12×). Write-up + numbers in
> `docs/internal/cpu-acceleration.md` ("Native K-quant matmul — evaluated, NOT shipped"). Code kept
> in `linalg/kquant*.go` (Experimental, tested, NOT wired into `WeightMat`). The cpubrrr win to chase
> is MXFP4 MoE (goinfer A2), not this. Original draft below.
>
> Drafted 2026-07-26 from a review of
> [`arizqi/cpubrrr`](https://github.com/arizqi/cpubrrr).
>
> **Read `docs/internal/task-native-q6k-kernel.md` first.** This task revives that
> DEFERRED spike on a *different axis*. That doc deferred the fused K-quant kernel
> because the fidelity-and-footprint case against int4-requant did not carry — a
> conclusion this task does not dispute. What it sets aside in §5 is the **compute**
> case, and cpubrrr is the evidence that the compute case carries on its own.
> Its §4 architecture notes are still valid and pick-up-ready.

---

## 1. The finding

cpubrrr is a CPU-only Apple-silicon inference runtime, ~5.5k lines of Rust, one
dependency (`memmap2`). On Qwen3-Coder-30B (Q4_K/Q6_K) it went from **losing** to
llama.cpp's CPU path (~71 vs ~86 tok/s) to **winning** (~92 vs ~82) with a single
algorithmic change, which its README describes plainly:

> quantize activations to Q8_K, accumulate sub-block integer dot products weighted by
> the 6-bit scales in int32, and convert to float *once per 256-value superblock*.

They found it by reading llama.cpp's ARM source rather than micro-optimizing their own
kernel. It is llama.cpp's own algorithm (`ggml_vec_dot_q4_K_q8_K` and friends); the win
came from adopting it and then out-scheduling the surrounding execution.

**Why this matters for aikit:** the deferred Q6_K task framed the kernel as a way to
avoid dequant→requant at load. That framing undersells it. The real prize is that the
inner loop never leaves the integer domain — no per-sub-block float multiply, no f32
accumulator — so it maps onto SDOT and integer add throughput instead of the FP pipeline.

## 2. What already exists here

`linalg` is closer to this than the deferred doc implies:

- `dot_i8dp_arm64.s` — SDOT-based int8 dot. **The SDOT half is already built.**
- `dot_i8_arm64.s` — the non-dotprod fallback for cores without the extension.
- `dotprod_arm64_{darwin,linux,other}.go` — runtime feature detection, already wired.
- `quant_i8_{arm64,amd64,other}.go`, `quant_w4a8*` — activation quantization machinery.
- `dot2x8_arm64.s`, `matmul_blocked.go` — the multi-row blocking the kernel plugs into.
- `weightmat.go` — the `WeightMat` abstraction where a new block kind belongs.
- `embed/gguf.go` — GGUF parsing, so the block formats are already reachable.

**The delta is not SDOT.** It is: superblock iteration, per-sub-block int32 scale
weighting, deferred float conversion, and the Q4_K min/offset correction.

## 3. Algorithm (implement exactly this)

### Q6_K × Q8_K
Superblock = 256 weights: 16 sub-blocks × 16 values, 8-bit sub-block scales, one f16 `d`.

```
for each superblock:
    acc_i32 = 0
    for j in 0..16:
        p_j = sdot(w_sub[j], a_sub[j])       # int8 × int8 -> int32, 16 values
        acc_i32 += scale[j] * p_j            # STAYS INTEGER
    out_f32 += (d_w * d_a) * (f32) acc_i32   # ONE float convert per 256
```

### Q4_K × Q8_K
Superblock = 256: 8 sub-blocks × 32, 6-bit scales **and 6-bit mins**, f16 `d` + `dmin`.
Same shape plus the offset term:

```
    out_f32 += d_w * d_a * (f32) acc_i32
             - dmin * sum_j( min[j] * bsums[j] )
```

`bsums[j]` = the per-sub-block sum of the quantized activations, produced by the Q8_K
activation quantizer. **Compute bsums during quantization, not in the kernel** — it is
free there and a second pass otherwise.

### Q8_K activation quantizer (new)
Blocks of 256 with a single f32 scale, int8 values, plus the per-sub-block `bsums`.
Model it on `quant_i8_arm64.go`; it is a sibling, not a replacement.

## 4. Weight layout: quad-interleaving

cpubrrr's single biggest MXFP4 win was **not** a kernel change — it was reordering the
weight bytes so that each of N worker cores reads **one sequential stream** rather than
striding through a shared buffer. Their words: "a byte-reordering, not a code change."

Decode is memory-bandwidth-bound, so this is often worth more than the arithmetic.

Scope it here as an **optional `WeightMat` layout variant**, gated behind a constructor
(`NewWeightMatInterleaved(nWorkers)` or a field on the existing config) so the default
layout is untouched. Requires:

- Interleave at prequant/build time, never at load.
- The kernel must know the stride; make it a property of the `WeightMat`, not a global.
- **Correctness must be layout-independent** — the same parity test must pass in both
  layouts, byte-identical output.

Land the kernel (§3) first and measure it. Add interleaving as a second, separately
measured stage. Two changes at once and you will not know which one paid.

## 5. Acceptance criteria

**Correctness (blocking):**
1. Bit-exact against a scalar dequant-then-f32-dot reference over the full block-format
   space, including edge cases: partial superblocks, all-zero sub-blocks, scale extremes,
   negative mins. cpubrrr verified their kernels bit-exact before trusting a single
   benchmark; do the same, and do it before measuring anything.
2. Property test over random weights/activations, both layouts, identical output.
3. Existing `linalg` parity and shapecheck tests stay green. `checks_on.go` build passes.

**Throughput (the gate):**
4. Q6_K×Q8_K GEMV beats the current dequant→int8-requant→W8A8 path on arm64 by **≥1.3×**
   at decode shapes (M=1). Below that, stop and write up the negative result — the
   deferred doc's caution was well-founded and this would confirm it on the last axis.
5. No regression on amd64. The scalar fallback must be correct; it need not be fast in
   this task.

**Hygiene:**
6. Everything new lands in the **Experimental tier**. Do not touch the frozen Hard-tier
   surface. Update the Experimental list in `README.md`.
7. `CHANGELOG.md` entry.
8. Benchmark numbers into `docs/internal/cpu-acceleration.md` with method and hardware,
   matching the provenance style of the existing tables.

## 6. Explicit non-goals

- Q5_K, Q3_K, Q2_K. Q6_K and Q4_K only.
- x86 AVX-512/VNNI port. cpubrrr lists it as *their* roadmap item too; separate task.
- Changing the default `WeightMat` layout.
- Anything about fidelity vs int4. That question was answered in the deferred doc; do not
  relitigate it here.

## 7. First move

Read `docs/internal/task-native-q6k-kernel.md` §4 (architecture, `WeightMat` seam,
`linalg`/`embed` sibling rule) and `docs/internal/cpu-acceleration.md`. Then write the
Q8_K activation quantizer + the scalar reference kernel + the bit-exactness test, and get
those green **before** writing a line of assembly.

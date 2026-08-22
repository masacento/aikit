# CPU & SIMD acceleration (internal notes)

How aikit's pure-Go compute is accelerated, where it lives, how to test it, and
the open micro-kernel follow-ups. Internal/maintainer notes — the user-facing
story is the README + godoc.

> **GPU is goinfer's now.** The WebGPU backend (`encoder/gpu`) was removed from
> aikit in the v0.4.0 split — it carries the cgo `webgpu` dependency, which the
> core deliberately excludes. GPU matmul lives in `goinfer/gpu` behind the
> `encoder.Backend` seam; aikit ships only the pure-Go CPU backend. This doc is
> CPU/SIMD only.

---

## Two layers

**1. `linalg/` — the shared SIMD kernels (public package).** The single home for
the hand-written assembly, used by both `encoder` and goinfer's decoder. Dispatch
by build tag + runtime CPU detection:

| Arch  | Files | Kernel |
|-------|-------|--------|
| arm64 | `linalg/dot_arm64.{go,s}`, `dot_i8*_arm64.s`, `dot_w4a8_arm64.s`, `dotprod_arm64_*.go` | NEON; int8 `dotI8` upgrades to `SDOT` on DotProd-capable CPUs (runtime HWCAP); `dotW4A8GroupsSDOT` is the fused int4×int8 decode kernel (nibble-unpack prologue + the `dot_i8dp` SDOT body) |
| amd64 | `linalg/dot_amd64.{go,s}`, `dot_w4a8_amd64.s`, `quant_w4a8_amd64.go` | AVX2+FMA (`dotFMA`/`dotFMA4`/`dotFMA8`), int8 `dotI8AVX2` (VPMOVSXBW+VPMADDWD), and `dotW4A8GroupsAVX2` — the fused int4×int8 decode kernel (nibble-unpack prologue + the `dotI8AVX2` sign-extend body); runtime CPUID/XGETBV detect, scalar fallback |
| other | `linalg/dot_generic.go`, `dot_other.go` | portable scalar |

On top of the dot kernels, `linalg` provides:
- `Dot`, `Dot4x4`, `Dot8x4`, `Dot2x8`, `MatmulBT` (f32 column-parallel),
  `MatmulBTInto` (serial). **`MatmulBT`/`MatmulBTInto` are cache + register blocked**
  (`matmul_blocked.go`: 32×32×768 tiles over the Dot8x4/Dot2x8 kernels) above an
  M·K·N threshold; below it they keep the naive dot-per-output span (small matmuls
  like attention QKᵀ don't want the tiling prologue). This blocked GEMM is the single
  shared home — the encoder's transformer matmuls and other kit consumers route
  through it (it was hoisted out of the encoder once the un-blocked `MatmulBT`, which
  re-streamed `b` per a-row, measured ~7% of peak at prefill shapes). `Dot2x8` (arm64
  NEON) is the MR×NR register kernel inside it: 2 a-rows × 8 b-rows, 16 accumulators
  held across the K loop so each b-load feeds 2 FMLAs — vs `Dot8x4`'s 1×8, which was
  load- and latency-bound (≈40% of the *measured* 95.4 GFLOPS M1-Pro f32 ceiling;
  `BenchmarkGEMMPeakFraction` + the `fmaPeakARM64` ceiling probe gate this). It
  computes each dot in `Dot8x4`'s accumulation order (bit-identical), so the blocked
  GEMM differs from the naive span only by f32 reassociation; `MatmulBTAcc64` stays
  f64-exact. Column shards are 8-aligned, so `SetParallelWidth` stays numerically inert.
  At **K≥2048** the blocked path (arm64) first **packs** each 8-row b-group into a
  contiguous low-stride buffer (`packedFill`): at large K the simultaneously-read b-rows
  are K·4 bytes apart and collide in L1 cache sets, so packing them ~kBlock apart kills
  the conflicts (prefill 46%→69%, K=3072 fc2 +15%) — bit-identical (same values, same
  order), via a pooled buffer. K=768 dims stay unpacked (already low-stride); amd64 stays
  on the unpacked AVX2 path (AVX2 packing deferred). A **padded pack stride** breaks the
  power-of-two L1 conflict when kSpan is a power of two (item 24): **−9.8%/−10.7%/−8.8%
  on large-encoder fc2** on `apple-m1pro` — arm64-only, and it measured as pure noise on
  `nvidia-rtx2070s` where the whole function is dead code (see perf-dead-ends §6.1 for
  why that flat sweep was a signal, not a null result).
- **`packedFillQ8` / `MatmulBTQ8Fused` — the int8-weight twin of the packed path**
  (`matmul_blocked_q8.go`, arm64). It widens each int8 weight to f32 **inside** the pack
  tile — the ≤32 KB L1-resident buffer above — instead of materializing the whole
  `[N,K]` f32 weight first, so the ~0.9 GB/forward `deqW` DRAM round-trip is gone.
  Bit-identical to `DequantizeRowsInt8Into` + `MatmulBTInto` (the widen value and
  k-tiling match; `TestMatmulBTQ8Fused_bitIdentical`, mutation-checked). The encoder Q8
  path routes here on arm64 (`FusedQ8Applies`, gated on `HasFusedQ8Kernel` + `N%8==0`):
  **−28%/−10%/−4.8%/−2.0% end-to-end at L=8/64/256/512** on `apple-m1pro`, 1/M-shaped
  (the widen is fixed per forward). The dispatch mirrors BOTH f32 parallel axes —
  columns for small M, rows for large M — so each worker runs the fused kernel over a
  small row block and never opens `packedFill`'s missing-m-blocking gap (item 23). amd64
  keeps the vectorized dequant-then-GEMM path (`HasFusedQ8Kernel` false there).
- The quant matmuls: `MatmulBTQ8` (int8 weights), `MatmulBTQ4` (int4 group, f32
  activations — prefill path), **W8A8** (`MatmulBTW8A8` + the zero-alloc
  `…Into(ws *Workspace)` and the fused `MatmulBTW8A8Batch`), and **W4A8**
  (`MatmulBTW4A8`, int4 weights × int8 activations — the int4 *decode* path).
  See `quant.go`, `workspace.go`, `quant_w4a8*.go`.
- `DequantizeRowsInt8Into` — bulk int8→f32 weight widen (`float32(q)*scale` per
  element), vectorized both arches: amd64 AVX2 (`VPMOVSXBD`+`VCVTDQ2PS`+`VMULPS`,
  32/iter), arm64 NEON (`dequant_i8_arm64.s`, `SXTL/SXTL2`→`SCVTF`→`FMUL`, 16/iter).
  Bit-identical to the scalar loop. This is item 22's fix (a); `packedFillQ8` above is
  fix (b), which fuses the same widen into the pack so the full f32 matrix never lands.
- Dispatch knobs: `SetParallelThreshold` (MAC count to parallelize above) and
  `SetParallelWidth` (cap fan-out shards, for P/E straggler control) — both
  numerically inert (output columns are partitioned). `pool.go` is the optional
  per-`Workspace` spin-then-park worker pool.

**2. `encoder/` — the encoder's matmul orchestration.** `encoder/linalg.go` is now
thin: `matmulBTInto` dispatches small shapes to a naive in-package loop and large ones
to `linalg.MatmulBTInto` (the shared blocked GEMM), with a lone-forward row-parallel
path (`parallel.go`, each worker calling `linalg.MatmulBTInto` on its row block). The
tiling + register kernels moved to `linalg`;
`encoder/linalg_q8.go` is its int8 variant; `encoder/parallel.go` row-splits a
**single** `Encode` across cores (gated by an atomic in-flight-forward counter +
its own `parallelThreshold`, so `EncodeBatch` — already core-saturated at the
document level — stays serial per forward). This is separate from `linalg`'s
knobs above.

**Parity invariant (both layers):** parallelization and re-blocking partition
*output columns/rows* — each output is computed by one worker doing the full
K-reduction — so they're **bit-identical** to the serial path, not just within
tolerance. Tests assert exact equality.

---

## Status

- **amd64 AVX2 validated on Linux (2026-06-02, Ryzen 7 3700X, Zen 2, Go 1.26.3).**
  Every `AVX2|Dot` test PASSes with `hasAVX2=true`, no SIGILL, `-race` clean; the
  single-row and register-blocked kernels bit-match the scalar reference. Numbers
  + the one tuning finding below.
- **Perf campaign landed (v0.5.0–0.5.2)** in `linalg`: zero-alloc W8A8 decode
  (`Workspace`/`…Into`), batched W8A8 (`MatmulBTW8A8Batch`), serial-decode
  threshold + `SetParallelThreshold`/`SetParallelWidth`, the spin-park pool, and
  the column-outer W8A8 re-block (weight reused across M rows). See CHANGELOG.
  goinfer's end-to-end decode is the arbiter for those (warm microbenches mislead).
- **2026-07 CPU perf campaign (arbited across two boxes).** The CPU/SIMD-relevant
  kernel outcomes are folded into this doc: item 22a (`DequantizeRowsInt8Into` NEON
  widen, **3.43×** kernel on `apple-m1pro`), item 22b (`packedFillQ8` fused widen,
  −28%/−10%/−4.8% forward), item 24 (padded pack stride, −9.8% fc2, arm64-only). The
  full audit trail is archived under `archive/perf-campaign-2026-07/`; the two Amdahl
  decompositions are `perf-amdahl-apple-m1pro.md` (6P+2E, §5 "what transferred") and
  `perf-amdahl-linux-amd64.md`, meant to be read side by side; and every kernel idea
  that was tried and did NOT ship — item 37 outer-product, item 25, pre-packing weights
  — is in `perf-dead-ends.md` with its mechanism and number. **The one-line rule from
  that work:** on `apple-m1pro` `dotNEON2x8` is at ~95% of FMLA peak, so the f32 kernel
  is compute-bound and no load-reduction lever helps here even where the amd64 analysis
  said it would (perf-dead-ends §2).
- **`ann.FlatI8.EnableGPUShardSplit` (CPU∥GPU shard-split QueryBatch) — shipped
  2026-08-20, real win bounded and share/box/scale-dependent, NOT the ~1.5-1.7×
  the whole-corpus throughput ratio implied.** Follow-on from the July 2026 GPU
  crossover finding (M×N readback fixed by fusing top-k on-device; see
  `archive/perf-campaign-2026-07/perf-campaign-2026-07-28.md`): split the corpus
  into a device-resident shard and a CPU-scored shard, score both CONCURRENTLY
  per `QueryBatch` call, merge. The premise (measured on `apple-m1pro` at
  N=100k/batch=64: CPU 2.3k q/s, Metal 3.4k q/s, "neither saturates DRAM, so
  splitting should be roughly additive") assumed CPU and GPU are close in
  speed. Measuring the actual shard-split path — not just the whole-corpus
  numbers it was extrapolated from — on both boxes shows that assumption only
  holds sometimes:
  - **`apple-m1pro` (Metal): real, modest wins, mostly positive.** At
    N=100,000/batch=64, `metal`-only is 3548 q/s; `cpu+metal` at share=0.60
    (60k-row GPU shard) is 4485 q/s (**1.26×** over GPU-only). At
    N=10,000/batch=256, `metal`-only 23,776 q/s → `cpu+metal` share=0.35
    34,803 q/s (**1.46×**, the best case measured). But batch=1 is always
    worse than CPU-simd alone (GPU dispatch overhead isn't amortized at one
    query), and a badly-chosen share can lose to GPU-only outright (N=10,000/
    batch=64, share=0.60: 15,804 q/s vs `metal`-only's 19,726 — **0.80×**).
  - **`nvidia-rtx2070s` (CUDA): wins only at small N, actively HURTS at
    N=100,000.** At N=100,000/batch=64, `cuda`-only is 33,386 q/s
    (**30×** over CPU's 1108 q/s — this box's CPU/GPU gap is an order of
    magnitude larger than the M1's ~1.5×). Every tested share LOSES to
    GPU-only there: share=0.60 gives 6,906 q/s (**0.21×** of GPU-only),
    share=0.35 gives 4,228 q/s (**0.13×**) — the CPU shard, even at 35-40% of
    the corpus, takes far longer than CUDA's much larger complementary share,
    so the `sync.WaitGroup` merge waits on the slow side and the combined
    result is WORSE than just using CUDA alone. At N=10,000 (CPU/GPU gap only
    ~6×), the same technique genuinely helps: `cuda`-only 38,083 q/s →
    `cpu+cuda` share=0.60 44,513 q/s (**1.17×** over GPU-only, and the best
    absolute number in the whole sweep).
  - **The one-line rule:** shard-split is worth it only when CPU and GPU
    throughput are within roughly the same order of magnitude for that N —
    check the box's own `Query`/`QueryBatch` crossover numbers first. When the
    GPU is 10×+ faster than CPU at the N in question (CUDA at N≥100k on this
    box), giving the CPU ANY meaningful share of the corpus makes it the
    bottleneck and shard-split should not be used. `gpuShare` is not
    auto-tuned (see `EnableGPUShardSplit`'s doc comment) — pick it from a
    measurement like this one, per box, per N, not from the whole-corpus
    throughput ratio.
  - Correctness held everywhere measured (`parity=true`, exact-CPU-matching
    recall, on both boxes, all shares, all batches) — this is a throughput
    finding, not a correctness one. Full sweep:
    `docs/bench-records/crossover-{metal,cuda}.jsonl` (`Backend:
    "cpu+metal"`/`"cpu+cuda"` rows), reproducible via
    `AIKIT_GPU_BENCH=1 go test ./gpu/annmetal/... -run Crossover` (and the
    CUDA mirror in `gpu/anncuda`).

### AVX2 kernel numbers (Ryzen 7 3700X, `-bench 'Dot'`, MB/s)

| K     | scalar `DotGo` | single-row `dotFMA` | Dot4x4   | Dot8x4        |
|-------|---------------:|--------------------:|---------:|--------------:|
| 64    | 7.4 GB/s       | 14.3 (1.9×)         | 30.8     | 35.3          |
| 768   | 8.1 GB/s       | 44.2 (5.5×)         | 51.9     | **86.5**      |
| 3072  | 8.4 GB/s       | 49.9 (5.9×)         | 51.4     | 40.5 ⚠        |

Single-row AVX2 is ~6× scalar at the linear-layer widths; register-blocking adds
a-reuse on top (`Dot8x4` peaks 86.5 GB/s at K=768). **⚠ `Dot8x4` regresses at
K=3072** — below `Dot4x4` and even single-row — the 8 live YMM accumulators plus
streamed b-rows exceed what stays hot at large K. See follow-up §1.

---

## Testing

The SIMD kernels and their differential tests live in `linalg/`:

```bash
# Kernel correctness — AVX2 asm bit-matches the scalar reference (all tail sizes).
# TestAVX2_detection logs hasAVX2; on a non-AVX2 box the asm tests SKIP.
go test ./linalg/ -run 'AVX2|Dot|W8A8|Batch|ParallelWidth' -v

# Race-clean parallelism (parallel matmul, W8A8 pool, width).
go test -race ./linalg/

# Kernel throughput.
go test ./linalg/ -run XXX -bench 'Dot|MatmulBTW8A8|DecodePool' -benchmem

# asmdecl validates every asm stack offset vs the Go signatures (CI runs this).
go vet ./linalg/
```

Encoder-level (forward-pass parity + single-forward parallelism):

```bash
go test -race ./encoder/...            # incl. parallel.go exactness + threshold benches
go test ./encoder/ -run XXX -bench 'MatmulParallel|MatmulSerial' -benchmem
```

Model-dependent encoder tests (golden cosine vs CodeRankEmbed) skip cleanly when
the checkpoint is absent (CI), and run when `testdata/encoder-model` is present.

---

## Open follow-ups (aikit, CPU-only)

1. **`Dot8x4` large-K cliff — already mitigated at the call site; now documented
   on the public kernel.** `Dot8x4` wins at mid-K (~768) but loses to `Dot4x4`
   past it (the K=3072 regression above). The encoder does NOT hit this: its
   blocked matmul tiles K at `kBlockDefault=768` (encoder/linalg.go), which is
   exactly `Dot8x4`'s peak — `fc2` (K=3072) runs as 4×768 strips, not one 3072
   strip. So there's no call-site heuristic to add; the M10 tile tuning already
   handles it. The real exposure was the *public* `linalg.Dot8x4` godoc not
   warning external callers — now fixed (it documents the cliff and the
   "tile K to ≤~768" guidance). Revisit only if a profile shows a real caller
   feeding it large-K rows.
2. **AVX-512 path** (optional, Zen 4 / recent Intel). 16-wide, more registers;
   AVX2 already covers ~all amd64 since 2015 and AVX-512 brings downclocking
   caveats, so low priority. Same shape: CPUID leaf 7 detect, `dot_amd64.s`
   entry points, `hasAVX512` gate.
3. **Per-head attention — QK^T parallelization CLOSED; scores·V vectorized
   instead.** End-to-end CPU profile of `Model.Encode` on real weights (~500-tok
   input, `BenchmarkEncode_singleLong`) overturned the microbench-driven premise:
   QK^T is already SIMD and only ~2.6% of `Encode`, so parallelizing it across
   heads (144 spawns/forward) chases nothing. The actual hotspot was the **scores·V
   context accumulation** — a scalar triple-loop (`ctx = scores · V` per head) that
   was the single hottest line at ~⅓ of `Encode`. Fixed by folding a per-head V
   transpose into the extract and routing scores·V through the SIMD `matmulBTInto`
   (A·Bᵀ), in both `selfAttention` and `selfAttentionBatched`. Bit-exact (golden
   cosine 1.0, batch==single, `-race` clean). The win is the L² term, so it scales
   with sequence length: **~2.85× single `Encode`** at ~500 tokens, neutral (no
   regression) at ~80-token rerank passages where scores·V is a small share.
   *Follow-up — DONE, and this line was stale:* `forward_q8.go` no longer has the
   scalar scores·V loop. Both its attention paths route QKᵀ and scores·V through
   `s.mm` — `forward_q8.go:180,185` (single) and `:247,253` (batched) — which
   dispatches to `matmulBTInto` or an attached backend, exactly as the f32 sibling
   does.
4. **amd64 AVX2 `MatmulBTW4A8` kernel** — ✅ **DONE** (`dot_w4a8_amd64.s`,
   `quant_w4a8_amd64.go`). The fused int4×int8 decode kernel now exists for amd64
   too: the same nibble-unpack prologue feeding the proven `dotI8AVX2`
   sign-extend body (VPMOVSXBW+VPMADDWD+VPADDD) — fully signed, no
   unsigned-offset trick, since the nibbles are centered to int8 in-register
   first. Gated by `hasAVX2`; non-AVX2 amd64 and non-DotProd arm64 keep the
   scalar `dotW4A8Scalar`. **Validated on a Zen 2 box (Ryzen 7 3700X, AVX2, no
   VNNI):** matches the scalar oracle bit-for-bit, race-clean; at M=1 decode it
   lands ~1.7–1.9× of W8A8 and ~32× faster than `MatmulBTQ4` — on par with the
   arm64 SDOT kernel (~2.0–2.3×).
   - **Remaining: a VNNI variant** (`VPDPBUSD`, one instruction replacing the
     VPMOVSXBW+VPMADDWD pair) behind the same CPUID gate, for Zen 4+ / Intel
     Cascade Lake+. Can't be validated on the Zen 2 box (no VNNI), so it's a
     drop-in for a VNNI-capable machine; the AVX2 path is the proven fallback.

5. **`packedFill` m-blocking (item 23) — DEFERRED WITH MEASUREMENT, not dead.**
   `packedFill` re-reads the a-panel once per 8-column group (it lost `blockedFill`'s
   m-blocking). On `apple-m1pro` serial `packedFill` still runs at 75–81% of the
   ~42 GMAC/s kernel peak (fc2 M512/M690), so the a-re-read is one bounded contributor
   to a ~20% gap it shares with the compute-overlapping b-copy and the reduce. The
   m-blocking fix trades a-reads for redundant b-re-packing — a wash-risk core-GEMM
   restructure for a sub-gap win, so it is **not built**. Note the Q8 path already
   sidesteps it: `MatmulBTQ8Fused` row-splits large M so each worker's `packedFillQ8`
   runs at small M where the gap does not open. Revisit only with a profile showing the
   a-re-read is the dominant term, or as part of the §2.12-roadmap 3-level Goto GEMM.
6. **`dequantRowInt8` (K=768) — marginal-FMA issue-width probe run for real —
   ✅ DONE, NOT issue-limited on either box.** `docs/internal/priors-microgpt-c.md`
   §1 proposed injecting independent dead FMAs into a hot loop and comparing the
   marginal ns/FMA cost against the same loop's cost measured alone; a match means
   the kernel already occupied those issue slots (issue-limited, the `dotNEON2x8`
   story above), a gap means idle slots exist. Built as
   `linalg/fma_issue_probe_test.go`'s `TestFMAIssueProbe`, run on both boxes (best
   of 3, least-squares slope over N∈{0,8,16,32,64}, reproduced twice each):
   `apple-m1pro` stacked/alone ratio **0.94–0.97**, `nvidia-rtx2070s`'s Ryzen host
   (AVX2) **0.79–0.82** — both comfortably below 1.0, so **not issue-limited on
   either architecture**, even though the exact ratio differs between them (the
   AVX2 box shows more slack, not less — the opposite of what a naive read of
   priors-microgpt-c.md §2's "amd64 has fewer registers, less headroom" caution
   might suggest; the *qualitative verdict* transferred fine even though the
   *quantitative ratio* didn't, which is the distinction §2 is actually about).
   Implication: `dequantRowInt8` has idle issue slots on both boxes — it is
   waiting on something else (memory is the obvious suspect, given the kernel is a
   load/sign-extend/convert/multiply/store chain), not issue-width bound, so a
   scheduling/unrolling change here would have nothing to win against. No load
   profile currently justifies chasing the memory side further at this kernel's
   real call frequency; revisit only if a profile flags it as a real hot path.
7. **`SoftmaxRowInto` vectorized via Go 1.27's `simd` package — ✅ DONE
   (item 13, "SIMD expF32"), Experimental tier, `linalg/exp_simd.go`.**
   **Read this framing before quoting a number from this item — it has been a
   point of confusion once already:** every ratio below compares two ways of
   running the SAME math on the SAME CPU core — today's scalar Go arithmetic
   vs. Go 1.27's `simd` package compiling to vector CPU instructions (NEON on
   arm64, AVX2 on amd64). **It has nothing to do with this repo's GPU
   (CUDA/Metal) kernels** — those are `gpu/*`, a completely separate code
   path measured in `gpu`'s own docs. If this item's numbers ever make it
   into a release note, state that explicitly, the way this entry does.

   Landed after validating an uncompiled prototype
   (`~/tmcode/go127-simd-audit`, 2026-08-20 — read the go1.27.0 `src/simd`
   source directly since the authoring container had no 1.27 toolchain) by
   actually compiling and benchmarking it on both boxes before touching
   `linalg`. Measured (production `BenchmarkSoftmaxRow_vs_scalar`, same
   function both builds, only the build tag differs):

   | box | width | scalar (today) | SIMD | speedup |
   |---|---|--:|--:|--:|
   | `apple-m1pro` (NEON) | 128-bit (native) | ~5.2-5.3 ns/elem | ~2.08-2.10 ns/elem | **~2.5x** |
   | `nvidia-rtx2070s` (AVX2) | 128-bit (default) | ~6.5-6.8 ns/elem | ~2.06-2.2 ns/elem | **~3.0-3.2x** |
   | `nvidia-rtx2070s` (AVX2) | 256-bit (opt-in, `GODEBUG=simd='+256'`) | ~4.7-4.9 ns/elem | ~0.56-0.68 ns/elem (exp only, isolated) | **~6.9-8.5x** |

   Go 1.27's `simd` package defaults conservatively to 128-bit even on AVX2
   hardware; the wider width needs an explicit runtime opt-in past a safety
   check (`GODEBUG=simd='+256'`) — correctness held there too, but a library
   can't force this on its consumers unilaterally, so it stays a documented
   knob rather than a default. Scope: only `SoftmaxRowInto` (real callers:
   `encoder`, `vision`) — `ExpF32Into` has zero production callers as of this
   writing and was deliberately NOT vectorized, since doing so correctly
   would mean replicating `ExpF32`'s NaN/overflow/underflow guards via SIMD
   masks for a function nobody calls; the validated prototype and this
   landing both use `expF32Core`'s narrower "already-bounded" contract,
   which is exactly `SoftmaxRowInto`'s real shape.

   NOT bit-identical to the non-experimental build (the vector exp differs
   by up to 1 ULP — FMA contraction; `TestExpF32CoreVec_matchesScalarULP`
   gates the bound) — gated behind `GOEXPERIMENT=simd`, off by default, so
   the default build is byte-for-byte what shipped before this landed
   (verified: full `linalg`/`encoder`/`vision` suites pass unchanged).
   Correctness verified on both boxes, both real hardware and forced
   emulation (`GODEBUG=simd=0`, runs on any box regardless of ISA) —
   `encoder`'s and `vision`'s golden/cosine tests also pass under
   `GOEXPERIMENT=simd`, confirming the ≤1 ULP shift doesn't reach them. CI
   coverage: a dedicated `simd` job (`.github/workflows/ci.yml`), both
   arm64 and amd64 runners, since this is the ONLY place the experimental
   path gets exercised — nothing else needed updating (`tools/gate`,
   `consumergate`, the release gates all operate on the module graph or the
   default build, neither of which this touches).

   Next: `A3`-`A6` siblings — `SiLUF32` done (item 10); `GELUF32`, `TanhF32`,
   `GELUTanhF32` not started — and goinfer's parity ladder
   (`queue-performance.md` items this unblocks for adoption).
8. **RoPE rotation vectorized via Go 1.27's `simd` package — ✅ DONE, both
   sites, Experimental tier.** A follow-up sweep of the rest of the
   codebase (beyond item 7's `linalg/exp.go` scope) for elementwise math
   the original audit missed found aikit's own RoPE (two independent
   implementations, both NeoX rotate_half, neither transcendental — pure
   `x*cos - y*sin` / `y*cos + x*sin`, so unlike item 7 there is no new
   minimax polynomial involved, just vectorizing arithmetic that already
   existed): `vision/qwen_encoder.go`'s `applyRotaryVision` (Qwen2-VL
   vision tower, own comment cites "~8k patches × 16 heads × 32 blocks"
   per image) and `encoder/rope.go`'s `rotateHalfInto` (text encoders,
   called per attention layer). Landed as
   `vision/rope_simd.go`/`rope_scalar.go` and
   `encoder/rope_simd.go`/`rope_scalar.go`, same build-tag dispatch shape
   as item 7.

   Measured (`BenchmarkApplyRotaryVision`/`BenchmarkRotateHalfInto`, same
   function both builds):

   | box | kernel | scalar | SIMD | speedup |
   |---|---|--:|--:|--:|
   | `apple-m1pro` | applyRotaryVision (headDim=128) | 0.534 ns/elem | 0.284 ns/elem | ~1.9x |
   | `apple-m1pro` | rotateHalfInto (half=32) | 0.401 ns/elem | 0.267 ns/elem | ~1.5x |
   | `nvidia-rtx2070s` | applyRotaryVision | 0.603 ns/elem | 0.336 ns/elem | ~1.8x |
   | `nvidia-rtx2070s` | rotateHalfInto | 0.532 ns/elem | 0.334 ns/elem | ~1.6x |

   Smaller than item 7's ~2.5-3.2x — these are tiny, cheap-per-element
   calls (no polynomial, just a handful of multiply/add/subtract), so
   fixed per-call overhead eats a bigger share of the win. Still real on
   both boxes.

   **Accuracy note, worth stating precisely because it differs from item
   7**: this is pure `Mul`/`Sub`/`Add` — no `MulAdd`, so no FMA
   contraction is *requested*. Measured anyway rather than assumed
   bit-identical (`TestApplyRotaryVision_matchesScalar`,
   `TestRotateHalfInto_matchesScalar`): **bit-identical (0 diff) on
   `nvidia-rtx2070s`**, but **~2.4e-7 max abs diff (~2 ULP-class) on
   `apple-m1pro`** — Go's arm64 scalar compiler auto-fuses `a*b - c*d`
   shapes into a real hardware FMA where amd64's does not (same asymmetry
   item 7's doc noted for `p*r+c`), so the scalar REFERENCE itself already
   differs by architecture; the SIMD build's divergence from IT differs
   correspondingly. `encoder`/`vision` golden and cosine-parity tests pass
   unchanged under `GOEXPERIMENT=simd` on both boxes.

   **Two related candidates from the same sweep, ruled out — not a
   priority call, a hard API gap:** `embed/model.go`'s `encodeIDs`
   (Model2Vec weighted-mean pooling, "77.8% of an index run") and
   `embed/pool.go`'s `L2Normalize` both carry an explicit, tested
   precision contract — sum-of-squares/weighted-sum accumulates in
   float64, each float32 element widened before the multiply-accumulate,
   narrowed to float32 only at the very end ("accumulating in float32
   silently drifts cosine below the 1−1e-5 parity bar... do not
   'optimize' it away," `embed/model.go`). **Go 1.27's `simd` package has
   no float32↔float64 conversion at all** — not even a widen — so this
   contract cannot be expressed with it, full stop. Not attempted; would
   need either a different technique entirely or waiting for a future Go
   release.

9. **SPLADE's `log1p` pooling — ✅ DONE, a new kernel, Experimental
   tier.** The fourth candidate from item 8's sweep, and genuinely
   different scope from items 7/8: `encoder/splade.go`'s pooling applies
   `float32(math.Log1p(float64(x)))` once per vocab entry post-max
   (perf-campaign item 2, already V=30522 calls, down from millions).
   Unlike items 7/8, aikit had no existing `Log1pF32` to vectorize, and
   Go 1.27's `simd` package ships zero transcendentals — this meant
   designing a new float32 kernel, not vectorizing an existing one.

   **Sourced from Cephes' verified single-precision `logf`** (Moshier,
   `single/logf.c`), not invented: small x (< 2^-12) uses
   `log1p(x) ≈ x - 0.5x²` directly (the dropped x³/3 term is empirically
   below float32 precision there); larger x computes `u = 1+x` (safe —
   x is never tiny in this branch, so no `1+x` cancellation), decomposes
   u's IEEE-754 bits into a frexp-equivalent mantissa/exponent, reduces
   into Cephes' exact domain (threshold `sqrt(2)/2`), and evaluates its
   verified 9-coefficient polynomial. Cross-checked before writing any
   code: the reduction bounds independently match Go's own
   `src/math/log1p.go` (`Sqrt2M1`/`Sqrt2HalfM1` = Cephes' `SQRTHF`
   exactly) — two independent sources agreeing on the same constants.
   Validated with a throwaway sweep script *before* touching the repo:
   max absolute error **2.97e-7** vs `math.Log1p` over x ∈ [0, 1e10]
   (`TestLog1pF32Core_matchesMathLog1p`, bound set at 1e-6 with headroom
   — comparable to `GELUF32`'s existing 1e-6 absolute bound). The vector
   kernel (`TestLog1pF32CoreVec_matchesScalar`) came back **bit-identical
   (0 diff)** to the scalar form of the same algorithm on `apple-m1pro`
   — unlike items 7/8's polynomial, this one's `p = p*m + c` reassignment
   shape apparently doesn't trigger arm64's scalar auto-fusion.

   `pooled[v]` is seeded at 0 and only ever raised by `max`, so it is
   never negative, and `log1pF32Core(0) == 0` exactly — the SIMD kernel
   applies unconditionally to every lane (`TestLog1pPoolInto_zeroIsIdentity`),
   no per-element mask needed for the scalar build's `x > 0` skip, which
   was always a call-count optimization, not a correctness requirement.
   `TestSPLADE_parity` (real cosine-vs-Python comparison) stays at
   **1.000000 unchanged** under `GOEXPERIMENT=simd` — the ~3e-7 error
   doesn't move it at all.

   Measured (`BenchmarkLog1pPoolInto`, V=30522, ~15% positive density):
   `apple-m1pro` 2.193 → 1.595 ns/elem (**~1.37x**), `nvidia-rtx2070s`
   3.354 → 2.323 ns/elem (**~1.44x**) — smaller than items 7/8 on both
   boxes because the vector kernel computes BOTH branches
   unconditionally for every lane and mask-selects, paying the full
   large-branch cost (bit manipulation + 9-term polynomial) even for the
   ~85% of lanes that are 0 and would have cost nothing in the scalar
   build's `x > 0` skip. Real on both boxes, but the most modest win of
   the three landed kernels — reported honestly rather than rounded up.
   Also bit-identical vector-vs-scalar on `nvidia-rtx2070s`, matching
   `apple-m1pro` (not the arch-dependent split items 7/8 measured).

10. **`SiLUInto` vectorized via Go 1.27's `simd` package — ✅ DONE, Experimental
    tier, `linalg/exp_simd.go`.** First of the `A3`-`A6` siblings item 7 left
    "next, not started" (`docs/prompts/simd-elementwise-autoresearch.md`'s T1,
    round 1) — chosen first because, unlike GELU/Erf/Tanh's two-branch
    cancellation shape, SiLU's only extra work is the guards, and it already
    has a real caller (`encoder/linalg.go`'s `silu`) that justifies building
    them.

    **The guards ARE the work.** `x/(1+e^-x)` feeds exp an UNBOUNDED
    argument — unlike item 7's `SoftmaxRowInto`, whose domain after
    subtracting the row max is always `x <= 0`, so `expF32CoreVec` could
    stay narrow and skip `ExpF32`'s overflow/NaN guards (that item's own
    entry says so explicitly: "`ExpF32Into` has zero production callers...
    deliberately NOT vectorized"). SiLU is the caller that changes that
    calculus. Two things had to be built, both new:
    - `expF32Vec` — the full vector `ExpF32`, wrapping `expF32CoreVec` with
      overflow (→ `+Inf`, via `IfElse`) and NaN (→ NaN, via the `x != x`
      IEEE identity, also `IfElse`) — the two-branch compute-both-and-select
      pattern this file already uses for Erf/Tanh, just for guards instead
      of algorithm branches.
    - `expF32CoreVec` itself needed a real fix, not just a wrapper: its
      e>=255 two-step scale (the boundary correction scalar `expF32Core` has
      for when `k` reaches 128 while the true result is still finite — `2^k`
      built in one step would encode `+Inf` and poison a finite answer) was
      never replicated, because softmax's `x <= 0` domain can never reach
      it. SiLU's `e^-x` does reach it (`x` near `-88`). Extended in place
      (same function, same existing `TestExpF32CoreVec_matchesScalarULP`
      gate, which never exercised the new branch since it's still
      unreachable for `x <= 0` — fresh coverage added specifically for the
      newly-reachable boundary: `TestExpF32Vec_matchesScalarULP`).

    SiLU's own saturating ends (per `SiLUF32`'s doc comment) fall out of
    `expF32Vec`'s guards with no additional branch: very negative `x` makes
    `e^-x = +Inf`, and IEEE division gives `x/(1+Inf)` a signed zero; very
    positive `x` flushes `e^-x` to 0 (already handled) and the result is
    exactly `x`.

    Measured (`BenchmarkSiLU_vs_scalar`, same `SiLUInto`, only the build tag
    differs — compared against the CURRENT shipped scalar path, not the
    `math.Exp` strawman the benchmark also carries for scale):

    | box | scalar (today) | SIMD | speedup |
    |---|--:|--:|--:|
    | `apple-m1pro` (NEON) | 3.86 ns/elem | 1.34 ns/elem | **~2.9x** |
    | `nvidia-rtx2070s` (AVX2, 128-bit default) | 7.48 ns/elem | 2.61 ns/elem | **~2.86x** |

    Both boxes land within 2% of each other — unusually tight agreement for
    this doc's SIMD entries, likely because the two new guard branches
    (`IfElse`-based, not a polynomial) cost about the same fraction of the
    kernel on both ISAs. Up to 2 ULP vector-vs-scalar drift
    (`TestSiLUIntoRaw_matchesScalar`, within `SiLUF32`'s own 4 ULP contract)
    — NOT bit-identical to the non-experimental build, same class of
    difference as items 7-9. Full gate green on both boxes: accuracy/ULP
    tests, `encoder`+`vision` golden/cosine suites under
    `GOEXPERIMENT=simd`, the `GODEBUG=simd=0` emulation leg, `-race`, and
    the default (non-experimental) build verified byte-for-byte unchanged
    (full `linalg` suite passes with zero diff).

    Next: `GELUF32`/`ErfF32` and `TanhF32`/`GELUTanhF32` (T1's remaining two
    targets — both two-branch cancellation kernels, a different shape from
    SiLU's guards-only case) — autoresearch loop round 2.

11. **`GELUInto`/`ErfF32` vectorized via Go 1.27's `simd` package — ✅ DONE,
    Experimental tier, `linalg/exp_simd.go`.** Second of the `A3`-`A6`
    siblings (autoresearch round 2), and a genuinely different shape from
    item 10's SiLU: Erf is a TWO-BRANCH kernel (series for `|x|<1`, no
    cancellation; the A&S 7.1.26 tail for `|x|>=1`, stable because
    `1-(small)` doesn't cancel there), so the vector form computes BOTH
    branches unconditionally and mask-selects with `IfElse` — exactly the
    pattern this file's SiLU entry named as the alternative shape, now
    built.

    New `erfVec`, its own `erfSIMDConsts` (kept separate from
    `expSIMDConsts` — Erf's ~20 extra broadcasts have no reason to ride
    softmax's or SiLU's hot path). The tail branch's `exp(-x²)` reuses the
    NARROW `expF32CoreVec` (not item 10's full-range `expF32Vec`): for the
    `|x|` in `[1,4]` the tail branch ever runs at (`|x|>4` saturates to ±1
    separately, before reaching it), `-x²` is always in `[-16,-1]` —
    comfortably inside `expF32CoreVec`'s validated domain, no overflow risk,
    same reasoning softmax's own narrow reuse already established. NaN is
    NOT specially guarded here (unlike item 10's guard for SiLU's real,
    NaN-reachable caller): GELU feeds on a forward pass's own activations,
    no production input has ever been NaN, and the existing scalar `ErfF32`
    has no stated or tested NaN contract to preserve — this vectorizes the
    domain the scalar function and its real callers actually see, not a
    hypothetical wider one.

    Measured (`BenchmarkGELU_vs_mathErf`, same `GELUInto`, only the build
    tag differs — the CURRENT shipped scalar path, not the `math.Erf`
    strawman the benchmark also carries for scale):

    | box | scalar (today) | SIMD | speedup |
    |---|--:|--:|--:|
    | `apple-m1pro` (NEON) | 12.97 ns/elem | 2.50 ns/elem | **~5.2x** |
    | `nvidia-rtx2070s` (AVX2, 128-bit default) | 16.64 ns/elem | 3.92 ns/elem | **~4.25x** |

    The biggest speedup of any item in this doc so far — consistent with
    the two-branch premise: the scalar kernel already pays for one branch's
    worth of work per call, but the vector kernel pays for BOTH branches on
    every lane regardless of which one a given element "needs", and the
    branches here (an 11-term polynomial vs. a division-heavy 5-term one
    plus an exp call) are each substantial, so lane-parallelism buys more
    than it did for SiLU's single-branch-plus-guards shape. Up to 2.4e-07
    max absolute vector-vs-scalar drift (`TestErfVec_matchesScalar`,
    `TestGELUIntoRaw_matchesScalar` — both within `GELUF32`'s own 1e-6
    absolute contract) — NOT bit-identical to the non-experimental build,
    same class of difference as items 7-10. Full gate green on both boxes:
    accuracy/ULP-class tests, `encoder`+`vision` golden/cosine suites under
    `GOEXPERIMENT=simd`, the `GODEBUG=simd=0` emulation leg, `-race`, and
    the default (non-experimental) build verified byte-for-byte unchanged.

    Next: `TanhF32`/`GELUTanhF32` (T1's last target, same two-branch
    cancellation shape as Erf) — autoresearch loop round 3.

GPU follow-ups (resident buffers, tiled kernel, batch-tiling) are **goinfer's** —
see `goinfer/gpu` and goinfer's perf docs.

---

## Native K-quant matmul (Q4_K/Q6_K × Q8_K) — evaluated, NOT shipped

**Result: negative for the stated gate.** `docs/internal/archive/task-q8k-integer-accum.md` asked whether a
native integer-accumulation K-quant kernel (the cpubrrr / llama.cpp `ggml_vec_dot_q6_K_q8_K`
algorithm — quantize activations to Q8_K, accumulate sub-block int dot products weighted by
the integer sub-scales, convert to float once per 256-superblock) beats the current decode
path (dequant→int8-requant→`MatmulBTW8A8`, which at decode is just W8A8 over the resident
int8 weight) by **≥1.3× at M=1**. It cannot, for Q6_K — provably.

**What was built and validated** (kept in `linalg/kquant*.go`, Experimental tier, tested and
correct, but deliberately **not wired into `WeightMat`**): `QuantizeActQ8K` (per-256-block
int8 + f32 scale + exact per-16 bsums); `unpackQ6K`/`unpackQ4K`, bit-identical to
`embed/gguf.go`'s dequant (drift-guarded, `TestKQuantUnpackMatchesEmbed`); scalar
integer-accum dots; and an SDOT path (`dotPartials16SDOT`, one Go→asm crossing per superblock)
bit-exact against the scalar reference. So the arithmetic half works and is fast.

**Measured (Apple M-series, this box, `BenchmarkGEMV_*`, M=1):**

| shape | Q6_K native | W8A8 baseline | ratio |
|---|---|---|---|
| K2048 × N2048 | 2.21 ms | 0.111 ms | **20× slower** |
| K4096 × N4096 | 8.79 ms | 0.196 ms | **45× slower** |

The SDOT does the same MAC count as W8A8 (~0.2 ms); ~98% of the native time is the weight
unpack. A SIMD bit-unpack would cut that — but it cannot cross the gate, because there is a
hard ceiling above any unpack:

**The byte-ratio ceiling.** A native K-quant kernel's only advantage is reading fewer weight
bytes. So the speedup cannot exceed the byte-count ratio:
- If W8A8 is **bandwidth-bound**, Q6_K reads 210 B/superblock vs int8's 256 → at best
  256/210 = **1.22×**, and it still adds unpack+SDOT compute → ≤ 1.22×.
- If W8A8 is **compute-bound**, Q6_K does the *same* SDOT MACs *plus* the unpack → strictly
  more work → ≤ 1.0× (loses).

Either way **Q6_K ≤ 1.22× < 1.3×**: the gate is unreachable regardless of kernel quality, so
the SIMD-unpack asm was not written. Q4_K's ratio is 256/144 = **1.78×** (headroom exists),
but cpubrrr's *own* optimized Q4_K/Q6_K kernel landed **~1.12×** over llama.cpp in practice —
below 1.3× — so Q4_K was not pursued either. This confirms `task-native-q6k-kernel.md`'s
original caution on the last axis it had set aside (compute/throughput): the win is real but
thin, and below this project's bar for adding a native-GGUF weight path.

Where the cpubrrr win actually lives is **MXFP4 MoE (~5×)**, which is a different format and a
missing model family — tracked as goinfer's A2 (`task-mxfp4-gptoss.md`), not here.

Reproduce: `go test -bench GEMV -benchmem ./linalg`.

---

## File reference

```
linalg/dot_{arm64,amd64,generic,other}.{go,s}  dot kernels + build-tag dispatch
linalg/dot_i8*_arm64.s, dotprod_arm64_*.go      int8 NEON / SDOT (HWCAP-selected)
linalg/dot_w4a8_arm64.s, quant_w4a8*.go         fused int4×int8 decode kernel + scalar fallback
linalg/dot_amd64.go                             AVX2 dispatch + CPUID/XGETBV detect
linalg/linalg.go                                Dot*/MatmulBT + SetParallelThreshold/Width
linalg/quant.go                                 Q8/Q4/W8A8 matmuls (+ Into/Batch)
linalg/dequant_i8.go, dequant_i8_{arm64.s,amd64.go}  bulk int8→f32 widen (item 22a)
linalg/matmul_blocked.go, matmul_blocked_q8.go  packed f32 GEMM + fused-widen Q8 (22b)
linalg/workspace.go, pool.go                    reusable scratch + spin-park worker pool
linalg/{dot,dot_amd64,width,quant,batch,pool}_test.go   kernel/parity/bench tests
encoder/linalg.go, linalg_q8.go                 encoder's cache-blocked matmul (uses linalg.Dot*)
encoder/parallel.go                             single-forward row-parallel (in-flight gate)
```

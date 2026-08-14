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

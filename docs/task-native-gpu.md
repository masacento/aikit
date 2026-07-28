# aikit native GPU (Metal + CUDA) — full plan

> **BLUF.** aikit becomes a cgo-free retrieval/embedding toolkit whose **batch** workloads
> — ANN search, vision ViT, text encoding — run on **native GPU (CUDA / Metal), cgo-free end
> to end**, via a GPU-compute substrate aikit *owns* — exactly the way `linalg` already owns
> the CPU substrate. Today aikit's only GPU path is WebGPU (cgo, quarantined behind an
> inversion). This plan gives it native, cgo-free acceleration **where a GPU actually pays
> (batch)**, and unlocks **ANN** — the one hot-path workload with no GPU today.
>
> **Scope note ("as much as possible"):** the plan lays out the *full* arc, but it's phased
> so each phase is independently valuable and independently gated. You can stop after any
> phase with a coherent result. ANN (Phase 2) alone is the headline; the rest compounds it.

## Why — the case, strongest first

1. **Batch is the regime where a GPU pays.** goinfer's single-stream decode is bandwidth-
   bound → a GPU buys parity-to-modest (the whole Metal/CUDA saga). aikit's core work — ANN
   over a whole corpus, a batch of texts, a ViT over hundreds of patches — is **compute-
   bound**, where the fat-slice MMA path delivers **3–9×** (the vision path already reads
   "minutes to seconds"). The substrate built for decode pays a *higher* return pointed here.
2. **ANN is the one workload with no GPU path at all.** `ann.FlatI8.query` is a pure int8
   corpus GEMV (queries × the whole index — the largest single dimension in the system,
   >1M vectors), quantized by construction, calling CPU `linalg` directly with no backend
   seam. Native-GPU ANN takes retrieval — aikit's actual product — from CPU to native-GPU.
3. **cgo-free, end to end — an identity upgrade WebGPU can't give.** aikit gets GPU accel
   today *only* through the cgo WebGPU backend. goinfer's native backends (gocudrv / purego)
   are **cgo-free**. Native Metal/CUDA makes aikit a cgo-free retrieval toolkit that is
   *also* cgo-free-GPU-fast — no cgo anywhere, even for acceleration.
4. **Reuse + unification, not a bolt-on.** The device substrate and the quantized GEMV
   already exist in goinfer's `cuda/` + `metal/`. Extracting them to aikit makes **one GPU
   substrate serve both repos** — aikit's retrieval/vision/encode *and* goinfer's decode —
   mirroring how `linalg` already unifies the CPU path. Cleaner architecture, not more code.

## End-state architecture — the shape

aikit owns the **compute substrate**; the products build kernels on top of it:

```
aikit/linalg   — CPU matmul (MatmulBT / W8A8 / W4A8, WeightMat)          [today]
aikit/gpu      — cgo-free GPU device substrate (Device/Buffer/Pipeline)  [NEW]
                 + generic quantized GEMV (W8A8/W4A8) — the GPU twin of linalg
   ├─ metal impl  = goinfer/metal/metal.go, lifted (already zero-decoder-deps, cgo-free)
   └─ cuda  impl  = a thin wrapper over gocudrv (external, cgo-free by construction)

dispatch seam:  linalg.WeightMat.{Int8,Int4,F32}()  +  per-consumer Backend interfaces
   ├─ encoder.Backend        (exists — webgpu today; add native)
   ├─ ann.Backend            (NEW — ann has no seam today)
   └─ vision.ResidentEncoder (exists — WebGPU SigLIP; add native + Qwen ViT)

built ON TOP of aikit/gpu:
   ├─ aikit consumers:  ANN batch-GEMV, vision ViT, encoder GEMM     (this plan)
   └─ goinfer/cuda, goinfer/metal:  attention / rope / kv / moe      (already built —
                        re-pointed at aikit/gpu, same as decode already uses linalg)
```

Two properties this preserves:

- **cgo-free stays cgo-free.** Native backends are gocudrv/purego (no cgo); the WebGPU
  backend (cgo) stays quarantined behind the `encoder.Backend` inversion (`goinfer/gpu`,
  `-tags gpu`) exactly as today (`aikit/docs/architecture.md`). Default build: pure-Go CPU.
- **goinfer depends on aikit for GPU, as it already does for CPU.** The substrate lift is the
  GPU analogue of the existing `linalg` relationship — not a new dependency shape.

## Phases — each independently valuable, each gated

**Phase 0 — the seam (prerequisite) — ✅ DONE.** The `linalg.WeightMat` unification (roadmap
§2.8) is complete: `vision.qmat` and `decoder.weightMat` are both migrated onto
`linalg.WeightMat` — all three consumers on the one type, no local `weightMat` struct left —
and goinfer is re-pinned to the tagged **`aikit v1.11.0`** (root + `cuda` + `gpu` modules),
off the pseudo-version. The GPU-dispatch seam (`Int8/Int4/F32`) is in place and validated
bit-identical. Nothing cheaper remains to do before the GPU work.

**Phase 1 — scoped first cut: the device layer + one proving consumer (a *named product bet*).**
The prerequisites (Phase 0) are cleared, so nothing cheaper remains to do first — but the
trigger the plan specifies (a concrete ANN-GPU adopter) has **not** fired, and the tuned
Metal/CUDA kernels are **still moving** (MoE / Gemma / prefill all landed recently). So start
*deliberately*, as an **owner product bet** — aikit-as-public-library gains cgo-free native-GPU
because it strengthens the product, not because a trigger fired — and scope the first cut to the
parts that are **stable and verbatim**, never the parts that are moving or braided:

- **`aikit/gpu` interface + Metal impl only.** Define `Device`/`Buffer`/`Pipeline`/`Encoder`
  (compile-a-kernel, alloc buffers, dispatch, sync) and lift `goinfer/metal/metal.go`
  **verbatim** as the Metal impl — already zero-decoder-deps, cgo-free, self-contained, proven
  standalone by `metal/gemv_w4a8_test.go`. Design the interface to *admit* a CUDA impl later
  (the two backends are near-mirror-images), but **do not build CUDA yet.**
- **A minimal, correctness-only int8 GEMV — NOT the tuned decode kernels.** Write one small,
  unoptimized W8A8 GEMV in MSL, just enough to run a real quantized matmul on the device layer.
  This is *not* the production `gemv_w4a8_sa` / COAL kernels — those are tuned for single-stream
  decode, braided in the shared MSL blob, and still changing. It's a fresh, gated-vs-CPU proving
  kernel. **Tuning + the blob-split are Phase 1b**, when the decode kernels stop moving.
- **`ann` as the one proving consumer** (so the device layer isn't dormant infra). Give `ann` a
  `Backend` seam and route `FlatI8.query`'s batch int8 dot through the minimal GEMV;
  **parity-gate top-k ≡ `linalg.MatmulBTW8A8` (CPU), break-it-first.** This exercises the whole
  path end-to-end — device layer + `WeightMat` dispatch + parity discipline — on aikit's
  highest-fit workload, with none of the moving/tuned surface.
- **Re-point goinfer's Metal backend's *device layer* at `aikit/gpu` — device types only.**
  `goinfer/metal` imports `Device`/`Buffer`/… from `aikit/gpu` instead of defining them locally;
  its **tuned kernels stay in goinfer**. Pure refactor: goinfer's Metal decode stays
  **bit-identical green** across the parity suite, or the extraction changed behavior — stop.
  This proves the shared-substrate relationship (goinfer builds its kernels on aikit's device
  layer — the GPU analogue of the existing `linalg` relationship) without touching a kernel.

**Phase 1b — the CUDA device impl — ✅ DONE (device layer + proving kernels).** The Linux mirror
of Phase 1, verified on an RTX 2070 SUPER:

- **`gpu/cuda.go` (`//go:build linux`)** — the same `Device`/`Buffer`/`Queue`/`Pipeline`/`Encoder`
  vocabulary as `metal.go`, over `gocudrv` (cgo-free by construction: dlopen'd libcuda, no
  toolkit at runtime). Build-tag mutually exclusive with the Metal impl, so the shared type
  names never collide and consumers read the same on both platforms. Ann-free, like `metal.go`.
  Three documented divergences, all forced by the hardware rather than chosen: transfers are
  explicit (a discrete GPU has no UMA mapping, so Metal's zero-copy `Floats()`/`SetU32` writes
  cannot be honored — faking them would silently drop writes); kernels must bounds-check (a CUDA
  launch rounds up to whole blocks where `dispatchThreads` launches exactly n); and dispatch
  returns `error` (a failed `cuLaunchKernel` must not read back as a buffer of zeros).
- **Thread affinity — structural, not hand-rolled.** The crash class that hit Metal via
  `NSAutoreleasePool` applies to CUDA's thread-bound contexts too, but no `runtime.LockOSThread`
  appears in this layer: `gocudrv`'s `Context` owns a dedicated `LockOSThread`'d executor
  goroutine and funnels every driver call through it. `TestCUDA_concurrentScore` holds that
  claim honest.
- **`gpu/anncuda`** (nested module, mirroring `gpu/annmetal`) — the CUDA `ann.Backend` with
  minimal, correctness-only `gemv_w8a8` / `gemm_w8a8` kernels, shipped as PTX built by
  `gpu/build_ptx.sh` (NVRTC, reproducible from the committed `.cu` — never hand-edited).
- **Parity-gated, break-it-first.** GPU top-k ≡ CPU `linalg.MatmulBTW8A8` top-k for both the
  single-query GEMV and the batched GEMM, worst score Δ `0.000e+00` — bit-identical, as the
  exact-integer-arithmetic argument predicts. Mutation-tested: dropping the row rescale and
  transposing the GEMM's index decode each fail the gate. One honest limit found that way — the
  off-block-boundary shape test does *not* catch a deleted bounds guard (the overhang lands in
  allocation slack); the device layer's sentinel-canary `TestCUDA_tailBlockGuard` is what does,
  and the tests say so.

**goinfer's CUDA device re-point — ✅ DONE.** goinfer `25a4711` deletes its raw-gocudrv device
layer and builds its tuned decode kernels on `aikit/gpu` instead — the CUDA analog of the Metal
re-point, and the second proof of the shared-substrate relationship. It is what drove the CUDA
surface past the ANN proving path's needs: `gpu/v0.2.0` dispatched buffers-only on derived 1-D
geometry, which a tuned kernel set cannot express, so **`v0.3.0`** added scalar kernel args passed
by value, explicit grid/block geometry with dynamic shared memory, async `Launch` + explicit
`Sync`, pinned host memory, and generic typed-buffer verbs; **`v0.3.1`** added `ArgNull` for
optional-buffer binds. A consumer can now express a whole decode loop importing **only**
`aikit/gpu`, with no gocudrv type in any signature.

**Still deferred (gated on the tuned kernels stabilizing):** the **blob-split + lift of the tuned
W4A8/W8A8 kernels** into aikit — the one genuinely expensive piece left, because the generic
quantized GEMV is physically fused with the LLM-specific kernels in goinfer's shared PTX blobs
(the risk this plan has flagged from the start: the device substrate is the clean part, the
kernels are the work). `gpu/anncuda`'s minimal, correctness-only kernels are the documented
swap-in point when that trigger fires.

**Phase 2 — ANN-GPU, completed (the headline unlock).** Phase 1 lands the `ann.Backend` seam
and a minimal single-query path; Phase 2 turns it into the real win: swap the proving kernel for
the **tuned W8A8** (from the Phase-1b blob-split), add **CUDA**, and **batch multiple queries →
an int8 GEMM** — the GPU's sweet spot, over the biggest N in the system (>1M vectors), including
the paged `LoadFlatI8MmapPaged` path. **Parity-gated:** GPU-ANN top-k ≡ CPU-ANN top-k on real
indexes (rank-exact within the int8 tolerance). This is *new* coverage (ANN has none today) and
the biggest single-workload win — the payoff the product bet is aimed at.

**Phase 3 — vision native + the Qwen ViT resident path — ✅ DONE on CUDA + Metal (SigLIP and Qwen2.5-VL on both).**
`vision.ResidentEncoder` already existed (WebGPU SigLIP, "~9×"); the native CUDA
*and* Metal implementations now exist too:

- **`gpu/vit.cu` + `cuda_vit.go`** — the transformer-encoder kernel set on top of the device
  layer: quantized GEMM, f32 GEMM, LayerNorm, tanh-GELU, bidirectional multi-head attention,
  per-row int8 quantize, and the two broadcast adds. These live in `gpu/` rather than beside
  the vision backend because **none of them is vision-specific** — they are the same ops a
  text encoder needs, so Phase 4 builds on them rather than duplicating them.
- **`gpu/visioncuda`** (nested module, mirroring `gpu/anncuda`) — the `vision.ResidentEncoder`
  itself. The tower uploads once and the `[np, hidden]` residual stream never leaves the
  device between the patch embed and the post-LayerNorm; only patches go up and the last
  hidden state comes back. One `Sync` per forward, not per op.
- **Parity-gated vs the CPU tower**, which is what forced every kernel to mirror
  `vision/encoder.go`'s *exact* formulation rather than a standard one — double-accumulated
  LayerNorm/softmax, tanh-GELU (not erf), eps on the variance. Result on the pinned
  siglip-tiny checkpoint: **cosine 1.000000000, worst abs Δ 7.15e-07**, with break-it-first
  (a negated input scores −0.49). Each kernel is *also* gated individually against a CPU
  reference — a whole-tower cosine can stay high while one op is subtly wrong.

- **`gpu/metal_vit.go` + `gpu/visionmetal`** (Apple box) — the Metal mirror: the same 8 kernels
  in MSL and the same `vision.ResidentEncoder`, exported-name-identical to the CUDA side so a
  consumer stays platform-agnostic. The one thing that did **not** port is `double` — MSL has
  none — so LayerNorm mean/variance and the softmax sum accumulate in f32 with a pairwise
  (tree) reduction, and GELU uses `precise::tanh`; the ViT library is compiled fast-math-OFF
  (`CompileLibraryPrecise`) so the per-row quant scale (`maxAbs/127`) is an exact divide, not a
  reciprocal approximation. That was enough to reach the **same** result as the double CUDA
  kernels: **cosine 1.000000000, worst abs Δ 6.71e-07** on siglip-tiny, break-it-first −0.49,
  each kernel individually gated (all beat the CUDA per-kernel bars: layernorm 9.5e-07, gelu
  4.8e-07, attention 2.4e-07, quant byte-exact). Metal-specific dispatch inversions: exact
  `dispatchThreads` (no bounds checks), UMA (`Floats()` is a live view, no upload/download),
  scalars as 1-element buffers, and one `LockOSThread` per forward for the autorelease pool.

**Qwen2.5-VL — ✅ DONE on CUDA + Metal.** `vision.QwenResidentEncoder` (the seam `qwen_encoder.go`
flagged as a follow-on) plus `gpu/qwencuda` (CUDA) and `gpu/qwenmetal` (Metal). Five kernels
beyond the SigLIP set — weight-only RMSNorm, NeoX 2D rotary on a fused QKV, **segmented**
attention (windowed vs full per block), gated SiLU, and an **erf**-GELU distinct from SigLIP's
tanh one. The seam itself (`vision/qwen_resident.go`: `BuildWindowPlan`, `IsFullAtt`,
`MergeHidden`, the fp32-or-int8 `QwenGPUWeights`) is platform-neutral, so the Metal side was only
the five MSL kernels + the resident wiring, structurally identical to `qwencuda`. Parity on the
pinned tiny tower, **identical on both platforms**: ViT **cosine 1.000000000** (Δ 1.13e-06) and
merged features **1.000000000** (CUDA Δ 6.71e-08, Metal Δ 5.96e-08), both attention kinds
exercised, break-it-first −0.999971. Per-kernel Metal gates all beat the CUDA bars (rmsnorm
9.5e-07, gelu_erf 4.8e-07 and distinct from gelu_tanh by 4.7e-4, silu 9.5e-07, rope exact with the
v-third proven untouched, attention_seg 1.8e-07 with bounds-enforcement proven).

Two Metal-specific notes worth recording. MSL's `half` is the f16 **type keyword** (the rotary's
`hd/2` had to be renamed off it), and — contrary to the CUDA-box hand-off — **MSL has no stdlib
`erf`** on this toolchain, so `gelu_erf` carries an Abramowitz-&-Stegun 7.1.26 approximation
(max err ~1.5e-7, at the f32 floor); the erf-vs-tanh discrimination is unaffected. The f32
reductions (no `double` in MSL) and the same `CompileLibraryPrecise` fast-math-off compile as the
SigLIP set apply here too.

Two deliberate host/device splits, documented in the package: the **window permutation** is done
by permuting pixel ROWS before upload (the patch embed is row-wise, so this needs no gather
kernel) and reuses `vision.BuildWindowPlan` rather than reimplementing the index arithmetic; and
the **patch merger** stays on the CPU, being three small ops over n_patches/merge² groups against
32 blocks of tower.

**Phase 3 is complete.** The correctness-first GEMMs have since been **tiled** and then
superseded by a **simdgroup_matrix** kernel on Metal (see Phase 4's GEMM section — the kernels
are shared, `gpu/`-level, not vision-specific). Note the parity fixtures are *tiny* towers: they
gate correctness sharply and say nothing about throughput, so all GEMM tuning was driven by the
`BenchmarkMetalGEMMF32` sweep, not by the fixtures (`docs/BENCH-gpu.md` is that methodology).

**Phase 4 — encoder native — ✅ DONE (f32); one int8 decision remains.** The seam is wired and
both native backends (`gpu/enccuda`, `gpu/encmetal`) ship, parity-gated, with the production
GEMMs and an end-to-end figure. The single open item is the **int8 (`LoadQ8`) encoder path**, and
it is a *decision*, not unbuilt work — it routes through the existing `WeightMat` W8A8 GEMV, NOT
by widening `Backend` (see the scope note below).

The plan said "`encoder.Backend` already has `webgpu`; add native Metal/CUDA", which read as
*plug a backend into a working seam*. It wasn't: the interface existed, `NewBackend` resolved
it and `goinfer/gpu` registered `"webgpu"`, but **nothing in the forward ever called
`Backend.MatmulBT`**. A caller could ask for a backend, receive one, and have it do nothing —
silently, and with every test still green, because the pure-Go path produced correct numbers
either way. That is now fixed:

- **`scratch.mm` is the dispatch point.** A `*scratch` was already threaded through every hot
  function (`selfAttention`, `selfAttentionBatched`, `geluMLP`, `swigluMLP`, the BERT/GTE/q8
  forwards), so the seam needed no signature churn — 32 call sites now route through it.
- **`UseBackend` on `Model` / `ModelQ8` / `BERT` / `GTE`** is the opt-in. apidiff vs `v1.12.0`:
  **4 compatible additions, 0 incompatible** — the Hard-tier guarantee holds.
- **Not setting a backend calls exactly the function the call sites called before**, so the
  pure-Go numerics are unchanged by construction rather than by argument.
- **Gated on dispatch, not just numerics** (`encoder/backend_wiring_test.go`): a spy backend
  must *observe* the matmuls (it observed none before), nil ≡ delegating ≡ `NewBackend("cpu")`
  bit-identical, a deliberately-wrong backend must *change* the output (or the seam is
  decorative), the MLP projections must be routed, and a pooled scratch must not leak a backend
  into the next forward. These use synthetic weights, not a checkpoint — a seam test that skips
  in CI is what let the dangling seam survive.

**Scope limit, deliberate:** this routes the **f32** path, which is all `Backend`'s single
`MatmulBT(a, b, dst, M, K, N)` can express. The int8 (`LoadQ8`) projections go through
`matmulBTQ8Into`, whose int8 weights and per-row scales do not fit that signature, and stay on
the CPU. Widening `Backend` is a separate, deliberate decision — it is Hard-tier surface.

> **Scope note — that f32 seam is f32-only, and the int8 path does NOT come along by widening it.**
> `MatmulBT(a, b, dst []float32, …)` carries only f32, so a native encoder wired through it
> accelerates the f32 path while the int8 `LoadQ8` projections (int8 weights + per-row scales)
> stay on CPU — their representation doesn't fit that signature. The fix is **not** to widen
> `Backend` with a quantized method: that grows an already **Hard-tier** interface *and* builds a
> second quantized-dispatch surface parallel to `WeightMat`, re-duplicating exactly what the
> §2.8 unification collapsed. encoder's Q8 projections are **already `linalg.WeightMat`s**, so the
> int8 encoder path routes through `WeightMat.Int8()` into the **same `aikit/gpu` W8A8 GEMV the
> Phase-1 first cut already builds for ANN** — no new interface, consistent with ANN and vision.
> Consequence to accept: the native path (via `WeightMat`) does **both f32 and int8**; the WebGPU
> path (via `Backend.MatmulBT`, f32-only) stays f32 — native is the more-capable fast path, WebGPU
> the portable fallback. Widening `Backend` is reserved for a concrete **int8-on-WebGPU** pull,
> which doesn't exist today; don't do it speculatively.

**CUDA backend — ✅ DONE (`gpu/enccuda`).** Registers `"cuda"`, uses `gemm_f32_tiled`, and
declines small shapes to the CPU path. Parity vs a float64 reference on both sides of the
threshold; measured crossover in `minGPUFlops`'s doc comment.

**Metal backend — ✅ DONE (`gpu/encmetal`).** The Apple mirror: registers `"metal"`, runs the
production `gemm_f32_sg` (via `Run2D`), declines small shapes to CPU, same float64-reference
parity and the same pooled-scratch regression test (`TestEncMetal_noStaleOperands`). The UMA
difference is the point — the "upload" is a `copy()` into the buffer's `Floats()` view and the
download a `copy()` back, no PCIe transfer and none of the CUDA upload race (so the
`Buffer.upload` sync `25d6de3` added on CUDA is deliberately NOT mirrored — there is no DMA to
order against). Its `minGPUFlops` is **derived independently** from an M1-Pro sweep: `1<<27` ≈
134 MFLOP (crossover between 24 MFLOP where the GPU loses 0.83× and 151 MFLOP where it wins
1.33×; 8590 MFLOP hits 4.25×). Successive GEMM kernels dropped this crossover ~16× from where
the tiled kernel would have put it — tiled ~2.1 GFLOP → `gemm_f32_sg` ~0.5 GFLOP →
`gemm_f32_sg_big` ~0.13 GFLOP (see below). The backend routes via `ViT.GEMMF32Plan`.

**Tiled GEMMs — ✅ DONE on both platforms.** `gemm_w8a8_tiled` + `gemm_f32_tiled` stage a
TILE×TILE block through shared/threadgroup memory. On Metal they launch via `Run2D`
(`dispatchThreadgroups` — UNIFORM whole threadgroups, the one Metal kernel family that keeps the
CUDA-style bounds checks, because cooperative tile-staging needs the edge threads to exist). The
asymmetric gate ports intact: W8A8 tiled is **BIT-IDENTICAL** to untiled (int32 sums are
order-independent), f32 tiled is gated against a **float64** reference (√K·ε, not against the
untiled kernel).

**Production simdgroup GEMM — ✅ DONE on Metal (two kernels, `ViT.GEMMF32Plan` routes).** The
correctness-first `gemm_f32_tiled` accumulates scalar; these use the Apple GPU's
`simdgroup_matrix` 8×8 cooperative matrix ALU. MPS was avoided deliberately — it is an Obj-C
framework dependency, whereas `simdgroup_matrix` is pure MSL and keeps the substrate
cgo-free/self-contained. Both bind buffers 0..5 (A,B,C,M,N,K), gated against a float64 reference.

- **`gemm_f32_sg`** (any shape) — a 32×32 tile, 4 simdgroups (2×2) each owning 2×2 accumulator
  fragments; per K-chunk the 128 threads stage A and Bᵀ into threadgroup memory ZERO-PADDED on
  the M/N/K edges (so `simdgroup_load` never goes out of bounds), then four
  `simdgroup_multiply_accumulate`. ~2.3–2.8× the tiled kernel (778 vs 339 GFLOP/s at 2416 MFLOP).
- **`gemm_f32_sg_big`** (aligned: M,N%32==0, K%8==0 — every large encoder/ViT GEMM) — the SAME
  32×32 tile and register footprint, but A/B fragments load DIRECTLY from device (no staging, and
  crucially **no two barriers per K-step**) and there are no bounds checks. That is the whole win:
  **~1080 GFLOP/s (≈23% of the M1-Pro f32 peak), up to 4.25× the CPU** and +4–21% over
  `gemm_f32_sg` across shapes. Two things that did NOT help and were dropped: wider register
  blocking (a 64×64 tile / 16 fragments per simdgroup spilled registers and tanked occupancy,
  ~150 GFLOP/s), and a threadgroup-count/M threshold (a stable 1s sweep showed the direct-load
  kernel wins at every aligned shape, so `GEMMF32Plan` routes purely on alignment).

`gpu/encmetal` and the ViT f32 paths (`visionmetal`/`qwenmetal` patch embed + Qwen fp32
projections) route through `GEMMF32Plan`; W8A8 stays on the tiled kernel (`simdgroup_matrix` has
no int8 form). All parity unchanged (SigLIP/Qwen cosine 1.000000000). CUDA has no analogue in
this set yet — a future `gemm_f32_wmma` (Tensor-Core `wmma`) would be the Linux-side mirror.

Two findings the interface forced, both worth acting on:

- **`encoder.Backend` has no residency hook.** It takes host slices, so both operands upload
  and the result downloads on *every* call — a 12-layer forward re-uploads the same weights
  ~72 times. The obvious fix, a pointer-keyed weight cache, is **unsound here**: in attention
  the `b` operand is a *pooled scratch* slice whose backing array is stable while its contents
  change every call, so a pointer key serves stale data silently. A size threshold does not
  rescue it — at long sequences the per-head QKᵀ outgrows any threshold. `gpu/enccuda` therefore
  does not cache, and carries a regression test for exactly that trap. **Widening `Backend`
  with a resident-weight concept is the real fix**, and is a Hard-tier decision.
- **The measured curve differs by platform — and both are CPU stories, not GPU ones.** On
  CUDA the GPU wins at every shape but by the *largest* margin on the *smallest* work (7.7× at
  1 MFLOP, 1.09× at 2416 MFLOP): the pure-Go path switches scalar→blocked-parallel at ~4 MFLOP,
  so it is weakest where the device looks strongest, and hits 329 GFLOP/s where the device is
  transfer-bound. On **Metal the crossover is higher but has moved down with each kernel** — with
  `gemm_f32_sg_big` the device wins from ~151 MFLOP (1.33×) up (1.28× at 604, 2.22× at 2416,
  **4.25× at 8590**) and loses below ~24 MFLOP, because the M1-Pro CPU is strong (~300 GFLOP/s)
  and even ~978 GFLOP/s (incl. copies) needs the size to amortize a ~250us dispatch+sync floor.
  (Tiled put this crossover near 2.4 GFLOP; `gemm_f32_sg` at ~0.5 GFLOP; `gemm_f32_sg_big` at
  ~0.13 GFLOP.) Same interface, different thresholds per platform *and* per kernel: the sweep *is*
  the deliverable, exactly as `BENCH-gpu.md` argues, and copying one constant to another box or
  kernel would have been wrong.

**End-to-end measurement — ✅ DONE on Metal (first Phase-4 end-to-end figure either box has).**
`testdata/minilm-model` is present on the Apple box, so `TestEncMetal_endToEnd` times a real
MiniLM `Encode`, CPU vs the `"metal"` backend: **CPU 88.9ms, Metal 89.1ms (1.00×), cosine
1.000000000**. The finding corroborates the threshold: even at the lower simdgroup crossover, a
single short-sequence forward (small L, hidden 384) has no matmul above it, so the backend
correctly stays entirely on the CPU and the numerics are identical — the device pays for batched
or larger-model encode. (`BENCH-gpu.md`: microbenchmarks tune, end-to-end publishes.)

**Still to do in Phase 4:** the int8 (`LoadQ8`) encoder path — the one open *decision* (route via
the `WeightMat` W8A8 GEMV, per the scope note; don't widen `Backend`). One optional throughput
follow-on remains, a **CUDA `wmma` (Tensor-Core) GEMM** on the Linux box — but note it is NOT a
straight mirror of `gemm_f32_sg_big`: Tensor Cores have no true fp32 path, so an f32-accumulate
`wmma` needs **tf32** inputs (~10-bit mantissa, ~1e-3 relative), which fails the encoder's 2e-4
f32 parity bound. It is therefore a precision decision (relax the bound and document it, or use a
non-Tensor-Core shared-memory GEMM), not a copy of the Metal kernel.

**Metal f32 GEMM — the peak-push levers were explored and the ~1.08 TFLOP/s kernel is the
practical ceiling for the simple approach.** `gemm_f32_sg_big` (direct device loads, no barriers)
sits at ~23% of the M1-Pro's ~4.7 TFLOP/s f32 peak. Three standard levers were implemented and
benchmarked (1s runs, `BenchmarkMetalGEMMF32`); none beat it, so none shipped:
- **Wider register blocking** (64×64 tile, 16 fragments/simdgroup) — ~150 GFLOP/s: 24 live
  simdgroup matrices spill registers and occupancy collapses.
- **Double-buffered staging** (ping-pong threadgroup buffers, one barrier/K-step, prefetch under
  compute) — *worse* at the shapes that matter (737 vs 968 GFLOP/s at 2416 MFLOP): the staged
  path's extra device→threadgroup hop and output staging outweigh the barrier it saves.
- **Register-level software prefetch** on the direct-load kernel (load next K-tile's fragments
  under the current MACs) — marginal and shape-dependent (+14% at large-K 604 MFLOP, +2% at 8590,
  but −24% at short-K 151 where the extra registers hurt); not a reliable win.
Closing the remaining gap to peak needs a genuinely sophisticated kernel (large-tile blocking
with co-tuned register/occupancy budgets, or Apple's newer tensor APIs) — a separate project with
uncertain payoff, well beyond tuning. The direct-load kernel is where the simple approach lands.

**Ruled out — not deferred:** `embed` (Model2Vec). This is a **settled decision, not a phase
waiting on a trigger**, and it should not be re-opened as "the last un-done item": Model2Vec is a
token→row gather + mean-pool with **no GEMM anywhere**. The work is memory-bound lookup, so there
is nothing for the MMA path to accelerate, and moving it to a device would *add* host↔device
round-trips to a workload that is already a cache-friendly scan — slower, not faster. It stays on
the CPU by design. Revisit only if `embed` itself grows a dense matmul (it has none today), never
merely for coverage's sake.

**WebGPU stays** — it's the portable "any GPU (Vulkan/DX12/Metal)" fallback; native is the
cgo-free-fast path on NVIDIA/Apple. They coexist (native preferred where present, WebGPU where
portable, CPU always).

## Cross-cutting discipline (non-negotiable)

- **Parity on every path.** Each GPU consumer path is gated vs its CPU `linalg` reference
  (ANN top-k, vision logits, encoder vectors), **break-it-first** to prove the gate isn't
  vacuous (per `docs/parity-hunt-playbook.md`). The release-qualification sweep
  (`goinfer/docs/task-release-qualification-sweep.md`) extends to aikit's consumer × backend
  cells.
- **cgo-free preserved and asserted.** Native backends never introduce cgo (the CI guard
  that fails if `webgpu` leaks into aikit's core generalizes to "no cgo in aikit core").
- **Opt-in / additive.** Native backends behind build tags (`-tags cuda` / `-tags metal`),
  `CGO_ENABLED=0`; the default aikit build stays pure-Go CPU and every consumer degrades to
  CPU on a non-GPU build.
- **Device CI is hand-run.** GitHub can't run device tests (the objc-`msgSend` SIGSEGV; no
  GPU runner) — the device/parity gates are the scripted real-hardware tier, same as goinfer.

## Honest risks

- **The blob-split (Phase 1) is the real cost** — the generic GEMV is physically fused with
  the LLM kernels in shared PTX/MSL. The device substrate (metal.go / gocudrv) is the clean
  part; the kernels are the work.
- **Two of three consumers already have cgo-WebGPU** — for encoder + vision the native win is
  incremental (cgo-free + faster); the genuinely new coverage is ANN. Sequence accordingly.
- **The device re-point must be a pure refactor** — goinfer's Metal decode bit-identical or it
  stops (parity suite is the tripwire). The *large* structural move — the CUDA impl + the
  tuned-kernel blob-split — is quarantined in **Phase 1b behind its trigger**, so the first cut
  stays small and low-risk.
- **Scope creep** — "as much as possible" is the arc, but the value is front-loaded (ANN).
  Don't let Phase 3/4 gate the ANN win.

## Recommended sequence

**Phase 0 ✅** → **Phase 1 ✅** (`aikit/gpu` + Metal device layer + minimal-kernel ANN proof +
goinfer-Metal device re-point) → **Phase 2 ✅** (ANN-GPU batch-GEMM — the headline) → **Phase 1b
✅** (CUDA device impl + CUDA ANN backend + the tuned-GEMV blob-split + the goinfer CUDA device
re-point, all parity-gated on an RTX 2070 SUPER) → **Phase 3 ✅** (SigLIP and Qwen2.5-VL both done
on BOTH CUDA and Metal, tiled and then simdgroup-GEMM'd) → **Phase 4 ✅ (f32)** (encoder native:
seam wired, `enccuda` + `encmetal` shipped with production GEMMs and an end-to-end figure; the
int8 path is a remaining `WeightMat`-dispatch decision, not unbuilt work).

**Nothing is gated behind an unfired trigger any more.** Every "wait for X to settle" in this
plan has been discharged: the tuned kernels stabilized, the blob-split turned out to be a clean
entry-point cut rather than a disentangling, and both device re-points landed bit-identical.
What remains is a single int8-encoder dispatch decision (route via `WeightMat`, not by widening
`Backend`) and optional CUDA-side GEMM throughput work; `embed` is ruled out on the merits
(above), not deferred.

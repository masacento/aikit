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

**Still deferred (gated on the tuned kernels stabilizing):** the **blob-split + lift of the tuned
W4A8/W8A8 kernels**, and re-pointing goinfer's CUDA backend at this device layer (the CUDA analog
of the Metal device re-point, tag-gated, after a `gpu/v0.2.0` cutting the new CUDA surface). Those
carry the real cost — a moving API and the blob-split — and there is no reason to pay it here.
When ANN-GPU is funded, swap the minimal proving kernels for the tuned ones.

**Phase 2 — ANN-GPU, completed (the headline unlock).** Phase 1 lands the `ann.Backend` seam
and a minimal single-query path; Phase 2 turns it into the real win: swap the proving kernel for
the **tuned W8A8** (from the Phase-1b blob-split), add **CUDA**, and **batch multiple queries →
an int8 GEMM** — the GPU's sweet spot, over the biggest N in the system (>1M vectors), including
the paged `LoadFlatI8MmapPaged` path. **Parity-gated:** GPU-ANN top-k ≡ CPU-ANN top-k on real
indexes (rank-exact within the int8 tolerance). This is *new* coverage (ANN has none today) and
the biggest single-workload win — the payoff the product bet is aimed at.

**Phase 3 — vision native + the Qwen ViT resident path.** `vision.ResidentEncoder` already
exists (WebGPU SigLIP, "~9×"); add native Metal/CUDA implementations, and build the
**Qwen2.5-VL** resident path (currently a documented follow-on, `goinfer/docs/prompts/
aikit-qwen25vl-vit.md`). Batch patches (M = np, thousands for dynamic-res Qwen) → the fat
GEMM the MMA path was built for. Parity-gated vs the CPU tower (HF-parity already exists).

**Phase 4 — encoder native.** `encoder.Backend` already has `webgpu`; add native Metal/CUDA.
Batch text embedding (M = L tokens / B·Lmax). Incremental over the existing cgo-WebGPU path —
the win here is *cgo-free + native-class*, not GPU-where-there-was-none.

**Not in scope:** `embed` (Model2Vec) — a token→row gather + mean-pool, no GEMM; GPU is the
wrong tool. **WebGPU stays** — it's the portable "any GPU (Vulkan/DX12/Metal)" fallback;
native is the cgo-free-fast path on NVIDIA/Apple. They coexist (native preferred where
present, WebGPU where portable, CPU always).

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

**Phase 0 ✅ done** → **Phase 1 ✅ done (scoped first cut)** — `aikit/gpu` + Metal device layer +
minimal-kernel ANN proof + goinfer-Metal device re-point, as a named product bet → **Phase 2
✅ done** (ANN-GPU batch-GEMM — the headline) → **Phase 1b ✅ done** (CUDA device impl + the CUDA
ANN backend, parity-gated on an RTX 2070 SUPER) → **next: the tuned-GEMV blob-split + the goinfer
CUDA device re-point**, *gated on the decode kernels settling* → Phase 3 (vision native + Qwen
ViT) → Phase 4 (encoder native). Each gated, each independently shippable, each leaving aikit
stronger and still cgo-free. The device substrate is now two-platform; the remaining cost and
moving-API risk are quarantined into the blob-split, behind its trigger.

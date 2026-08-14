# Kickoff prompt — a faster CUDA f32 GEMM (Phase 4, optional throughput)

> Linux/NVIDIA box. `gpu/enccuda` and the CUDA ViT f32 paths use `gemm_f32_tiled` — a
> correctness-first 16×16 shared-memory tile. Metal added `gemm_f32_sg_big` (~2.8× the tiled
> kernel, ~1.08 TFLOP/s on M1 Pro) and it moved the encoder crossover down ~16×. This is the
> CUDA analogue. **Optional throughput work**, not gating; do it when the box is free.
> ONE session per repo tree; `git fetch` before tagging, confirm the tag commit is on `main`.

## The precision constraint (decide this first)

Do **NOT** reach for `nvcuda::wmma`/Tensor Cores as a straight port of the Metal kernel. Tensor
Cores have **no fp32×fp32 path**: an f32-accumulate `wmma` needs **tf32** inputs (~10-bit
mantissa, ~1e-3 relative), which **exceeds the encoder's 2e-4 f32 parity bound** (the same bound
`enccuda`'s float64-reference gate and the int8 weight-only path both rely on). So:

- **Recommended — option (b): a non-Tensor-Core, register-blocked f32 kernel** that stays within
  2e-4. This is the true analogue of `gemm_f32_sg_big` and keeps the existing parity gates intact.
- **Only if you also want a separate fast lane — option (a): a tf32 `wmma` kernel** gated on a
  *deliberately relaxed, separately documented* bound, kept OFF the parity-exact path (its own
  pipeline, its own `minGPUFlops` note, flagged in `BENCH-gpu.md`). Don't relax the existing gate
  to admit it.

## What Metal learned (don't re-run the dead ends)

The winning Metal kernel was **not** a bigger tile — it was the *same* small tile with the
staging removed. Concretely, benchmarked (1s runs):
- **Wider register blocking** (a 64×64 threadgroup tile, 16 accumulator fragments per warp) —
  **lost badly** (~150 GFLOP/s): register spilling + collapsed occupancy. Don't chase big tiles.
- **Double-buffered shared-memory staging** — *worse* than the barrier-free path at the shapes
  that matter: the extra global→shared hop outweighs the barrier it saves.
- **Register-level software prefetch** — marginal and shape-dependent (+14% at large-K, −24% at
  short-K). Not a reliable win.
- **What won:** load A/B fragments **directly from global memory** into registers with **no shared
  staging and no barriers**, modest register blocking (each thread computes a small micro-tile of
  outputs, e.g. 4×4 or 8×8), for the **aligned** case; keep the existing tiled kernel as the
  any-shape fallback.

CUDA differs from Metal (explicit shared memory, warps, `wmma` exists), so re-measure rather than
assume — but the prior is: **register-blocked, coalesced global loads, minimal/no shared staging**
beats a fancier tile here. A classic CUDA register-blocked SGEMM (each thread a KxK output
micro-tile, coalesced loads, `#pragma unroll` the K step) is the target.

## The shape (mirror the Metal split)

- Add `gemm_f32_reg` (or your name) to `vit.cu`, plus an **aligned fast path** and a **dispatch
  helper** — the CUDA analogue of Metal's `ViT.GEMMF32Plan` (aligned → fast kernel, else →
  `gemm_f32_tiled`). Same signature and buffer binding as `gemm_f32_tiled`, so callers only swap
  the launch.
- Route `gpu/enccuda`'s **`gpuMatmul`** (f32) **and `gpuMatmulQ8`** (the dequant-then-f32 path)
  through the plan — both currently launch `gemm_f32_tiled`. Also the CUDA ViT f32 consumers
  (`visioncuda`/`qwencuda` patch embed + Qwen fp32 projections), matching what Metal did.
- **Regenerate `gpu/testdata/vit.ptx`** via `build_ptx.sh` (never hand-edit).

## Gates & deliverable

Port the **float64-reference** parity gate across shapes incl. tile/K overhang (mirror
`TestMetal_gemmF32SG`), 2e-4 bound. Add a `tiled` vs `reg` benchmark (mirror
`BenchmarkMetalGEMMF32`), report GFLOP/s. **Re-measure the crossover** and update `enccuda`'s
`minGPUFlops` doc table (the faster kernel should lower it, as on Metal). All parity unchanged
(SigLIP/Qwen cosine, the int8 weight-only 2e-4). `CGO_ENABLED=0`, `CgoFiles=[]`. Don't touch
`//go:build darwin`. Update `docs/task-native-gpu.md` (CUDA GEMM done; crossover). Cut
**`gpu/v0.14.0`**.

Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

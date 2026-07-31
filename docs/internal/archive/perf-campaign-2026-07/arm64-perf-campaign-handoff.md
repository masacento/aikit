# Prompt — aikit perf campaign, arm64 half (MacBook / M1 Pro)

> **For:** the aikit session on the Apple M1 Pro.
> **Why now:** the Linux/amd64 box has closed ~30 items of
> [`perf-campaign-2026-07-28.md`](perf-campaign-2026-07-28.md). What
> is left is almost entirely **arm64-gated** — several items live in code that
> does not execute on amd64 at all — plus one change that was tuned on amd64 and
> needs checking on yours.

## Read first, in this order

1. [`docs/internal/measuring-performance.md`](../../measuring-performance.md)
   — the campaign's measurement discipline, 24 entries, every one a mistake that
   actually happened. **§1.11 is the one that matters most to you: per-kernel
   tables do not transfer between machines.** Every number quoted below was
   measured on a 3700X and is evidence about that box, not yours.
2. The campaign doc's §7.21, §7.32 (why items 23/24 are arm64-only) and §7.14
   (the parallel-axis change described in Task 0).

---

## Task 0 — FIRST, and the reason this handoff is urgent

**I changed the default intra-op matmul axis based on amd64 measurements, and
the original tuning table for that code says the opposite on your box.**

`encoder/matmulBTInto` now prefers a **column** split (partition the weights)
over a **row** split (replicate the weights per worker) whenever
`M < minRowsPerWorker·NumCPU`. On the 3700X columns won at every trunk shape, up
to 3.33×, for a structural reason: a row split streams the `[N,K]` weight matrix
once **per worker**, a column split streams it once.

But the cost of that replication depends on the cache, and `parallelThreshold`'s
table in `encoder/parallel.go` — measured on **your** M1 Pro — shows row
splitting delivering 2.6–4.5×. A large unified cache makes replication much
cheaper than it is across a desktop part's split-CCX L3. **So this may be
neutral-to-negative on arm64 and nobody has checked.**

What to do:

- Run `BenchmarkParallelAxis` (`encoder/parallel_sweep_test.go`). It sweeps
  serial/rows/cols over real trunk shapes with a 12-deep weight bank.
- Then the end-to-end arbiter: `BenchmarkGTEEncode`, `BenchmarkSPLADEExpand`,
  `BenchmarkCrossEncoderScore` (fetch recipes in `scripts/README.md`).
- If columns lose on arm64, the fix is **not** to revert — make the crossover
  arch-aware and say why in the code, the way `wantParallelCols` already
  documents its amd64 evidence.

While you are there, these constants are all amd64-measured and unverified on
arm64. Re-measure any that gate something you touch:

| constant | file | tuned on |
|---|---|---|
| `minColsForSplit`, the `minRowsPerWorker·NumCPU` crossover | `encoder/parallel.go` | 3700X |
| `parallelRowsThreshold` (softmax/GELU row split) | `encoder/parallel.go` | 3700X |
| `flatParallelThreshold` | `ann/flat.go` | 3700X |
| `batchTokenBudget`, `maxBatchSeqs` | `encoder/model.go` | 3700X |

Also worth re-measuring, because it should be **better** on your box: item 13's
float32 transcendentals. The campaign's original table put `math.Exp` at
34.3 ns/elem (measured on an M1 Pro); the 3700X measures 15, which is why the
amd64 win was smaller than predicted. On arm64 `linalg.ExpF32`/`GELUF32` should
pay more than the −20% geomean seen here.

---

## Task 1 — the arm64-only items (nobody can measure these but you)

`blockedFill` gates `packedFill` on `has2x8Kernel`, which is **`false` off
arm64** (`linalg/kernel_other.go`), and additionally requires
`K ≥ packKThreshold` (2048). **`packedFill` is dead code on amd64.** Three items
live inside it:

- **Item 24 — packed stride is a 4096 B power-of-two.** Attempted here and
  reverted (§7.21) because it cannot be measured. Two corrections to the item as
  written:
  - **The proposed "+4 float pad" is too small to work.** Four floats is 16
    bytes, less than a cache line, so it cannot move a row to a different L1
    set: with stride 4112, `floor(bi·4112/64) mod 64 = floor(bi/4)`, splitting 8
    rows across just 2 sets. Use **16 floats** (stride 4160 = 65 lines →
    `bi mod 64`, all 8 distinct).
  - **It may be a no-op on your hardware specifically.** The finding itself says
    the pathology "escapes only on Apple P-cores (128 KB L1D ⇒ 256 sets)" — and
    Apple P-cores are the only place `packedFill` runs. Measure before building;
    a flat sweep is a legitimate answer here, and §1.1's corollary applies.
- **Item 23 — `packedFill` lost `blockedFill`'s m-blocking**, so the a-panel is
  re-read per 8-column group. Never measured on any box. Sized "plausibly large
  at prefill".
- **Item 22(b) — fuse the int8→f32 widen into `packedFill`'s b-panel pack.**
  Fix (a) (a SIMD widen, `linalg.DequantizeRowsInt8Into`) landed on amd64 and is
  worth **−58% at short input** (§7.18); the arm64 path still falls back to the
  scalar widen, so **an arm64 NEON widen (`SXTL`+`SCVTF`+`FMUL`) is the cheap
  win to do first**. (b) additionally removes the 0.9 GB DRAM round-trip and
  `scratch.deqW`.

Then the two arm64 kernel items:

- **Item 20 — int8 register blocking (1×4 / 4×1); the arm64 kernel has none.**
  Estimated 1.2–1.6× GEMV, more at prefill.
- **Item 25 — arm64 `Dot2x8` has the wrong MR×NR:** 4×4 needs 8 loads per 16
  FMLAs versus today's 10. Estimated 1.1–1.25× arm64 f32 GEMM, S–M effort.

**Item 36 (i8mm/SMMLA) is NOT for you** — the doc is explicit that an M1 Pro
cannot run it. Needs M2+/Graviton3.

---

## Task 2 — the largest remaining lever, if you want a big one

**Item 37 — outer-product f32 microkernel via by-element `FMLA`** (4.0 FMLA per
load versus today's 1.6), L effort, and the doc marks it `≠` (numerics change).

Evidence from the amd64 side that this is the right target: after ~30 items,
`GTE.Encode` at L=690 profiles at **74% `dotFMA8`** — the GEMM microkernel
itself — running at ~171 GFLOP/s against a realistic ~400–500 achievable on that
part. Roughly **2× is still sitting in the microkernel**, and the same
dot-product-shaped kernel is what arm64 uses. Item 37 is the arm64 fix for
exactly that; **there is no equivalent item for amd64, which is a real gap in
the campaign doc** worth noting if you touch this area.

---

## Non-negotiables

- **`CGO_ENABLED=0` throughout**, and assert `CgoFiles=[]` — unchanged from the
  earlier native-GPU work.
- **Do not touch `//go:build linux` / CUDA files** (`gpu/cuda*.go`,
  `gpu/anncuda`, `gpu/enccuda`). Symmetrically, I have not touched the darwin
  ones.
- **Parity gates are the acceptance criterion**, not the benchmark. Anything
  claiming bit-identity must be gated with exact equality and the gate must be
  **mutation-checked** — break it deliberately, confirm it goes red. Item 17 on
  this side shipped a graph-corrupting bug that every structural test passed and
  only a *recall* gate caught (§7.28); item 15's gate was fooled twice (§1.21).
- Gates before every commit: `gofmt -l`, `go vet ./...`, `golangci-lint run
  ./...` (v2.11.4 here), `go test ./...`, plus `-race` on anything concurrent.
  Note `chunk/treesitter` is a **separate module** — run its gates too.
- Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- `git fetch` before tagging and confirm the tag commit is on `main`. **Both
  boxes push to `main`**, so expect to rebase — it happened three times in the
  amd64 session.

## Keep the docs current

Both are living documents and the agreement is to update them as you go:

- `docs/internal/perf-campaign-2026-07-28.md` — a §7.x entry per item, the
  measured number **and the prediction it missed**, plus the scoreboard row.
- `docs/internal/measuring-performance.md` — a §1.x entry whenever a measurement
  misleads you, a §3 row for any machine constant you establish for the M1 Pro
  (the amd64 section has cross-invocation drift, kernel costs, and the row-split
  worker cap; yours has none yet).

The scoreboard's standing pattern, which has held for 30 items: **derived
predictions hold, extrapolated ones do not.** Item 13's cross-encoder share came
from a closed form and landed; item 22's cost model came from a structural
argument and landed exactly. Every estimate that came from scaling a
microbenchmark missed, usually high — and once (item 27) low by 6×.

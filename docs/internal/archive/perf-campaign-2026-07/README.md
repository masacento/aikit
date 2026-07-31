# Archive — 2026-07 performance campaign

This directory is the **audit trail** of the July 2026 perf campaign, frozen. It is
history, not current guidance. The campaign closed at commit `4034ffd`; the durable
conclusions were extracted into the live docs (below) and consolidated at that point.
Nothing here is deleted — it is kept intact because the §7 results log in
`perf-campaign-2026-07-28.md` records the measured provenance of every number, and a
future reader auditing a surviving claim needs to trace it back to the run that
produced it.

**If you are looking for what is still true, read the live docs, not these:**

| Want | Read (live, in `docs/internal/`) |
|---|---|
| What was tried and did NOT ship, with mechanisms + numbers | **`perf-dead-ends.md`** |
| How to measure without fooling yourself (the failure catalogue) | **`measuring-performance.md`** (§1.35–36 are new from this campaign) |
| CPU/SIMD kernel state (packed GEMM, fused Q8 widen, dispatch) | **`cpu-acceleration.md`** |
| The 6P+2E vs 8C+SMT decomposition + "what transferred" | **`perf-amdahl-apple-m1pro.md`** §5, **`perf-amdahl-linux-amd64.md`** |
| The bm25-persistence (N4/N6) deferral, as an API decision | **`roadmap.md`** §2.14 |
| User-facing durable truths | package godoc (`StaticModel.EncodeBatch`, `fuse.RRF`/`RSF`, `FlatI8.Query`, `bm25.Build`, `SafetensorsFile.ReleaseTensors`) and `CHANGELOG.md` |

## What each archived file was

**Results / audit trail (the reference-of-record):**
- `perf-campaign-2026-07-28.md` — the campaign spine: §2 status table (every item,
  struck when done), §6 dead ends, **§7 the numbered results log** (one entry per item,
  with the measured number and box). This is *the* audit trail; the live
  `perf-dead-ends.md` and the scoreboards point back here for provenance.
- `task-perf-lens-scans.md` — the five subsystem "lens" scans (bm25/ann/chunk/embed/
  examples) with inline per-item annotations, including the §4.2 arm64 arbiter note.
- `task-perf-memoization.md` — the first lens scan (wordPiece memo, interning, the
  StaticModel presum), standalone.

**Handoff RESULT sections (per-box, arbiter of record):**
- `task-perf-handoff-linux.md` — the `nvidia-rtx2070s` (Ryzen 3700X) task doc + RESULTs
  (Phase A/B, lens scans, Phase D footprint/cold-start work).
- `task-perf-handoff-macos.md` — the `apple-m1pro` (M1 Pro) arbiter RESULTs: item 22(b)
  shipped, §4.2 and item 9 measured out, and the M4 arbiter finding that §4.5's
  peak-RSS win is Linux-only (726.2 MiB unchanged here). This is the source of
  measuring-performance §1.35–36.

**Pre-campaign task docs (superseded, kept for lineage):**
- `task-perf-linalg.md` — the "a microbenchmark proposes, an end-to-end run disposes"
  arbiter-discipline precedent (the pulled worker pool). Its discipline is now
  measuring-performance's; its kernel state is cpu-acceleration.md's.
- `task-perf-mblock.md`, `task-perf-pcore-width.md` — the m-block tiling and P/E width
  investigations that fed `SetParallelThreshold`/`SetParallelWidth` and the packed GEMM.

**Campaign prompts (the handoff kickoffs, moved from `docs/prompts/`):**
- `arm64-perf-campaign-handoff.md`, `macbook-phase-a-handback.md` — the two-box handoff
  prompts.
- `aikit-mmap-residency-leaf.md`, `bench-ann-first-slice.md` — the mmap-residency and
  ann-benchmark kickoffs.

(`docs/prompts/cuda-f32-gemm.md` is NOT here — it is native-GPU workstream, not this
CPU campaign, and stays in `docs/prompts/`.)

## Notes on references

- **Intra-archive links** (files here citing each other) use the files' plain names —
  they are all co-located in this directory now, so a co-located `name.md` link resolves. Some archived
  text still names the old `docs/internal/...` path in prose (e.g. the campaign doc
  describing where a companion "was"); those are historical and left intact per
  "archive, don't rewrite."
- **Live docs point in here** with `archive/perf-campaign-2026-07/<file>` paths
  (measuring-performance, both Amdahl docs, BENCH-gpu), and a few `_test.go` comments
  cite the archived handoff/lens docs by that path.
- **Bare-name citations** in package `_test.go` files (e.g. "task-perf-memoization §1b",
  "perf-campaign-2026-07-28.md finding A") name a file, not a path — they resolve here
  by name and were left untouched.

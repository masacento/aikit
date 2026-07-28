# Kickoff prompt — the ANN GPU first-slice benchmark (Phase 2 headline)

> The first real number for the native-GPU substrate, and the sweep that validates the
> whole `docs/BENCH-gpu.md` methodology. ANN is "the one workload with no GPU path at all"
> — an int8 corpus GEMV over the largest single dimension in the system. Do it FIRST,
> before the encoder/ViT benchmarks: it produces the first dispatch threshold and shakes
> out the harness. Read `docs/BENCH-gpu.md` end to end before starting — this prompt is the
> instantiation of it, not a substitute.

## The measurement

**`FlatI8` CPU-SIMD vs `annmetal`/`anncuda` `QueryBatch`, an (N × batch) crossover on REAL
Model2Vec embeddings**, gated on recall + parity against the exact-CPU top-k, with the four
cost components broken out. One machine per run (Metal on Apple, CUDA on NVIDIA) — they
never co-reside, so emit a **per-machine table**, never a fabricated `cpu|metal|cuda` row.

## Wiring (the surface already exists — parity-gated, not yet benchmarked)

- **Real embeddings, not synthetic** — `bench/bench_test.go:realCorpus` is the pattern:
  `embed.LoadFromFS(os.DirFS("../testdata/model"), ".")` then embed a clustered corpus.
  Random high-dim vectors concentrate distances and make recall@k meaningless
  (`benchmarks/README.md`); model-gate the benchmark (`Skip` when the checkpoint is absent).
- **CPU baseline** — `FlatI8` built from the corpus, its pure-Go SIMD `Query` (per-query)
  and the exact top-k it produces are BOTH the recall/parity ground truth and the
  same-box CPU number.
- **GPU path** — import `gpu/annmetal` (or `gpu/anncuda`); its `init()` calls
  `ann.RegisterBackend`. Then `f.EnableGPU()` makes the int8 codes device-resident, and
  `f.QueryBatch(queries, k)` routes through `I8BatchIndex.ScoreBatch` (one dispatch for the
  whole batch — batching is the entire reason the GPU is here). **Device-gate** it: `Skip`
  when `CreateSystemDefaultDevice` fails, so `go test` stays green on a GPU-less box.

## What every row must report (BENCH-gpu.md §2 — do NOT fold these together)

Break out the four cost components; reporting one "GPU time" is the most common way to
mislead here:
- **one-time**: context create, pipeline compile, and **`EnableGPU` residency upload** —
  amortized over a session, measured once, NOT per query.
- **per-launch**: kernel launch / dispatch overhead.
- **transfer**: H2D query upload + D2H score readback — **label it per backend**: on Metal
  (UMA) it is a `copy()` into a shared buffer, ~free; on CUDA (discrete) it is an explicit
  `WriteInt8s`/`ReadFloats` copy. Same code path, different cost — say which.
- **steady compute**: the `ScoreBatch` GEMV itself, warm.

Benchmark the way the code runs: **resident** codes (`EnableGPU` before `ResetTimer`),
**warm** (≥N warmup batches), **batched** (`QueryBatch`, never a single-query loop). Include
D2H only when the result must land on host — the top-k selection needs the scores, so it does.

## Couple perf to parity (BENCH-gpu.md §3) — and hold precision fixed

`annmetal`/`anncuda` compute the SAME int32 dot as CPU `linalg.MatmulBTW8A8`, so GPU and CPU
**rank identically**. Refuse to emit a perf number for any (backend, N, batch) whose
recall-vs-exact-CPU-top-k is not within the documented tolerance — the recall check is
simultaneously the parity gate and the quality metric. Both sides are int8 here, so precision
is already controlled; state it.

## Deliverable

1. A device-gated benchmark (`ann/*_bench_test.go` or under `gpu/annmetal`) sweeping
   **N ∈ {1e4, 1e5, 1e6} × batch ∈ {1, 8, 32, 128}** at a fixed `dim`/`k`, emitting the four
   components + queries/s + recall@k per point, and **the crossover N×batch** where GPU
   overtakes CPU — that crossover IS the deliverable (the GPU analogue of `linalg`'s
   `parThreshold`), not a single speedup headline.
2. A short results note (per-machine table + the threshold). If the record-schema emitter
   from BENCH-gpu.md exists by then, emit `records.jsonl` rows and let `bench report`
   generate the table; if not, a hand-written per-machine table is fine for this first cut —
   just don't invent a cross-machine row.

## Guardrails

`CGO_ENABLED=0`; GPU rows device-gated + model-gated so default CI stays green (GPU perf is a
documented periodic pass on self-hosted runners, never CI — see BENCH-gpu.md). Don't touch the
cross-tag device files. Report the achieved recall bound and the measured crossover.

Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

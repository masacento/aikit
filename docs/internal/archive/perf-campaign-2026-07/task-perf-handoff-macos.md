# Phase D handoff — macOS / M1 Pro (the arbiter box)

This doc holds the M1 Pro's Phase D results. It is the arbiter-of-record side; the
Linux/amd64 box writes `task-perf-handoff-linux.md` and `perf-amdahl-linux-amd64.md`.
Do not consolidate the two.

Box: `apple-m1pro` (measuring-performance §3) — Apple M1 Pro, 6 P-cores + 2 E-cores,
no SMT, macOS. Method: `-benchtime 2s -count=6+`, min-of-N, spread ≤3% on large
stages (noisier on full-forward benchmarks; isolate the stage when resolving <5%).

---

## RESULT — Phase D, macOS/M1 Pro (2026-07-31)

Four scope items (M1–M4). **One shipped, two measured-out-and-reverted, one arbiter
pass with a footprint finding that overturns the amd64 story.** The through-line is
this box's two macro-architectural facts: the **6P+2E fork/join tax** (Amdahl §3) and
**macOS never dropping clean file-backed pages on `madvise`** — between them they sank
both footprint-shaped items and one kernel-shaped one.

### M1 — Item 22(b): fuse the Q8 dequant into `packedFill`. **SHIPPED.** ✅

The one genuinely open kernel item, arm64-only. `packedFillQ8` widens int8→f32 straight
into `packedFill`'s ≤32 KB L1 pack tile, so the up-to-9.4 MB f32 weight matrix never
materializes and the ~0.9 GB/forward `deqW` DRAM round-trip is gone.

| L | dequant (fix a) | fused (fix b) | |
|---|--:|--:|--:|
| 8   | 43.0 ms | **31.0 ms** | **−27.9%** |
| 64  | 143 ms  | **128 ms**  | −10.5% |
| 256 | 606 ms  | **577 ms**  | −4.8% |
| 512 | 815 ms  | **799 ms**  | −2.0% |

1/M-shaped (the widen is fixed per forward), so it helps most at the short inputs a
reranker sees. Bit-identical (`TestMatmulBTQ8Fused_bitIdentical`, exact over all four
shapes × M∈{1,2,3,10,80,91,357}, serial==parallel, mutation-checked); cosine vs f32
0.997; `CGO_ENABLED=0` clean. **The trap, recorded because the naive cut measured
+158% at L=256:** `packedFill` has no m-blocking (deferred item 23), so forcing it on
the K=768 projections re-streams the a-panel N/8 times. The fix was not a K threshold
but **mirroring both parallel axes** — columns for small M, ROWS for large M — so under
the row split each worker runs the fused kernel over a small row block and the
m-blocking gap never opens. See campaign §7 finding 38.

### M2 — Lens §4.2: gate column-blocking in `swigluMLP`. **MEASURED OUT, reverted.** ❌

Column-block the gate into `val[:, j0:j1]`, dropping the full `[L,I]` gate buffer to one
jb-wide tile. Bit-identical (`TestSwigluMLP_colBlockBitIdentical`, I∈{256,3072,3080,128}
× L∈{1,7,32,80}, mutation-checked) and the footprint win is **real and measured here —
gate scratch 44.0 → 3.67 MB/worker** (jb=256). But it is **not** latency-neutral on
arm64 as it was on amd64 (2126→2123 ms):

| jb | isolated `swigluMLP` L=3584, min-of-8 | vs full-gate |
|---|--:|--:|
| full gate | 212.4 ms | — |
| 256 | 224.3 ms | +5.6% |
| 512 | 219.9 ms | **+3.5% (best)** |
| 768 | 227.0 ms | +6.9% |
| 1024 | 223.7 ms | +5.4% |

The amd64 "free" reading is spent by the **6P+2E fork/join**: the full gate is one
parallel matmul, the blocked gate is I/jb of them, each barrier paying the E-core
straggler tax. That is the same latency-for-footprint tradeoff §4.2 rejected the
row-tile variant for (6%), so by the lens's own bar it does not ship here. The full
batch forward was +0–6%, too noisy to resolve below that — the isolated stage is the
number. Reverted (`forward_q8.go`/`scratch.go` byte-identical; `mlp.go` keeps a
pointer). amd64 may still ship it. Campaign §7 finding 39; lens §4.2 annotated; amdahl
§5 "did not transfer" row.

### M3 — Item 9: `Touch(b+1)` pipelining in `scorePaged`. **MEASURED OUT, reverted.** ❌

A caller-side one-block-ahead WILLNEED (a state-free `SpanCache.Prefetch` hinting b+1
while scoring b — scores byte-identical, parity gate green) measured **+12% SLOWER**:
budget 64/1875 blocks, base 1.95 ms/query → prefetch 2.18 ms. On darwin `MADV_DONTNEED`
is a no-op for RSS, so "evicted" blocks stay resident and every prefetch WILLNEED is a
syscall on an already-resident page — pure overhead, no fault to hide. The win it targets
is Linux-only (where DONTNEED reclaims), i.e. **unmeasurable on the only box available**;
and even granting Linux, one block (~24K int8 MACs, ~µs) is too little compute to cover a
~10 µs fault without a multi-block-ahead window. Not shipped; kept as a `scorePaged`
comment. Campaign §7 finding 31 (extended). **Sub-item** (`mmap` never advises WILLNEED
on map; no `Advise` in `ann/`): not attempted — measuring an eager-fault-in win needs the
larger-than-RAM index §1.23 asks for, which this box cannot exhibit for the same
DONTNEED-no-op reason, so per the handoff I state I could not measure it rather than ship
it blind.

### M4 — Arbiter pass on the Linux box's Phase D footprint work.

The Linux box's headline Phase D number is **§4.5 `LoadWeightsQ8` peak RSS 727.6 → 242.3
MiB (3.00×)** via `ReleaseTensors` = one `MADV_DONTNEED` per f32 tensor after quantizing
it. **On the representative laptop this win DOES NOT EXIST.** Measured (getrusage
`ru_maxrss`, this box):

| stage | amd64 (released) | amd64 (would-be unreleased) | **M1 Pro (as shipped)** |
|---|--:|--:|--:|
| peak RSS after `LoadQ8` | 242.3 MiB | 727.6 MiB | **726.2 MiB** |

The M1 Pro number lands on the amd64 *unreleased* figure, not the released one:
`ReleaseTensors`→`mmap.Advise(DONTNEED)` is a documented no-op on darwin
(`madvise_darwin.go`), so the 547 MB f32 checkpoint stays resident until `Close`. It is
clean and file-backed, so macOS *will* reclaim it under pressure (no OOM, the "runs on a
laptop" guarantee holds) — but the **peak-RSS number the footprint story quotes is a
Linux-only artifact**. This ties §1.32 (a footprint claim names a quantity — and RSS vs
heap vs "reclaimable" are three different ones) to the item-9 finding: on this box the
entire DONTNEED-based footprint lever is inert. Anyone quoting "3.00× peak RSS" as a
laptop number is quoting the wrong machine.

(The Linux box's other Phase D pieces — `LoadMmap` heap 5.8×, W3 cold-start reframing N4
to 21.5% — are heap/time quantities that do transfer in kind; only the DONTNEED-based
RSS drop is machine-specific. Not separately re-benched here; the RSS one is the finding.)

---

## Negatives (things built and taken out)

- **§4.2 gate column-blocking** — +3.5–6.9% arm64, footprint-only; reverted (M2 above).
- **Item 9 `Touch(b+1)` prefetch** — +12% arm64, Linux-only benefit unmeasurable; reverted (M3).
- **§4.5 peak-RSS release** — not ours to revert (Linux-owned, shipped), but *arbited as
  non-transferring* on darwin: 726.2 MiB, no improvement (M4).

The pattern for this box: **kernel/instruction fixes that keep everything L1-resident
transfer and win (22b); footprint and paging fixes that lean on `madvise` reclaim are
inert or negative on macOS** — DONTNEED does nothing, and the extra parallelism footprint
restructuring introduces pays the 6P+2E barrier tax.

## Suite status

`go build ./...`, `go vet ./...`, `golangci-lint run ./...` (0 issues), `go test ./...`,
and `CGO_ENABLED=0 go build ./...` all green on `phase-d-arm64` at the M2 commit. `-race`
green on the touched packages (linalg/encoder Q8 paths, ann/mmap). `scripts/` parity pins
(`pin_*.py`) unchanged — no checkpoint or golden regenerated this round; the bit-identity
gates run against the committed testdata, and `TestModelQ8_cosineMatchesF32` holds at
0.997. Tree clean apart from this doc.

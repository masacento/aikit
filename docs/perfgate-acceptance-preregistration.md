# perfgate acceptance — pre-registration (amd64 / nobara)

**Written 2026-08-13, BEFORE the run exists.** The criterion, the reconstruction, and every
branch are fixed here so that a marginal result cannot be read toward the answer after the
fact — the same discipline as goinfer's A/B pre-registrations
(`goinfer/docs/measurements/aikit-v1.17.1-*-ab.md`), and the reason perfgate itself never
compares against a stored baseline.

## Why this is owed

`tools/perfgate` exists to catch the class of regression that shipped in aikit **v1.17.0**: a
numerically-identical W8A8 kernel ~3–5% slower at streamed shapes, green the whole way. Its
plumbing is verified on an Apple M1 Pro — working tree vs the previous tag comes back flat,
and a reconstructed access-pattern regression drives every shape to REGRESSION, so the
interleaving, the characterization floor, the branches, and the verdict all work.

What is **not** yet shown is that it catches the **~5% class it was built for**, because the
faithful v1.17.0 reconstruction is **amd64-only**. v1.17.0's regression was the AVX2
eight-column kernel `dotI8x8AVX2`, which advances eight column-streams K bytes apart and
defeats the hardware prefetcher once B is streamed from DRAM. On arm64 `dotI8Cols8` fell back
to a sequential `dotI8Cols8Generic` and **never regressed** — so an arm64 reconstruction
cannot reproduce the effect at its true magnitude (a scalar 8-stream loop is slow everywhere,
not "fast-resident / slow-streamed", which is a different signature).

Until this acceptance runs, **a green from perfgate is not evidence at the 5% class** — only
that no regression larger than each shape's derived floor was seen. Same two-box shape as
`gpudevice`: the gate runs, and its verdict has a stated scope.

## The reconstruction (faithful, amd64)

On the nobara box (Ryzen 7 3700X, linux/amd64), on a throwaway branch off the commit under
test:

1. Restore v1.17.0's eight-column kernel and its dispatch:
   ```
   git checkout v1.17.0 -- linalg/doti8cols.go linalg/doti8cols_amd64.go \
                           linalg/doti8cols_other.go
   ```
   (plus any `linalg/doti8cols_amd64.s` if present at that tag — restore every `doti8cols*`
   file v1.17.0 shipped, so `dotI8x8AVX2` and its assembly come back intact).
2. Replace the body of `w8a8Span` in `linalg/quant.go` with v1.17.0's eight-column form
   (`git show v1.17.0:linalg/quant.go`, the `w8a8Span` that loops `for ; j+8 <= j1; j += 8`
   and calls `dotI8Cols8`). Keep everything else in the working tree.
3. Confirm it still passes correctness: `go test ./linalg/ -run 'W8A8|Matmul' -count=1`.
   (The kernel is numerically identical — integer accumulation, order-independent — so a
   correctness failure means the reconstruction is wrong, not that the regression is real.)

The comparison arm is the previous tag, `v1.17.1`, which carries the reverted linear span.

## The run (fixed here)

```
go run -C tools ./perfgate --visits 8 --benchtime 500ms v1.17.1
```

`--visits 8 --benchtime 500ms` rather than the defaults: nobara is quiet, and more/longer
samples tighten the characterization floor so the streamed shapes have a chance to resolve the
5% class. The instrument is `BenchmarkW8A8SpanShapes` (both regimes) + `BenchmarkGEMV_W8A8_baseline`.
The streamed shapes of interest are **K768_N200000** and **K3584_N18944** (B ≫ the 32 MB L3);
the resident shapes are where v1.17.0 looked like a **+30% win**.

## Branches — all fixed now

1. **The streamed shapes go RED (REGRESSION), at floors ≤ 5%.** → **ACCEPTANCE MET.** perfgate
   catches the class it was built for, on the architecture where the incident happened. Paste
   the VERDICT + the per-shape `Δ`/`floor`/`covers` lines into this file as the result. The
   resident shapes may come back flat or *faster* — that is expected (the eight-column kernel
   genuinely wins there) and is **not** a failure of the gate.

2. **The streamed shapes come back flat because nobara's floors are ALSO > 5% on them** (the
   `covers = >5% BLIND` marker fires there). → This is a **finding about the INSTRUMENT**, not
   a pass or a fail of the gate: even on a quiet box the streamed shapes are too noisy to
   resolve the 5% class at these settings. Owed follow-up: raise `--visits` / `--benchtime`,
   or enlarge the streamed N, until the streamed floor drops below 5%, then re-run. Record the
   floors observed. Do **not** report acceptance as met.

3. **The streamed shapes come back flat WITH floors ≤ 5%** (the gate had the sensitivity and
   still did not fire). → The reconstruction did not reproduce the regression on this box:
   investigate before concluding anything about the gate. Likely causes — the eight-column
   kernel was not actually wired (check `w8a8Span` really calls `dotI8Cols8` and the amd64
   dispatch selects `dotI8x8AVX2`), or the regression magnitude on this exact CPU is below the
   floor. This is a reconstruction problem to resolve, not a gate result to publish.

4. **A resident shape goes RED.** → Unexpected: v1.17.0 was *faster* at resident shapes. A red
   there means the reconstruction is wrong (e.g. a scalar loop that lost the SIMD entirely, as
   the arm64 mechanism-check did). Fix the reconstruction and re-run.

## What this does not decide

Nothing here says anything about prefill, other quantizations, or the GPU path. It is a single
question: **does perfgate, on the box where v1.17.0 regressed, go red on the streamed shapes at
a floor tight enough to matter?** Branch 1 answers yes; branches 2–4 are each a specific,
pre-named "not yet", not a hedge written after seeing the number.

---

# Result — appended after the run on nobara

_(pending — the run is owed; hold until the box is free.)_

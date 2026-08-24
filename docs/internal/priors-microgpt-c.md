# External priors — microGPT-C (2026-08-19)

Source: [`vixhal-baraiya/microgpt-c`](https://github.com/vixhal-baraiya/microgpt-c)
(reviewed at `38485b7`; the material is `src/microgpt.c` + `docs/PERFORMANCE.md`).
Reviewed on a goinfer question — "can goinfer take anything from this?" — and the
CPU-kernel-relevant residue lands here because the kernels live in `linalg`, not goinfer.

**Charter.** Everything in this file is an EXTERNAL prior: measured by someone else, on
someone else's boxes (Apple **M5 Pro** / NEON and **Ryzen 5 5600H** / AVX2 — neither is
`apple-m1pro` or `nvidia-rtx2070s`), on a 4,192-parameter model. Nothing here is an aikit
result and nothing here goes into a kernel-numbers table. A prior that gets acted on is
measured on our boxes under `measuring-performance.md` rules, and the RESULT files where
results file — `perf-dead-ends.md` if negative, `cpu-acceleration.md` if it ships. This
doc then points at that entry rather than restating it.

## 0 · Why most of it does not transfer — the regime line

microGPT-C's entire weight set (~16 KB) is L1-resident, and its perf doc demonstrates the
decode loop is **issue-width-limited**: injected independent FMAs cost 0.20 cycles each
against a 0.19 theoretical floor — no idle slots, so every technique in the file is about
issuing fewer instructions. aikit/goinfer batch-1 decode streams weights from DRAM and is
**bandwidth-bound** (BENCH-gpu.md; `roofline-2026-08.md`), where the lever is bytes moved,
not FMA scheduling. Their own dead-ends table confirms the divide from the far side: fp16
weights halved the bytes and moved the MLP by one cycle in 241 — bytes were not their
bottleneck, exactly as bytes ARE ours.

The exception that makes the priors below worth keeping at all: aikit's small-K /
L1-resident corners sit in microGPT-C's regime, and we have already met it once —
`dotNEON2x8` at ~95% of FMLA peak on `apple-m1pro` (cpu-acceleration.md §Status) is an
issue-width-limited kernel, found by comparing against peak. Their diagnostic (§1) gets to
the same verdict without needing to know peak.

## 1 · Instrument: marginal-FMA injection as an issue-width probe

Their method: add N independent (dead) FMAs to the hot loop and measure the marginal cost
per added instruction. If it lands at the core's theoretical issue floor (theirs: 0.20 vs
0.19 cy/FMA), the kernel is issue-limited — the machine is busy, not waiting — and further
scheduling/unroll/load-reduction work is provably dead before it is built. A gap between
marginal cost and floor means stalls exist and are worth hunting.

Why keep it when the %-of-peak comparison already exists: it needs no peak model of the
box (no FMA-pipe count × width × frequency arithmetic, which is exactly where the
`perf-amdahl-*` decompositions burned time), and it distinguishes "busy" from "waiting"
directly rather than by residue. Candidates when next touched: dequant inner loops, the
norm/rope-shaped small fixed-size kernels, any kernel a profile says matters at small K.
One bench file, in-process A/B per measuring-performance §0.3. Until someone runs it on
our boxes it stays a prior, not a practice.

**Run, 2026-08-19 — `linalg/fma_issue_probe_test.go`, `TestFMAIssueProbe`.** Applied to
`dequantRowInt8` (the "dequant inner loops" candidate above; the norm/RoPE candidate
doesn't exist in `linalg` — that lives in `encoder` — and "small K" wasn't a citation of
an existing finding, so neither was forced). Compared the same dead-FMA loop's marginal
cost alone vs stacked on the kernel, in-process, rather than against a theoretical
floor — sidesteps needing a peak model at all, closer to the spirit of why this technique
was worth keeping over the %-of-peak comparison. Result and both boxes' numbers now live
in `cpu-acceleration.md`'s Open follow-ups item 6, not restated here: not issue-limited on
either `apple-m1pro` or the `nvidia-rtx2070s` host's Ryzen/AVX2. This item is now a
practice, not a prior.

**Demotion, 2026-08-23 — this instrument is now 0-for-2 as a load-bearing decision input; treat
it as a hint, never a verdict.** Two independent campaigns have now trusted a single reading from
this probe and been wrong in different ways:

- **AVX2 (`perf-dead-ends.md` §8.9, `nvidia-rtx2070s`):** ratio 0.91 read as "not issue-limited" →
  built a 4-accumulator kernel expecting a real win → measured ~1%, noise. The probe wasn't
  *wrong* about FMA-port occupancy, but it answered a narrower question than it was read for: idle
  FMA-port capacity doesn't rule out a *different* port (here, the unpack prologue's shuffle ops)
  being the actual bottleneck. A port-class blind spot.
- **W4A8 NEON (goinfer `docs/task-w4a8-neon-bandwidth.md`, `apple-m1pro`, Gate 0 → item-3 harness
  results):** ratio 1.11 read as "issue-limited" → motivated a whole grid of instruction-removing
  kernel variants → the first one measured a flat 1.000x. Re-running the SAME probe 4 times on a
  settled box gave ratio 0.99-1.03 every time — the 1.11 reading does not reproduce. Likely cause:
  that measurement's own session ran a STREAM bandwidth probe that had *just* been caught
  "accidentally measuring swap on a box with other sessions running" — the box was loaded when
  the issue-width probe ran too, and a loaded box is the plausible culprit both times this
  instrument has misled: measurement noise reads as a false verdict either direction, not just as
  a wider error bar.

**Consequence: this probe is a hint, never load-bearing.** Its correct use going forward is to
motivate trying a fix cheaply (as intended, per the paragraph above), never to justify skipping a
direct A/B measurement of the fix itself, and never to be read as ruling a mechanism out on one
reading alone (the arm64 case above is the fix that DID work — dotW4A8FoldSDOT2Acc measured a real
1.4-1.75x — precisely the mechanism a mistrusted "not issue-limited"-shaped reading would have
told someone not to bother building). Before citing a ratio from this probe in a decision doc:
re-run it at least twice on a settled box, and treat disagreement with a direct kernel A/B as the
A/B winning, always.

**Extension, 2026-08-19 — the norm/RoPE candidate this doc said "doesn't exist in `linalg`"
turned out to exist in goinfer** (`decoder/rmsnorm.go`, `decoder/rope.go`) — flagged after a
scoping correction, since this doc's own charter line only scoped the search to where the
CPU-kernel *residue* lands (`linalg`), not to where every candidate the source project named
actually lives. Same probe, same file shape, run in goinfer as
`decoder/fma_issue_probe_test.go`. First pass at the original N∈{0..64} sweep did not
reproduce — both kernels flipped verdict run to run, because these kernels (~1.1-2.7us) are
an order of magnitude bigger than `dequantRowInt8` (~35-50ns), so N≤64 added too small a
fraction of runtime to clear jitter. Widening to N∈{0..512} fixed it: both `rmsNorm` and
`applyRoPE` reproducibly land at ratio ~1.0 — NOT issue-limited on `apple-m1pro`, unlike
`applyRoPE`'s brief false-positive at the narrow sweep. The Ryzen/AVX2 run (`nobara`, once
its goinfer 1.0 prep sweep cleared) agrees: both kernels ~1.0, not issue-limited, on both
boxes. Full numbers live in goinfer's `docs/measurements/aikit-fma-issue-width-probe.md`, not
restated here.

## 2 · Porting caution: accumulator-chain counts do not survive arm64 → amd64

Their finding, stated as mechanism: an accumulator set is 16 floats — 4 registers and 4
independent chains on NEON, but 2 of each on AVX2, so the SAME source hands x86 half the
chains; and arm64 has 32 vector registers to amd64's 16, so register-hungry choices that
win on NEON go flat or negative on AVX2. Measured on their side: raising Wo/lm_head from
4 to 8 chains on the 5600H — flat.

This independently corroborates a finding we already own: the `Dot8x4` K=3072 cliff
(cpu-acceleration.md, AVX2 table ⚠ + follow-up §1), where 8 live YMM accumulators plus
streamed b-rows exceed what stays hot. The rule the two findings share, worth stating
once: **an arm64-tuned chain/blocking count is a hypothesis on amd64, not a port** —
re-A/B it on the Zen 2 box before trusting it. Applies forward to any NEON-first kernel
that grows an amd64 twin (the w4a8 pair is the shape).

## 3 · Imported dead-end priors (theirs, not ours — do not cite as aikit results)

Kept because each has mechanism + number and would plausibly be re-proposed here:

1. **fp16 multiply / fp32 accumulate (FMLAL):** halves weight bytes, not op count; one
   cycle out of 241 in their compute-bound loop. (In OUR bandwidth-bound regime the same
   trade points the other way — which is the regime line of §0, not a contradiction.)
2. **fp16 accumulation:** does halve the op count, gained 5% — at three orders of
   magnitude of logit accuracy. Dead on arrival under parity-pinned numerics (design rule
   3, architecture.md) regardless of speed.
3. **Newton-refined reciprocal / rsqrt:** 2–4% SLOWER than hardware `fdiv`/`fsqrt` —
   five instructions against one. Prior for anyone tempted inside an RMSNorm-shaped loop.
4. **Breaking the token-serial dependency:** replayed a recorded trajectory so
   consecutive tokens were independent — no faster. At batch 1 nothing is waiting on the
   inter-token dependency; the serial structure of decode is not itself an issue-slot cost.

## 4 · Considered against goinfer/aikit and rejected on inspection (no measurement owed)

Recorded so the ideas are not re-derived from the same source later:

- **Hoisted `(token,pos)` layer-0 table** (their `build_pretok`): needs
  `vocab × block_size` tiny; at a ~150k vocab the layer-0 QKV table is hundreds of MB,
  and RoPE moves position into the rotation anyway. Does not scale off the toy.
- **Deferred RMS scale** (factor the norm scalar past the MLP): requires a positively
  homogeneous activation — holds for their ReLU², fails for SwiGLU/GeGLU, and real norms
  carry learned per-channel gains. Architecture-dead for every model family we serve.
- **Bit-trick fast exp** (`fexpf`/`vfexpq`): changes every draw by far more than a ULP;
  collides with goinfer's summation-order and machine-independence contracts
  (`sampler_chunked.go`). goinfer's P2b already took the other road — exact exp, made
  parallel and deterministic.
- **Column-major packing to eliminate horizontal reductions:** decisive at nin=16 where
  the hadd is a visible fraction of the row; at real widths the per-row reduction
  amortizes to noise (dot_arm64.s already prices it at ~1 ns/row). Row-dot layout stands.
- **Fused sampling over unnormalized weights:** already shipped in goinfer (P2b), with
  the determinism guarantees the C version does not need single-threaded.

Their closing observation — batching is what lifts the ceiling, because it turns matvec
into matmul and gives weights reuse — is the premise `MatmulBTW8A8Batch` and goinfer's
block-drafting work already stand on. Corroboration, not a lead.

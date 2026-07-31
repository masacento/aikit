# Prompt — aikit perf campaign: Phase A hand-back to the M1 Pro

> **For:** the aikit session on the Apple M1 Pro.
> **From:** the Ryzen 7 3700X (Zen 2, 8C/16T, AVX2, no VNNI), which has finished
> [`task-perf-handoff-linux.md`](../internal/task-perf-handoff-linux.md) — Step 0
> and all of Phase A, plus two items from outside it that Phase A promoted.
>
> **You are the arbiter of record.** Every figure below is amd64 and exists to
> rank work, not to be quoted. The CHANGELOG numbers should be yours.

## Read first

1. [`docs/internal/perf-amdahl-linux-amd64.md`](../internal/perf-amdahl-linux-amd64.md)
   — the new Amdahl table and every Phase A result, with what each prediction
   missed and why. **This is the map.**
2. [`docs/internal/measuring-performance.md`](../internal/measuring-performance.md)
   §1.29–§1.31, added during this phase.

---

## 1. Your first job: the same table, on your hardware

Step 0c built the two workload benchmarks that did not exist, and they are
portable — they need only `testdata/model`:

```sh
go test ./bench/ -run XXX -bench 'BenchmarkW1|BenchmarkW2' -benchtime 2s -count=6
go test ./embed/ -run XXX -bench BenchmarkEncodeSplit      -benchtime 2s -count=6
go test ./bench/ -run XXX -bench BenchmarkW1/sum -benchtime 6x -cpuprofile w1.prof
```

Write the M1 Pro table into `docs/internal/` beside the amd64 one. Do not edit
the amd64 one — two tables, comparable, is the deliverable; one averaged table is
not.

**Three things to check specifically, because they are where the boxes should
disagree:**

- **A1's concurrency curve is the one figure this box could produce cleanly and
  you cannot.** 8 homogeneous cores + SMT gave 91% of linear at 2 workers, 87% at
  4, 78% at 8, then a further 1.31× from SMT to 16 — **8.21× at NumCPU**. Your
  6P+2E asymmetry will bend that curve, and where it bends is worth recording:
  `runtime.NumCPU()` is 8 on your part and the E-cores are the straggler. If
  `concurrency = NumCPU` is not the best setting on Apple Silicon, that belongs
  in `EncodeBatch`'s doc comment.
- **A2 and A4 both pay a `utf8.ValidString` scan** to be exact on invalid UTF-8
  rather than gating on the `cleanText` config flag. Measured at **1%** of the
  sliced path here. If it is materially worse on arm64, say so — the fallback
  design survives, the cost does not have to.
- **A5's eight non-`fuse` sort sites measured at ZERO here** (`FlatI8.Query`
  −2.9%, `bm25.TopK` +2.5% — opposite directions, both inside drift). The
  mechanism is real in isolation (2.95× at n=50); it does not show because the
  real sites sort a heap array that is already partially ordered. Check whether
  that holds on your part before anyone quotes the lens doc's "3.3–4.6× each".

---

## 2. What landed, so you know what you are re-measuring

| item | measured here | predicted | note |
|---|---|---|---|
| **A1** `StaticModel.EncodeBatch` | **8.21×**, 3.53× on a whole index run | 4–6× | new exported API |
| **A2** added-token carve-out | 1.22× on the stage | 26.5× | see below |
| ~~A3~~ `wordPiece` memo | — | — | **was already landed** |
| **A4** `preTokenize` slicing | **1.95×**, 43× fewer allocs | 2.65× | |
| **A5** `fuse.RRF`/`RSF` | **4.80×**, −20% of a query | 2.18× | 8 other sites: zero |
| **A6** `NewFlatI8` streaming | **−79.7% bytes**, 1.17–1.27× | 2.02× | memory item |
| ~~B~~ `dotI8AVX2` | — | — | **was already landed** |
| lens §3.7 `bm25.Build` | **1.27×** | 1.33× | not in Phase A |
| lens §3.1+§3.2 chunker | **1.76×** | 1.62× | not in Phase A |

An index run: **382.73 ms serial → 98.08 ms batched, 3.90×.**

**Two of the eight Phase A items were already done** — A3 and B, both landed
before the handoff was written, both at or above their predicted ratios. Check
each item against the code before pricing it; the handoff describes the tree as
of when it was written, and this cost time twice.

**A1 reordered everything after it.** The embed stage fell from 77.8% to 33.0% of
an index run, which promoted `bm25.Build` (27.6%) and the chunker (18.7%) above
A2 and A4 — two items from outside Phase A outranking two inside it. Expect the
same re-ranking on your box and re-derive it from your own table rather than
reusing this one.

---

## 3. Corrections to the source documents, so they do not mislead you next

Three claims in the lens/handoff docs did not survive measurement. They are
recorded in the commits, but you will read the docs first:

- **`regexp.LiteralPrefix()` does not do what lens §3.2 says.** It lists
  `^func\b`→`func`; it returns `""`, because it reads the *compiled program's*
  extracted prefix and a following `\b` or `\s` blocks the extraction. It is
  empty for **all four of Go's definition patterns**. The prefix has to come from
  the `regexp/syntax` tree instead — that is what `anchoredLiteralPrefix` does,
  and it lifted coverage from 3 patterns to 7 for Go.
- **A2's 26.5× was the loop in isolation.** In place the loop still hands its
  segments to `encodeSegment`, so the carve-out measured ~4× in situ and 1.22× on
  the stage.
- **A6's time figure was 1.6× optimistic while its memory figure was exact** to
  three digits. It is a footprint item; land it on those terms.

---

## 4. What is still open

**Arm64-only, and yours alone:**

- **Item 23** — `packedFill` lost `blockedFill`'s m-blocking, so the a-panel is
  re-read per 8-column group. Never measured on any box. `packedFill` is gated on
  `has2x8Kernel`, false off arm64.
- **Item 24** — the packed stride is a 4096 B power-of-two. Note the campaign doc
  already records that the proposed "+4 float pad" is **too small to work**: four
  floats is under a cache line, so it cannot move a row to a different L1 set. Use
  ≥16 floats. It may also be a no-op on your specific part — measure before
  building, and a flat sweep is a legitimate answer.

**Architecture-neutral, left undone deliberately, with the reason:**

- **MaxScore for long queries** (item 39). WAND's pivot loop costs O(query terms)
  per iteration, so it stops paying between 6 and 12 terms and is guarded at 8.
  That guard sends every real SPLADE query down the exhaustive path — MaxScore is
  the algorithm for that case, and it is why `sparse`'s WAND was reverted.
- **A first-byte SET for the chunker's TypeScript rules.** Their patterns lead
  with `(export\s+)?`, an optional group, so no literal prefix exists and they get
  no prescreen: TypeScript gained 1.14× where Go gained 1.68×. Bounding them needs
  a first-byte set computed from the syntax tree rather than a prefix.
- **Item 40**, flash-attention / online-softmax tiling — the last untouched Tier 3
  item, and `≠` (numerics change).

**Deferred out of the campaign entirely — do not pick these up as perf work:**

- **lens N4** — `bm25.Index` serialization, and **lens N6** — the `[][]string`
  seam in `bm25.Build`. Both are decisions about `bm25`'s input/output surface,
  not optimizations, and they are entangled: a streaming `Builder` and a versioned
  on-disk format constrain each other. N4's headline "67% of cold start" was
  re-measured at **21.5%** once the real checkpoint load was in the denominator
  (17.6 ms), which is not enough to justify a permanent compatibility promise on
  a benchmark's say-so. If someone takes them, they take them as an API proposal
  before the v1.0 surface freeze, starting from the format and the version-skew
  policy. **This is the whole remaining backlog of the three perf docs** — with
  N4 and N6 deferred, the campaign is closed.

**Not reachable anywhere:** item 35 (VNNI, needs Zen 4+/Ice Lake+), item 36
(i8mm, needs M2+/Graviton3+).

**If someone revisits `VPMADDUBSW`** for the amd64 int8 kernel: the route is
`VPABSB`/`VPSIGNB` — |a| as the unsigned operand with a's sign folded into b —
which keeps every pair sum under 2·127² = 32258 and so cannot saturate. That is
what the existing kernel comment's "needs range-limited codes" means concretely.

---

## 5. Non-negotiables (unchanged)

- **`CGO_ENABLED=0` throughout**, and assert `CgoFiles=[]`. `-race` needs cgo;
  run it explicitly with `CGO_ENABLED=1` on the concurrent packages.
- **Do not touch `//go:build linux` / CUDA files.** I have not touched the darwin
  ones.
- **Parity gates are the acceptance criterion, not the benchmark.** Anything
  claiming bit-identity gets exact equality over the WHOLE corpus and the gate
  gets **mutation-checked**. Every item above shipped with its dead mutants named
  in the commit, and three gates in this phase were found to prove nothing until
  they were strengthened — see §1.29–§1.31.
- Gates before every commit: `gofmt -l`, `go vet ./...`, `golangci-lint run ./...`,
  `go test ./...`. `chunk/treesitter` is a separate module — run its gates too.
- Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- `git fetch` before tagging; confirm the tag commit is on `main`. **Both boxes
  push to `main`** — expect to rebase.

## 6. Keep the docs current

- `docs/internal/perf-campaign-2026-07-28.md` — a §7.x entry per item, the
  measured number **and the prediction it missed**, plus the scoreboard row.
- `docs/internal/measuring-performance.md` — a §1.x entry whenever a measurement
  misleads you, and a §3 row for any machine constant you establish.

The scoreboard's standing pattern held again through Phase A: **derived
predictions hold, extrapolated ones do not.** A6's memory figure came from a
structural argument (the staging block is n·d·4 bytes) and was exact; its time
figure came from scaling a microbenchmark and was 1.6× out. Same item, both
kinds, one document.

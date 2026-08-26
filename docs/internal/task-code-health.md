# Task: aikit code health — the repowise findings, triaged

> Scoping doc. Opened 2026-08-26 from the repowise trial (`docs/prompts/repowise-trial-results.md`),
> the first index+health run over this repo. Sibling to goinfer's `docs/task-code-health.md`, which
> came out of the same run. **Status: §4.1, §4.1a, §4.3 and §4.4 all resolved 2026-08-26 — four items landed, four
> declined with evidence. §4.2 and five §4.4-old complexity findings remain genuinely open.** Unlike goinfer's,
> this list is mostly real: 13 of the 20 targets are production files, not tests. Two hard
> constraints shape it, §3.

Tooling: repowise 0.45.0, `apple-m1pro`, 2026-08-26, `--no-prose` (structural only, no LLM).
Reproduce per goinfer's `docs/task-code-health.md` §5.

## 1 · Where aikit stands

Health average **8.04/10 (Healthy)** · hotspot 4.79 · worst **1.95** (`ann/flat.go`) ·
64.5% healthy by volume (488 files) / 25.0% warning (132) / 10.5% alert (31).
Index: 651 files · 3,738 symbols · graph 5,097 nodes / 15,246 edges · 564 files with git history ·
105 hotspots · 58 decisions · 31 unreachable files / 33 unused exports · 112 perf-risk findings
(18.26 per 10K covered LOC).

Calibration against the trial's other repos: aikit sits second of four on health average, behind
fin (9.21) and ahead of goinfer (7.92) and ken (7.43). Its 15 of 20 lowest-health files had a bug
fix in the last ~6 months — **4.56× the 16% repo baseline**, the same effect that held across all
four repos, so the ranking is worth reading even where individual findings are not.

The five worst files, all with paired tests:

| file | score | CCN | nest | NLOC |
|---|--:|--:|--:|--:|
| `ann/flat.go` | 1.9 | 17 | 4 | 141 |
| `tools/vulncheck/main.go` | 2.0 | 15 | 5 | 173 |
| `ann/hnsw.go` | 2.1 | 19 | 5 | 355 |
| `gpu/metal_vit_test.go` | 2.1 | 15 | 5 | 670 |
| `gpu/anncuda/backend_test.go` | 2.1 | 20 | 7 | 853 |

## 2 · This list is more actionable than goinfer's

goinfer's equivalent run put **14 of 20** targets on test files, which made most of it unactionable.
aikit inverts that: **13 of 20 are production files** — `linalg/quant_w4a8_arm64.go`, `ann/flat.go`,
`encoder/mlp.go`, `ann/flat_i8.go`, `linalg/rowblock_amd64.go`, `encoder/forward_batch.go`,
`tools/vulncheck/main.go`, `linalg/quant.go`, `ann/hnsw.go`, `linalg/matmul_blocked.go`,
`encoder/weights.go`, `tools/consumergate/check.go`, `encoder/forward_q8.go`. Only seven are tests.

That difference is not a quality judgement about either repo — it reflects aikit having a smaller,
denser production surface — but it does mean the findings here are worth reading individually
rather than dismissing wholesale.

## 3 · Two constraints on anything in this list

**3.1 · `linalg/` was off-limits; the freeze LIFTED 2026-08-26.** AVX-512 VNNI kernel work was in
flight and uncommitted in the working tree, authored in a cloud session because neither local box
can execute VNNI — the MacBook is arm64 and the Linux box is Zen 2. It landed in `2a7199a`, so the
conflict is gone. **Four of the 20 targets live in `linalg/`** (`quant_w4a8_arm64.go`,
`rowblock_amd64.go`, `quant.go`, `matmul_blocked.go`), and so do **nine of the fourteen duplication
clusters — 164 of the 227 duplicated lines**, concentrated in `exp_simd.go` (four clusters, the
largest 33 lines across 2 sites and 28 across 3) and `quant.go` (three clusters). That is the bulk
of the mechanical work on this list and it is now available.

One caveat inherited from what just landed: `dotW4A8` is no longer single-valued on amd64 — the
VNNI kernel matches the scalar oracle to 1e-5 but is not bit-identical to the AVX2 one, so hosts
now split by VNNI+VL. Anything touching `quant.go`'s W4A8 side should hold that tolerance, not
tighten to `==`.

**3.2 · The 33 unused exports are NOT free to delete.** aikit is a published library at v1.27.0
with goinfer and ken as consumers, and the two-series module invariant (`tools/gpupins`) means
eight backend modules pin exact versions. Removing an exported symbol is a breaking change
regardless of whether this repo references it — repowise measures *internal* reachability and
cannot see downstream consumers at all. Treat the 33 as an API-surface *inventory*, useful input to
a future major version, and not as a deletion list. Same caution on the 31 unreachable files: the
detector flags `in_degree=0`, which is simply wrong for `examples/`, `benchmarks/`, and
build-tagged GPU backends (it named `benchmarks` and `tools` "zombie packages" — both are real).

## 4 · The work

### 4.1 · `embed/safetensors.go` typed accessors — STARTED 2026-08-26

Four typed tensor accessors (`Float32s`, `Float64s`, `Int64s`, `Int32s`) carry an identical
six-line body differing only in type parameter and dtype string — repowise's largest
non-`linalg` duplication cluster, 21 lines across 4 sites. Chosen as the starting item precisely
because it is outside `linalg/` (§3.1), on the model-loading path rather than a hot loop, and
covered by existing `embed` tests.

**It is not the trivial extraction it looks like.** Each body opens with
`defer runtime.KeepAlive(t.owner)` guarding against a mid-decode `munmap` (§2.5 of this file's own
notes). That line *cannot* move into a shared helper: `KeepAlive` fires when the frame it is
deferred in returns, so a helper's deferred `KeepAlive` would release the mapping before the
caller has used the aliasing slice it just returned. The guard stays in each exported method; only
the dtype check and the `reinterpretLE` call are shared.

See `embed/safetensors.go` and the commit that lands this for the resulting shape.

### 4.1a · The `linalg/` clusters — EXAMINED 2026-08-26, MOSTLY DECLINED

The freeze lifting (§3.1) made nine clusters / 164 duplicated lines available, which looked like
the bulk of the value on this list. Read rather than applied, most of it is **deliberate
duplication that should stay**, and two cases would have caused real regressions.

**`linalg/quant.go`'s four span-dispatch sites (12-line cluster) — DECLINED, would regress a
tested invariant.** `MatmulBTQ8Into`, `MatmulBTW8A8Into`, `MatmulBTW4A8Into` and `MatmulBTQ4`
each carry the same shape: serial short-circuit under `ws.thr()`, else `ws.parallel(N, closure)`.
The code says why it is written twice — *"Serial fast-path calls the named span directly (no
closure → no heap escape → zero alloc, the steady-state decode case). Only the parallel branch
pays a closure allocation."* Extracting a helper that takes a `func(j0, j1 int)` puts a closure on
the **serial** path too, and since the helper also hands it to `ws.parallel`, escape analysis
marks it escaping in both. That is an allocation added to the decode hot path, and it is gated:
`batch_test.go:148` asserts `AllocsPerRun == 0`. The refactor would turn that test red, correctly.

**`linalg/exp_simd.go`'s five clusters (33+28+27+14+8 lines) — DECLINED, the duplication buys the
speed.** `softmaxRowIntoRaw` and `softmaxRowScaledIntoRaw` share their pass structure and differ
by folding a scale into pass 2 — which is the entire reason the second exists (it replaces a
caller's separate O(L²) pass and is bit-identical to that sequence, not an approximation).
Unifying them means either a per-lane branch or a multiply-by-1.0 in the unscaled path, inside a
`//go:build goexperiment.simd` kernel whose measured reason for existing is 2.5–4.6×. Cosmetic
gain, real risk to a validated number.

**Still open in `linalg/`:** `quant.go`'s 24-line and 10-line clusters and `kquant.go`'s 8-line
pair were not examined in that pass. They may be genuine; they are also small.

**The lesson worth carrying:** 164 of the 227 duplicated lines this tool reported sit behind an
explicit comment or a passing test explaining why they are duplicated. The count was never the
finding.

### 4.2 · `embed/safetensors.go` sharded-open pair — NOT STARTED

`OpenSafetensorsShardedMmap` and `OpenSafetensorsShardedFromFS` share ~13 lines of index-parse and
aggregate-setup (205-217 / 243-255). Genuinely duplicated, but the two diverge immediately after:
one mmaps and must `finalizeMmaps` on failure, the other `fs.ReadFile`s and has nothing to unwind.
The shared prologue is real; the extraction is worth doing only if it does not force the divergent
cleanup through a shared abstraction. Medium value, non-trivial.

### 4.3 · `ann/flat.go` — ATTEMPTED AND REVERTED 2026-08-26; the conditional is load-bearing

**The finding: `scanFlat`'s 8-term `||` guard cannot be simplified without costing bounds-check
elimination in the hot loop.** repowise flags it as a *critical* complex-conditional, and it reads
like one — the eight slices are spelled out three times over (destructure, guard, array literal).
Replacing the guard with `for _, v := range group { if len(v) != d { ... } }` is the obvious fix and
it is wrong.

Measured on the generated code, because the box could not resolve it otherwise (see below):

| | code size | instructions | `panicBounds` sites |
|---|--:|--:|--:|
| as-is (8-term `||`) | 1344 B | 463 | **22** |
| range-loop guard, array-indexed call | 1296 B | 444 | **36** |
| range-loop guard, named-slice call | 1216 B | 419 | **36** |

The explicit `||` chain gives the compiler a per-slice `len(v) == d` proof it carries into the eight
`&vN[0]` uses. A range loop over the array proves the same fact to a human and *nothing* to the
prover, so all eight indexings regain their checks — **7 extra bounds checks per 8-vector group in
the library's hottest loop.** The third row isolates the cause: keeping `&v0[0]`-style arguments and
changing *only* the guard still costs 36, so it is the guard, not how `Dot8x4` is indexed.

**Benchmarks could not settle this, which is why the assembly did.** `BenchmarkFlatQuery` on this
box, best-of-6 then median-of-8 warm and interleaved, gave N1k 0.90×, N10k 1.02×, N50k 0.92× — two
directions at once, with N1k's own spread at 119%. A first attempt also compared a cold `before` run
against a warm `after` one, which flattered the change by ~7%; re-running both warm removed that but
not the noise. Do not re-attempt this on a shared laptop.

**Reverted; `ann/flat.go` is byte-identical to its prior compiled form.** The file keeps its 1.9/10.
That score is the honest cost of an 8-way unrolled SIMD dispatch with a defensive ragged-group
guard, and paying it is the right call.

### 4.3-old · original scoping note

Score 1.9. `scanFlat` carries a **critical** complex-conditional finding plus a nested-complexity
and a complex-method finding, and the file is in the top 5% for change entropy. `ann/flat_i8.go`
is in the **top 1%**. These are the query hot paths, so any change is a benchmark-gated change,
not a readability one — `BenchmarkFlat*` and the ANN parity tests are the gate. Real finding,
real cost; do not start it casually.

### 4.4 · Extract-method plans — two done, one declined, rest open

**Done (`resolveUnkID`, `lengthSortedOrder`).** Both are cold-path — a tokenizer parse and a
per-batch dispatch-order build — so neither carries §4.3's hazard. Both landed **with new tests**,
and that detail is the point: every test that would otherwise cover them
(`TestEncodeBatch_*`, `TestBPESpan_*`) is gated on a model checkpoint and **skips** on a machine
without `testdata/`. A green `go test` proved nothing about either extraction. `encoder/order_test.go`
and `embed/unkid_test.go` need no checkpoint, and both were run against the pre-refactor inline
logic as well to prove behaviour identity rather than assert it.

**Declined: `topKWANDState` (bm25/wand.go).** The 14-line insertion sort sits inside WAND's main
walk — the sparse-query hot loop — and a helper holding a nested double loop is far over Go's
inline budget, so extracting it adds a real call per iteration there. Same trap as §4.3, different
mechanism.

**Still open:** `chunkWith` (chunk/regex/chunker.go) — extractable, but the region ends in a closure
capturing `boundaries`, so it wants a small struct rather than a function, and that is a design call
rather than a mechanical lift.

### 4.4-old · The rest, unstarted and unranked

`encoder/mlp.go` (`moeMLP` CCN 16), `encoder/forward_batch.go` (`forwardBatch` CCN 16),
`encoder/weights.go` (`buildWeightsFromSafetensors` nests 5 deep), `tools/vulncheck/main.go` (`run`
nests 5 deep), `ann/hnsw.go` (7 boolean operators in one conditional). Six extract-method plans
are also on offer, of which the non-`linalg` ones are `encodeBatch` (encoder/model.go, 35 lines,
−2 CCN), `chunkWith` (chunk/regex/chunker.go, 26 lines), `parseBPETokenizer` (embed/tokenize_bpe.go,
7 lines), `query` (ann/flat_i8.go, 20 lines — see §4.3's gate) and `topKWANDState` (bm25/wand.go,
14 lines).

## 5 · What is noise here

**5.1 · `co_change_scatter` is an artifact of three toolchain sweeps — verified.** Four targets are
flagged for co-changing with 48–49 distinct files (`gpu/annmetal/topk_bench_test.go`,
`tools/skips/skips_test.go`, `gpu/metal_sharedevent_test.go`, `linalg/matmul_strided_bench_test.go`,
`tools/consumergate/check.go`). Checking the actual commits: `topk_bench_test.go` appears in five
commits, and three of them are **the go-1.27 directive bump, its `go fix` modernizer sweep, and the
revert of that sweep** — 49, 49 and 51 files respectively. The "shotgun surgery" reading is wrong;
these files were passengers in a mechanical repo-wide edit. Release commits are *not* the cause
(`gpupins --fix` touches 3–9 files). Discount all five.

**5.2 · `change_entropy` findings track active development.** `linalg/quant_w4a8_arm64.go` (top
10%), `ann/flat.go` (top 5%), `ann/flat_i8.go` (top 1%), `encoder/forward_q8.go` (top 2%) are the
files the last two months of kernel work actually touched. Useful as *where to look*, not as
*what to fix*.

**5.3 · Test-file complexity is mostly correct.** `gpu/anncuda/backend_test.go` (CCN 20, nesting 7)
and `gpu/metal_vit_test.go` (CCN 15) are parity harnesses; `chunk/regex/chunker_test.go`'s
`TestPrescreen_neverHidesAMatch` nests 7 deep because it is an exhaustive cross-product. Leave them.

**5.4 · Perf-risk was not triaged.** 112 findings at 18.26/10K LOC — the highest density of the
three Go repos except ken (44.61). repowise scopes this to I/O-in-loop / N+1, resource and
defer-in-loop, blocking-in-async, and explicitly *excludes* algorithmic blowups and GC pressure —
which is where this repo's performance work actually lives (`perf-dead-ends.md`,
`cpu-acceleration.md`). Low expected value.

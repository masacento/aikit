# Task: aikit code health — the repowise findings, triaged

> Scoping doc. Opened 2026-08-26 from the repowise trial (`docs/prompts/repowise-trial-results.md`),
> the first index+health run over this repo. Sibling to goinfer's `docs/task-code-health.md`, which
> came out of the same run. **Status: IN PROGRESS — §4.1 started 2026-08-26.** Unlike goinfer's,
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

**3.1 · `linalg/` is off-limits right now.** AVX-512 VNNI kernel work is in flight and uncommitted
in the working tree (`linalg/quant_i8_amd64.go`, `linalg/quant_w4a8_amd64.go` modified, eight
untracked `dot_*_avx512vnni_amd64.*` files), authored in a cloud session because neither local box
can execute VNNI — the MacBook is arm64 and the Linux box is Zen 2. **Four of the 20 targets live
in `linalg/`** (`quant_w4a8_arm64.go`, `rowblock_amd64.go`, `quant.go`, `matmul_blocked.go`), plus
the largest duplication clusters (`exp_simd.go`, `quant.go`). Touching them now would conflict with
work that cannot be rebased from here. Defer all of it until that lands.

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

### 4.2 · `embed/safetensors.go` sharded-open pair — NOT STARTED

`OpenSafetensorsShardedMmap` and `OpenSafetensorsShardedFromFS` share ~13 lines of index-parse and
aggregate-setup (205-217 / 243-255). Genuinely duplicated, but the two diverge immediately after:
one mmaps and must `finalizeMmaps` on failure, the other `fs.ReadFile`s and has nothing to unwind.
The shared prologue is real; the extraction is worth doing only if it does not force the divergent
cleanup through a shared abstraction. Medium value, non-trivial.

### 4.3 · `ann/flat.go` — the worst file, and a hot path

Score 1.9. `scanFlat` carries a **critical** complex-conditional finding plus a nested-complexity
and a complex-method finding, and the file is in the top 5% for change entropy. `ann/flat_i8.go`
is in the **top 1%**. These are the query hot paths, so any change is a benchmark-gated change,
not a readability one — `BenchmarkFlat*` and the ANN parity tests are the gate. Real finding,
real cost; do not start it casually.

### 4.4 · The rest, unstarted and unranked

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

# aikit roadmap v4 — post-1.5.0

> Rewritten 2026-06-11, after v1.5.0. History: v1 (at v1.1.0) and v2 (at
> v1.2.0) were engineering roadmaps; v3 (at v1.4.0) predicted the backlog
> would empty into adoption work; v1.5.0 made that literal. Annotated prior
> versions live in git history (`git log -- docs/internal/roadmap.md`).
>
> **There are zero unblocked engineering items.** Four releases in three days
> (v1.2.0 → v1.5.0) shipped the full 2026 hybrid-retrieval bar — dense
> f32/int8 + lexical + learned-sparse, bi- and cross-encoder reranking,
> persistence/mmap/`//go:embed`, parity pins on every model path, fuzzing,
> benchmarks, an automated release gate. This document is now an adoption
> campaign and a list of triggers. **The standing rule: new engineering
> enters only through §2's triggers or §1's feedback. If a future session
> finds itself adding kernels while §1 is unstarted, stop and do §1.**

## Scorecard (cumulative, v1.2.0 → v1.5.0)

Retrieval: Flat / FlatI8 / HNSW (Alg-4, recall@10 0.68→1.00) / int8 HNSW
(Δ0) / BM25 ×2 analyzers / SPLADE end-to-end / RRF+RSF / QueryFilter.
Models: Model2Vec (both formats), CodeRankEmbed, MiniLM BERT, SPLADE head,
ms-marco cross-encoder — all parity ≤1e-5, golden-pinned, cgo-free.
Embedded: versioned blobs + zero-copy mmap + `//go:embed` demo (~50 ms
startup). Proof: README comparative table, BeIR/scifact nDCG@10 0.638,
~21 texts/s pure-Go MiniLM. Health: fuzzing (5 fixes), typed errors, scoped
knobs, pool deleted, surface audit (deliberate keep), format policy
(rebuild-per-minor), release gate (ran 1.4.0 and 1.5.0 unattended).

---

## 1. The adoption campaign — the only live work

In order. Items 1–3 are a sequence, not a menu; each feeds the next.

1. **Post the announcement** — [high / trivial]. The r/golang draft was written
   as `r-golang-post.md` (repo root — strip the posting-notes footer), but that
   file is **not in the repo** as of 2026-07-27 (never committed; regenerate or
   locate before posting). Post it; optionally follow with Show HN a few days later (HN
   wants a different lead: the war story or the `//go:embed` demo, not the
   feature list). Submit to awesome-go the same week (its bar: API docs,
   coverage, CI — all already met).
2. **Work the response** — [high / reactive]. Answer every comment and
   issue fast for the first two weeks; the prepped Q&A (footer of the post
   draft) covers the four predictable threads. Triage feature requests
   against §2's triggers rather than accepting them reflexively — but a
   request that *matches* a trigger (e.g. a Windows consumer, a >1M-vector
   corpus) is the trigger firing, and unblocks that item immediately.
3. **Land one named adopter** — [high / not-engineering]. The strategic
   item the last three roadmap versions agreed on. Shapes, best first: an
   external Go service wiring in semantic search (offer hands-on help, as
   the post does); an MCP server `//go:embed`-ing a docs corpus (build it
   *with* someone, not speculatively); ken publicly badged as built-on-aikit
   (first-party, so it half-counts, but it's honest social proof and free).
   Ship v1.6 *with* the adopter named in the README.
4. **The recall war-story blog post** — [medium / low]. Separate asset from
   the announcement: "synthetic vectors can't measure recall" (0.99 → 0.68
   → 1.00) is a standalone engineering post that travels on its own and
   back-links the repo. Write after #1's response settles, using what the
   comments reveal about which framing lands.

## 1b. Unblocked items (from the 2026-06-12 goinfer cross-repo review + external kernel review)

The review's headline: **the split is holding** — goinfer consumes aikit's
loaders/kernels properly, deps point inward only, no container-format
duplication. One deduplication earns immediate work; the rest is gated (§2).

1. **Typed tensor accessors on `embed.SafetensorsFile`** — ✅ **DONE.** [medium / low].
   goinfer hand-writes the same shape-checked typed reads ≥6 times across
   `decoder/weights.go`, `vision/encoder.go`, `decoder/lora.go`,
   `decoder/gptq.go` (`tensorF32`, `loadF32(want…)`, `f16Tensor`,
   `i32Tensor`). Add `TensorF32(name string, want ...int)` (+ I32, and F16
   widening) as methods. Additive to the Hard tier; goinfer deletes its
   helpers at its next aikit bump. Permitted under the standing rule as
   measured deduplication with a named consumer, not new capability.

2. **2×8 register GEMM micro-kernel** — ✅ **DONE.** [high / medium]. External
   review flagged `Dot8x4` as a load-bound 1×8 kernel (each b-load → 1 FMLA, 8
   accumulators < the ~16 to hide FMA latency). The gate — a peak-fraction bench
   against a *measured* f32 ceiling (`fmaPeakARM64` clocked 95.4 GFLOPS, ≈15
   FMA/cyc, settling the 8-vs-16-FMA/cycle question empirically) — put the GEMM at
   40–49% of peak (≤50% ⇒ proceed). `linalg.Dot2x8` (2 a-rows × 8 b-rows, 16
   accumulators) recovered it to **68–73%**: encoder FC matmuls 1.5–1.6×, end-to-end
   encode 1.27–1.36×, bit-identical (same accumulation order as `Dot8x4`, golden
   parity unchanged). arm64 NEON only. Remaining levers, *not* taken (measured win
   already lands the target, and the standing rule discourages speculative kernels):
   the AVX2 port belongs with §2.4 (gated on Zen 4+ access); a 4×8/B-packed
   outer-product kernel could chase the last ~25% to ~90% of peak but needs a real
   throughput trigger to justify the packing path.

3. **Blocked GEMM hoisted into shared `linalg.MatmulBT`** — ✅ **DONE.** [high / medium].
   goinfer's gate (recorded in its `matmulbt-prefill-headroom` note) found the *public*
   `linalg.MatmulBT` was a naive dot-per-output span with **no cache blocking** —
   re-streaming `b` per a-row, **~7% of peak** at prefill shapes. That's a defect every
   kit consumer of `MatmulBT` inherits (goinfer's own f32 vision encoder sat there), not
   just a goinfer concern — so it was fixed despite goinfer deferring its *own* adoption.
   The encoder's blocked + register-tiled GEMM was hoisted into `linalg`
   (`matmul_blocked.go`) as the single shared home; `MatmulBT` (column-parallel) and a new
   Experimental `MatmulBTInto` (serial) both use it, and the encoder delegates
   (bit-identical, golden parity unchanged). Measured **7%→46%** at M=512×4096×4096
   (~6.3×), **68–75%** at the K=768 transformer tiles; width stays numerically inert
   (8-aligned shards). *(Update, v1.7.2: the naive-span threshold for small matmuls was
   removed so `MatmulBT`'s per-output result is M-invariant — the threshold switched
   reduction order at the M=1↔M=K boundary, an avoidable f32-reassociation footgun. All
   M now route through the blocked kernel, which measured faster at small-M decode/
   attention shapes anyway. Gated by `TestMatmulBT_MConsistent`; see §1b.4 for the
   over-attribution that prompted it.)*

   Then the 46% itself was chased and mostly closed: the large-K shortfall wasn't tile
   size but **L1 associativity conflicts** — the 8 b-rows a `Dot2x8` reads are K·4 bytes
   apart and collide in the same cache sets. **B-panel packing** (copy each 8-row group
   into a low-stride buffer first) fixed it **bit-identically**: prefill M=512 **46%→69%**,
   and the encoder's own K=3072 fc2 **+15%** (golden parity unchanged). Gated to K≥2048
   (K=768 dims are already low-stride) and arm64 (packed kernel is NEON `Dot2x8`; amd64
   AVX2 packing → §2.4). Remaining: large M (≥~2048) recovers less (≈53%) because the
   a-panel is re-read per column-group — full 3-level (Goto) blocking with A-packing would
   close it (~70%+) but is a substantial new path, promoted to **§2.12** with its trigger
   (a real large-M f32 prefill that's watched — concretely goinfer's multimodal
   image-prefill). Measured along the way: smaller kBlock and wide n-panels both *hurt* —
   the simple 8-col pack is the sweet spot.

4. **`MatmulBT` made M-invariant — and the mis-attribution that prompted it** — ✅
   **DONE (v1.7.2), with a correction.** [high / low]. A consumer (goinfer) reported a
   same-model speculative-decoding parity failure (`TestSpeculativeGreedyParity`,
   acceptance 0.893 vs ~1.0) after bumping its pin, and it was **mis-attributed to
   aikit**: the theory was that §1b.3's naive/blocked threshold in `MatmulBT` made the
   f32 result M-dependent (M=1 naive vs M=K blocked → ~1e-5 reassociation → flipped
   argmax). We removed the threshold so `MatmulBT` is now **M-invariant** (every output
   bit-identical regardless of M; all M route through the one blocked-kernel order,
   which measured **2–3.8× faster** at small-M decode/attention shapes than the naive
   span it replaced — no perf cost). That is a real improvement and worth keeping —
   `MatmulBT` being M-dependent was an avoidable footgun. **But it did not fix the
   reported bug.** The speculative-parity failure was **consumer-side**: goinfer's
   dense attention computed QKᵀ/AV through two paths (`attendQuery` vs
   `attendBatchedHeads`) that were not bit-identical, and goinfer fixed it there by
   moving attention onto f64 accumulation (`MatmulBTAcc64`, untouched by aikit). The
   threshold removal *transiently* shifted goinfer's f32 attention numerics until that
   f64 move landed; once it did, goinfer's quantized forward stopped calling f32
   `MatmulBT` entirely (W8A8/Q8 weight kernels never did), so the issue is moot.
   `blockedFill`'s internal M-invariance (paired `Dot2x8` vs odd `Dot8x4`) is pinned by
   `TestMatmulBT_MConsistent`; the invariant is documented on `MatmulBT`; the quantized
   kernels (`MatmulBTW4A8`/`Q8`/`W8A8*`) were already M-consistent — untouched.
   **Meta-lesson (recorded so it isn't repeated):** localize a failure in the consumer
   before pointing at the dependency, and never put a downstream-effect claim
   ("fixes X in goinfer") in a release note when only the local property
   (M-invariance) was verified. Chasing the dep cost a release of churn.

   **Follow-up (v1.7.3): the threshold removal exposed a latent amd64 kernel bug —
   found and fixed, not reverted.** Routing *all* shapes through the blocked kernel
   meant small odd-`n4` shapes hit the amd64 AVX2 `Dot8x4`/`Dot4x4` for the first
   time (nothing had since the 1.6.0 hoist), and those kernels were wrong for odd
   `n4`: the trailing single 4-group used an XMM/VEX.128 FMA that zeroed the upper
   128 bits of each YMM accumulator, dropping the loop's lane-4..7 partials (K=13,
   K=300 failed; even-`n4` shapes passed — and arm64 / non-AVX2 were always
   correct). 1.7.3 fixes the kernel (YMM-form tail FMA that preserves the upper
   lanes), keeping 1.7.2's M-invariance — which is now actually correct on amd64.
   A direct `TestAVX2_blockedKernels_oddN4` (odd + even `n4` vs scalar) closes the
   test gap: the prior AVX2 test only exercised the single-row `dotFMA`, never the
   multi-row blocked kernels, which is why the bug stayed latent. Meta-lesson #2:
   a kernel test that doesn't cover the tail × every register-block width leaves a
   blind spot exactly where reuse hides it.

## 2. Gated — unchanged triggers, now the only path for engineering

1. **HNSW zero-copy mmap** — bundle with the next format bump (specced at
   the bump sites), never standalone.
2. **Binary/Hamming pre-filter + rescore** — trigger: an adopter with >1M
   vectors.
3. **Windows real mmap** — trigger: a sizable Windows consumer.
4. **x86 tail (VNNI W4A8, AVX-512)** — trigger: Zen 4+/Cascade Lake+ access
   (cloud c7i/c7a spot remains the cheap unblock); bundle both.
5. **`GOEXPERIMENT=simd` migration** — trigger: archsimd ships arm64 AND
   graduates (amd64-only + gated as of June 2026; arm64/SVE on dev.simd for
   Go 1.27). Then re-spike; within ~10% of hand asm ⇒ migrate, delete `.s`.
6. **In-place index mutation** — trigger: a real consumer proving
   rebuild/delta/swap insufficient. Design rule 4 holds.
7. **Experimental→Hard graduation** — *new, the long-game item:* BERT /
   SPLADE / CrossEncoder / int8 indexes / persistence graduate to the
   semver tier once they survive two quiet consecutive minors under an
   external consumer (the same bar the original Hard tier met). Trigger:
   §1.3's adopter + that stability window.
   *Note (§2.8), updated 2026-07-27:* `WeightMat.MatmulBTInto`'s f32/W8A8
   paths route through the `Workspace`-scoped matmul (honoring
   `SetThreshold`/`SetWorkers`), and are now **consumed in-repo** — the
   `aikit/vision` tower threads a `Workspace` through all six per-layer
   projections via `lw.MatmulBTInto(&s.ws, …)` (`vision/encoder.go`, the
   audit-#12 Workspace threading). So the "unconsumed" caveat is resolved;
   the method has real coverage for the graduation audit.
8. **`linalg.WeightMat` — unify the quantized-weight abstraction** —
   ✅ **DONE (type + 3 of 3 consumers).** The precision-hiding
   weight-matrix wrapper was implemented **three times** — aikit
   `encoder.LayerWeightsQ8`, goinfer `decoder.weightMat`
   (f32/int8/int4-group/W8A8, the richest), goinfer `vision.qmat`
   (f32/W8A8) — all dispatching into `linalg`. The shared Experimental
   `linalg.WeightMat` now exists (storage-only: precision/scales/dispatch;
   model policy stays with each consumer) and **aikit's `encoder` Q8 path
   is migrated onto it, bit-identically** (cosine 0.997 unchanged, Q8
   golden green, -race clean).
   *Gate note:* the stated trigger (goinfer must change the abstraction
   anyway, or a 4th impl) was **not** actively firing — goinfer is on main,
   clean, no in-flight `weightMat` change. This proceeded as **Francis's
   owner override**, and was scoped to land the type + in-repo consumer
   without disturbing goinfer's fastest-moving internal type.
   *Completed since:* goinfer's `decoder.weightMat` migrated (the type is
   gone; `decoder` now holds `linalg.WeightMat` directly — see `eagle.go`
   fields and `gguf.go`'s `streamMat`), and `vision.qmat` migrated as the
   in-aikit refactor §2.9's move made it. The open-coded wrapper is deleted
   in all three places; `vision` keeps only `newQMat`, which is storage
   *policy* (which weights quantize, and that f32 is copied because the
   source is a released mmap) rather than a second abstraction.
   *Bit-identity is gated, not assumed:* the tower's own parity tests need
   checkpoints and skip without them, so `vision.TestQMat_migrationIsBitIdentical`
   reconstructs the old kernel calls and asserts exact float-bit equality
   against the WeightMat path (plus the no-alias and int8-export contracts).
   Break-it-first: flipping the `w8a8` flag moves the result and turns it red.
   *Now consumed (cf. §2.7), updated 2026-07-27:* `aikit/vision` exercises
   `WeightMat.MatmulBTInto` — the audit-#12 Workspace threading routes all six
   per-layer projections through `lw.MatmulBTInto(&s.ws, …)` (`vision/encoder.go`),
   so the `Workspace`-scoped f32/W8A8 paths have an in-repo consumer. The decoder
   migration landing on `MatmulBT` is no longer what the re-evaluation waits on.
9. **`vision` (SigLIP/ViT encoder) → aikit** — ✅ **DONE (aikit side).**
   The vision tower moved into `aikit/vision` (Experimental), verbatim and
   parity-preserving — decode/preprocess/forward/qmat/resident + the
   import-free GPU-export seam; deps are `embed`+`linalg` only and it adds
   **no** external dependency (image codecs are stdlib). aikit is now the
   only **cgo-free** image-embedding library. The full decision/scoping
   record is `docs/internal/archive/vision-move-decision.md`.
   *Gate:* the stated trigger (launch feedback / an adopter asking for
   image or multimodal retrieval) had **not** fired — the move proceeded as
   **Francis's owner override**, recorded in the decision doc + CHANGELOG.
   *Dependency audit (the one real risk):* the decode path is stdlib
   `image/jpeg`+`png` only — no `x/image`, no cgo — so the "stdlib + x/text
   only" promise holds without a decode quarantine.
   *Scope held:* the Gemma-specific projector (vision→LLM tokens) and the
   image-soft-token sentinels stay in goinfer; the projector↔encoder
   boundary is a plain `[]float32`, no decoder imports `vision`.
   *Remaining (goinfer side):* delete goinfer's encoder copy, rename its
   leftover to `package multimodal`, point `gpu/vision_*.go` at
   `aikit/vision`, bump the pin on aikit release — validated green via
   `go.work replace` first. And see the new §2.13 (text tower).
10. **Explicitly left in goinfer** (reviewed, no move): GPTQ/AWQ
   reconstruction (decoder-checkpoint formats, no aikit consumer);
   BPE/SentencePiece tokenizers (generation-side; note a future
   bge-m3-class multilingual embedder would need SentencePiece *Unigram*,
   which neither repo has); `rmsnorm`/`rope` (small, intentional
   duplication per the split); constrain/chat/sampler/serve/gpu.
11. **AMX** — out of scope.
12. **3-level (Goto) f32 GEMM for large-M prefill** — *new (from the §1b.3 packing
   work).* B-panel packing took large-K shapes to ~69% at M≤~1024, but large M
   (≥~2048) recovers less (≈53%): the a-panel is re-read once per output column-group.
   Closing it needs full Goto blocking — an L2-resident packed B panel reused across
   M-blocks, plus A-packing, plus a 3-level (nc/kc/mc) loop. Substantial new kernel
   path. Trigger: a real large-M f32 prefill that anyone watches — concretely goinfer's
   multimodal **image**-prefill latency (the same trigger as §2.9 vision), since that is
   the one hot f32 `MatmulBT` consumer with M≥2048. Measured dead-ends recorded so they
   aren't re-tried: smaller kBlock and wide n-panels both regress. Bundle the amd64 AVX2
   packed kernel (§2.4) with it.
13. **SigLIP *text* tower** — *new (opened by §2.9's vision move).* The vision
   encoder shipped in `aikit/vision` does image→image and image-as-document
   today, but **text↔image** retrieval needs SigLIP's text tower — which exists
   in neither repo (Gemma drives the text side with its LLM, which aikit
   doesn't have). It's BERT-shaped, so `encoder`'s transformer machinery + the
   parity toolchain make it tractable, but it is its own parity-pinned work: its
   own pin script, its own golden, a shared embedding space verified against the
   HF `SiglipModel` (image+text). Trigger: someone actually needs text↔image (a
   cross-modal search adopter), **not** just image→image — don't build the text
   tower speculatively off the image move alone.
14. **`bm25` persistence — a serialized `Index` and/or a `Builder.Add` seam** —
   *new (the 2026-07 perf campaign's N4/N6, deferred as an API decision, not a perf
   item).* The campaign measured the cost of the absent format — `bm25.Build` from
   scratch is **21.5% of the flagship example's cold start** (`nvidia-rtx2070s` W3;
   the original lens heading said 67%, which was measuring against a hand-written
   checkpoint instead of the real 64 MB load), and eliminating it entirely takes
   time-to-first-result from 82.0 → 64.4 ms. Still the largest single cold-start item,
   no longer two thirds of it.
   *Why it is gated, not scheduled:* the cost is a permanent compatibility promise. A
   versioned on-disk format means every future field on `Index` has to be carried,
   defaulted for old files, and gated, for as long as the format exists — a decision
   about what `bm25` **is**, and 17.6 ms is not sufficient grounds for a benchmark to
   make it. Whoever takes it should start from the format and the version-skew policy,
   not the percentage, and should decide alongside it whether `bm25` wants a
   persistence story at all: **rebuilding from an already-embedded corpus may be the
   honest answer** for a package whose build is O(corpus) and fast (the `bm25.Build`
   godoc now says this).
   *N6 (`Builder.Add(tokens)`) is held WITH N4, not separately,* because it is the same
   question from the other side: a serialized `Index` and an incremental `Builder` are
   both about `bm25`'s input/output surface, and committing to either shape first
   constrains the other. Take them together as an API review or not at all.
   *The general lesson (§5.6, "two of the five biggest items are missing APIs"):* N1
   (`StaticModel.EncodeBatch`) shipped inside the campaign and N4 did not, and the split
   is principled — **a perf campaign can measure an absent API's cost but cannot decide
   its shape.** The measurement says how much the absence costs; it never says what the
   presence should look like. N1 was additive with a contract obvious from the serial
   loop it replaced; N4 is a format. Trigger: an adopter who needs warm-restart or
   incremental indexing badly enough to own the format's compatibility surface.

---

## 3. Speculative backlog — post-adoption, not triggered

*Folded in from the former `docs/internal/ideas.md` (2026-08-14): 0 code
citations, not linked from anywhere, drifting as an orphan doc alongside the
one roadmap. Merged here so there is one live-work document, not two.*

A backlog of "AI-cool" additions that fit aikit's ethos: **pure Go, no cgo,
parity-tested, independently importable, composes with what's already here.**
This is deliberately a menu, not a commitment under the standing rule above —
none of these has a trigger; do not start one without finding or declaring
one first, same as §2. Each entry says what it is, why it fits aikit
specifically, how hard it is, and what it composes with.

Context: aikit is the pure-Go **retrieval** toolkit (`topk`, `ann` flat+HNSW,
`bm25`, `fuse` RRF, `embed`, `encoder`, `chunk`, `linalg`); the **generation**
half (decoder / tokenizer) now lives in goinfer. This backlog is the aikit-side
retrieval + primitive ideas. (Written when both halves were one repo, so some
entries reference generation — those have moved to goinfer's backlog.)

Rough effort key: **S** = a few days, **M** = a week or two, **L** = a month+.

### Tier 1 — define what aikit *is*

#### 1. `rag` — the end-to-end retrieval-augmented-generation pipeline  ·  M

**What.** A package that composes the pieces aikit already has into one
grounded-answer pipeline: chunk a corpus (`chunk`) → embed (`embed`) and index
(`ann` + `bm25`) → at query time retrieve from both, fuse the rankings (`fuse`
RRF) → rerank the survivors (`encoder`) → build a prompt with the top passages
and generate a cited answer (`decoder`). A small, opinionated `Pipeline` type
with swappable stages and a `Answer(query) (text, []Citation)` surface.

**Why it fits aikit specifically.** This is the one idea that makes the whole
library *more than the sum of its packages*. Today a consumer has to hand-wire
six packages and know how they interlock; the pipeline is the product, and
"local, pure-Go, no-service RAG with citations" is a thing essentially nothing
else offers. It also turns every other package into a load-bearing component
rather than an island, which is good pressure on their APIs.

**Design notes.** Keep retrieval, fusion, rerank, and generation as
interfaces so each stage is testable and replaceable (e.g. BM25-only, or
rerank-off). Citations come from tracking which chunk ids survived into the
prompt and (optionally) attributing generated spans back to passages via
attention or n-gram overlap. Ship a `demo/rag` CLI: point it at a folder, ask
a question, get a cited answer.

**Composes with.** Literally everything. **Risk:** prompt-format and
context-window management are fiddly; lean on the `decoder` chat template work.

#### 2. Constrained / structured generation — guaranteed-valid output  ·  M

**Status: JSON shipped** (`constrain` package). The logit-mask seam
(`decoder.SamplingParams.LogitProcessor`) + a `Masker` over a byte-level
`Grammar`, with a streaming JSON grammar — a model that physically cannot emit
malformed JSON (`demo/gemma --json`; hard-invariant test vs `encoding/json`).
Still open below: a general GBNF/regex engine and a JSON-Schema → grammar
compiler on the same `Grammar` interface.

**What.** Decode under a constraint so the model *cannot* emit invalid output:
mask the logits at each step to only the tokens a grammar permits. Two layers:
a JSON-Schema → grammar compiler, and a general GBNF-style grammar engine
(regex and CFG). At each step the grammar's current state yields an allowed
token set; everything else gets `-inf` before sampling.

**Why it fits aikit.** Structured output is the feature people actually reach
for in production (function calls, JSON extraction, classification), and a
correct, dependency-free implementation in Go is rare. It sits right on top of
the existing `Sampler` — the logit vector is already in hand at each step. Great
demo value: "a 270M Gemma that physically cannot produce malformed JSON."

**Design notes.** The hard part is mapping grammar terminals to *token*
boundaries (a token may span a grammar boundary, or several tokens compose one
terminal) — precompute, per grammar state, the set of vocab tokens whose string
is a valid continuation, using a trie over the vocab. Cache aggressively; the
allowed-set computation must not dominate the per-token cost.

**Composes with.** `decoder` (logit masking in the sample loop), `tokenizer`
(vocab trie). Pairs beautifully with the `rag` pipeline for structured
extraction. **Effort** climbs to **L** if you want full CFG with good perf.

### Tier 2 — inference-time algorithms the forward pass unlocks

#### 3. Speculative / assisted decoding  ·  M

**What.** Use a small fast model (Gemma 270M) as a *draft* that proposes k
tokens, then verify them in a single batched forward of the larger *target*
model, accepting the longest prefix that matches the target's own
distribution. 2–3× fewer target forwards for the same exact output
distribution (it's provably distribution-preserving, not an approximation).

**Why it fits aikit.** It's the canonical "cool inference trick," it's a
showcase for the multi-model work (draft and target can be different
checkpoints from the same family), and the speedup is real and measurable. The
`decoder` already separates `runLayers` from the LM head, which is most of the
batched-verify plumbing.

**Design notes.** Needs a *batched* forward over k candidate positions (the
verify step), so it leans on the M7 perf work and a KV-cache that can roll back
rejected tokens. Acceptance test: output distribution identical to plain
sampling at the same seed (the correctness gate).

**Composes with.** `decoder` (two models, shared sampler), the M7 batched
matmul. **Risk:** KV-cache rollback is the subtle part.

#### 4. Logprobs, perplexity, and LLM-as-scorer  ·  S

**What.** Expose what the forward pass already computes: per-token log-probs,
sequence perplexity, and a `Score(prompt, continuation)` that returns the
total/average logprob of a continuation. Then an LLM-reranker that scores RAG
candidates by how well the model "expects" them.

**Why it fits aikit.** Almost free — the full logit vector is in hand at every
step; this is mostly API surface plus a softmax-logprob helper. It immediately
enables evaluation, candidate scoring, and a quality signal for the `rag`
pipeline. Cheap, high-utility, low-risk.

**Composes with.** `decoder`, `rag`, the eval harness (#9). **Effort: S** —
the lowest-cost item here with outsized leverage.

#### 5. Sampler family expansion  ·  S

**What.** Add to the existing `Sampler`: **min-p** (keep tokens ≥ p·max_prob —
the current favorite over top-p), **Mirostat** (targets a perplexity
set-point), **contrastive search**, **beam search** (for tasks that want it),
and repetition controls (**repetition/frequency/presence penalties**, **DRY**,
**no-repeat-ngram**).

**Why it fits aikit.** Pure post-processing of the logit vector; each is a
small, independently-testable function alongside `topFilter`. Min-p and the
repetition penalties in particular noticeably improve small-model output, which
matters since aikit's sweet spot is small local models.

**Composes with.** `decoder.Sampler`. **Effort: S** per sampler.

#### 6. KV-cache techniques — prefix caching, attention sinks, cache quant  ·  M

**What.** Three related upgrades to `KVCache`:
- **Prefix caching / sharing** — cache the KV for a fixed prefix (a system
  prompt, a RAG context) and reuse it across requests instead of re-prefilling.
- **StreamingLLM / attention sinks** — keep the first few tokens + a sliding
  window so generation continues coherently past the trained context length
  with bounded memory.
- **KV-cache quantization** — store cached K/V in int8 to roughly halve cache
  memory, the dominant cost at long context.

**Why it fits aikit.** Prefix caching is a direct, large win for the web GUI
(every chat turn re-uses the system prompt) and for `rag` (the retrieved
context is a long, reused prefix). Attention sinks are a genuinely clever,
low-code trick. All three are local-memory wins that matter precisely because
aikit targets laptops.

**Composes with.** `decoder`, the `demo/gemma-web` GUI, `rag`. **Risk:** prefix
caching interacts with position ids and the sliding window — careful invalidation.

### Tier 3 — retrieval-side depth

#### 7. Late-interaction retrieval (ColBERT-style multi-vector)  ·  M

**What.** Instead of one vector per document, store per-token embeddings and
score a query against a document by summing each query token's max similarity to
any document token (MaxSim). Much stronger recall than single-vector,
especially for short or keyword-ish queries.

**Why it fits aikit.** It reuses `encoder` for token-level embeddings and slots
in as an alternative retriever behind the `rag` pipeline's retrieval interface.
It's a well-defined algorithm with a clear correctness story (MaxSim is exact),
and it differentiates aikit's retrieval from "yet another cosine index."

**Design notes.** Storage blows up (N tokens × dim per doc), so this wants the
quantized index (#8) underneath to be practical. Two-stage is natural: cheap
single-vector ANN to get candidates, MaxSim rerank on the survivors.

**Composes with.** `encoder`, `ann`, `rag`. **Risk:** memory; pair with #8.

#### 8. Quantized / large-scale vector index — PQ, IVF, scalar quant  ·  M

**What.** Make `ann` scale past "fits in RAM as f32": **scalar quantization**
(int8 vectors, 4× smaller), **product quantization** (PQ — split the vector,
quantize subspaces, asymmetric distance), and **IVF** (inverted-file coarse
clustering so a query only scans a few cells). Optionally **binary embeddings**
+ Hamming rerank for a very fast first stage.

**Why it fits aikit.** The current flat + HNSW indexes are great up to a point;
PQ/IVF is what lets a laptop hold millions of vectors. It's classic,
well-specified, pure-Go-friendly numerical code with exact recall/latency
tradeoffs you can measure — squarely aikit's style.

**Composes with.** `ann`, `embed`, `rag`, late-interaction (#7). **Effort:** M,
more if you want the full IVF-PQ combo with good recall tuning.

#### 9. Retrieval + generation eval harness  ·  S–M

**What.** A small `eval` package with the standard metrics: **nDCG, MRR,
recall@k, MAP** for retrieval; **perplexity, exact-match, ROUGE-ish overlap**
for generation; plus a runner that takes a labeled set and a pipeline and prints
a scoreboard.

**Why it fits aikit.** The project is already parity-obsessed (golden tests
everywhere); giving every feature a *quality* scoreboard, not just a
correctness one, is the natural extension. It makes #1, #7, and #8 comparable
instead of vibes-based, and it's the kind of thing a serious retrieval library
is expected to have.

**Composes with.** Everything; especially `rag`, `ann`, logprobs (#4).

### Tier 4 — classic ML primitives (cool, broadly useful, pure-Go)

#### 10. Embedding-space clustering — k-means(++), mini-batch  ·  S

K-means/k-means++ over the vectors `embed`/`encoder` produce: semantic
clustering, topic discovery, IVF cell construction (#8), and "summarize a corpus
into N themes." Small, classic, and a building block several other ideas reuse.
**Composes with** `ann`, `embed`, IVF (#8).

#### 11. Near-duplicate detection — SimHash / MinHash + LSH  ·  S

Fingerprint documents and find near-duplicates cheaply. Invaluable as a
*pre-indexing* cleaning step for `rag` corpora (dedup before you embed and
index), and a genuinely useful standalone utility. Pure bit-twiddling, very
Go-friendly. **Composes with** `chunk`, `rag`.

#### 12. Dimensionality reduction + Matryoshka embeddings  ·  S–M

**PCA / random projection** for shrinking embeddings and 2-D visualization, plus
first-class support for **Matryoshka** embeddings (use the first k dims of a
model trained for it — instant 2–4× storage cut at a known quality cost).
**Composes with** `embed`, `ann`, #8.

### How these sequence

A reasonable order once multi-model lands, by leverage-per-effort:

1. **#4 logprobs/perplexity (S)** and **#5 sampler family (S)** — quick wins that
   strengthen `decoder` and unblock eval.
2. **#1 `rag` pipeline (M)** — the centerpiece; turns the library into a product.
3. **#9 eval harness (S–M)** and **#11 dedup (S)** — make `rag` measurable and its
   corpus clean.
4. **#2 constrained generation (M)** — the headline capability.
5. **#8 quantized index (M)** then **#7 late-interaction (M)** — retrieval depth at
   scale.
6. **#3 speculative decoding (M)** and **#6 KV-cache tricks (M)** — perf/UX, best
   after the M7 batched-matmul backend exists.
7. **#10 clustering / #12 dim-reduction (S)** — fill in as needed; several earlier
   items reuse them.

If only two ever get built: **#1 (`rag`)** because it defines what aikit is, and
**#2 (constrained generation)** because guaranteed-valid structured output is the
capability people most want and least often find in pure Go.

### Explicitly out of scope (for now)

Training / fine-tuning (aikit is inference-only by design), encoder-decoder and
non-transformer architectures (T5, Whisper, Mamba/SSM — different skeletons),
and anything requiring cgo or a GPU-only path (the WebGPU backend is the one
sanctioned accelerator, behind the existing `Backend` seam).

---

## Competitive context (refreshed 2026-06-11)

Unchanged from v3 in substance; deltas only:

- **hugot** — the last pipeline gap closed (cross-encoder, same checkpoint,
  Δ5e-6). The comparison is now fully head-to-head and framed honestly in
  the README (deployment tradeoff, not a drag race). Their Go-SIMD bet
  matures with archsimd (§2.5's trigger) — the announcement window is open
  *now* and narrows when that graduates.
- **Antfly/Termite** (Zig core), **coder/hnsw** (measured 0.22 vs 0.995),
  **Bleve/chromem-go/sqlite-vec** (index-only), **Ollama** (server) — all
  unchanged; the README table and capability matrix carry these.
- **Model2Vec upstream** — supported including standard format; watch for
  new potion releases (a better static model is a free quality bump through
  the parity pipeline).

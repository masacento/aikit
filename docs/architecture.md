# aikit architecture

How the pieces fit together: the package DAG, the surrounding repo ecosystem,
the two dependency quarantines, and where the load-bearing invariants live.
For what each package *does*, see the [README package table](../README.md#packages);
for API stability, the [stability tiers](../README.md#stability-tiers). This
doc is about structure and the decisions behind it.

## Design rules

Four rules generate most of the structure:

1. **The core stays pure Go (no cgo).** Enforced in CI (`CGO_ENABLED=0` build
   + a dependency-graph grep). Anything that would break this is pushed
   behind a seam (§ Backend) or into a separate module (§ Quarantines).
2. **Packages are small leaves; the DAG stays shallow.** Most packages depend
   on nothing but the stdlib. `encoder` is the one public composite; two
   Go-internal packages (invisible to importers) cut duplication between
   otherwise-independent leaves that had drifted apart: `bm25` and `sparse`
   share `internal/accum` (a pooled scoring accumulator), and `ann` and
   `embed` share `internal/cursor` (the bounds-checked little-endian reader
   behind HNSW/FlatI8's blob formats and GGUF's container format).
3. **Numerics are parity-pinned.** Every model-touching path (`embed`,
   `encoder`, the `linalg` quant kernels) is tested against golden fixtures
   produced by Python references in [`scripts/`](../scripts/) — see
   [`testdata/README.md`](../testdata/README.md).
4. **Indexes are immutable after build.** Every retrieval index — `ann.Flat`,
   `ann.HNSW`, `ann.FlatI8`, `bm25.Index`, `sparse.Index` — is read-only once
   built. This is a cornerstone, not an accident: it buys two things the rest of
   the design leans on, and gives them up the moment in-place mutation is allowed.

   - **Lock-free concurrent reads.** A built index takes no locks in `Query`, so
     it scales across goroutines for free. Mutation would force an `RWMutex` (or
     hand-rolled lock-free structures) onto the query hot path — slower, and a
     whole class of concurrency bugs that immutability simply does not have.
   - **Snapshot consistency without coordination.** A "the corpus changed" update
     builds a *new* index and swaps a pointer atomically (ken's ADR-012 pattern);
     a reader holds one consistent snapshot for the life of a query — no torn
     reads, no mid-query mutation.

   Changing corpora are served *without* breaking this, in increasing order of
   freshness: **rebuild-and-swap** (the default — re-index, publish the new
   pointer); a **base + delta + fuse** split (a small, frequently-rebuilt delta
   index fused with the big base via `fuse.RRF`/`RSF`, periodically folded in);
   and **logical delete** via a caller-supplied tombstone predicate
   (`Flat`/`HNSW`/`FlatI8` `QueryFilter`) consulted at query time — the index
   itself is never mutated. True in-place mutation (HNSW tombstone graph-repair,
   concurrent `Add`-during-`Query`, incremental BM25 segments) is deliberately out
   of scope: it's a mutable-database concern that would trade away both properties
   above, for a use case outside aikit's embedded, read-heavy niche.

## Package DAG

```mermaid
graph TD
    subgraph core ["aikit core (one module, pure Go, no cgo)"]
        topk["topk<br/><i>bounded top-K heap</i>"]
        ann["ann<br/><i>flat / FlatI8 / HNSW cosine ANN</i>"]
        bm25["bm25<br/><i>identifier-aware lexical index</i>"]
        sparse["sparse<br/><i>learned-sparse (SPLADE) index</i>"]
        fuse["fuse<br/><i>RRF + RSF rank fusion</i>"]
        chunk["chunk (+ regex, markdown, line)<br/><i>chunker registry</i>"]
        linalg["linalg<br/><i>SIMD f32/int8/int4 kernels</i>"]
        embed["embed<br/><i>Model2Vec + safetensors/GGUF loaders</i>"]
        encoder["encoder<br/><i>BERT-family embedders (9 architectures) +<br/>SPLADE expansion + cross-encoder reranker</i>"]
        vision["vision<br/><i>SigLIP/ViT image encoder (Experimental)</i>"]
        bench["bench<br/><i>recall + latency harness</i>"]
        encoder --> embed
        encoder --> linalg
        encoder --> sparse
        vision --> embed
        vision --> linalg
        ann --> linalg
        ann --> topk
        bm25 --> topk
        sparse --> topk
        bench --> ann
    end
    ts["chunk/treesitter<br/><i>separate module</i>"] -.->|implements chunk.Chunker| chunk
    gpu["gpu/* — 8 native CUDA/Metal backends<br/><i>separate modules, cgo, ~14.6k LOC</i>"]
    gpu -.->|EnableGPU| ann
    gpu -.->|RegisterBackend #quot;cuda#quot;/#quot;metal#quot;| encoder
    gpu -.->|RegisterResident| vision
    ts --> gts["gotreesitter<br/>(pure Go, large grammars)"]
    embed --> xtext["golang.org/x/text"]
```

Everything not shown depending on something depends only on the stdlib.
`topk`, `fuse`, `chunk`, and `linalg` are leaves; the index packages (`ann`,
`bm25`, `sparse`) use `topk` for selection, and since v1.2 `ann` scores
through `linalg`'s SIMD dot kernels; `embed` adds the one external dependency
(`golang.org/x/text`, for tokenizer normalization); `encoder`, `vision`, and
`bench` are the composites. `encoder` also imports `sparse` directly:
`encoder.SPLADE.Expand` produces the `sparse.SparseVec` values the `sparse`
package indexes, so the SPLADE loop (BERT + masked-LM head → sparse vector →
inverted index) runs end-to-end in-process. `vision` (the SigLIP/ViT image
encoder) is a new leaf consumer on `embed`+`linalg` and adds **no** external
dependency — its image decode uses only stdlib codecs (`image/jpeg`,
`image/png`), and like `encoder` it exposes an import-free GPU-export seam
(`GPUWeights`) plus a `RegisterResident` inversion so a WebGPU backend (e.g.
goinfer's) plugs in without the core importing it. The dotted edges are
registry-based, not imports: `chunk/treesitter` registers itself via
`chunk.Register` on import, and each `gpu/*` backend registers itself via the
matching seam (`FlatI8.EnableGPU`, `encoder.RegisterBackend`,
`vision.RegisterResident`) on import — see § Quarantines and
[`docs/task-native-gpu.md`](task-native-gpu.md) for the full backend-by-backend
detail (device substrate, per-seam kernels, parity gates).

## Repo ecosystem

aikit is the stable middle of a three-repo system:

```mermaid
graph LR
    ken["<b>ken</b><br/>code-search CLI / MCP server<br/><i>consumer; home of the ADRs</i>"]
    aikit["<b>aikit</b><br/>retrieval library<br/><i>this repo — semver, cgo-free</i>"]
    goinfer["<b>goinfer</b><br/>LLM decoder runtime<br/>Gemma/Qwen/Llama, tokenizers<br/><i>faster-moving</i>"]
    gpu["goinfer/gpu<br/>WebGPU backend (cgo)<br/><i>opt-in, -tags gpu</i>"]
    ken --> aikit
    goinfer -->|"embed, linalg, vision"| aikit
    gpu -.->|"RegisterBackend(#quot;webgpu#quot;)"| aikit
    gpu --- goinfer
```

aikit was extracted so the retrieval core could make a semver promise while
the LLM runtime keeps moving (the split is recorded in goinfer's
[`migration-plan.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/migration-plan.md);
the motivating critique in
[`internal/archive/road-to-1.0-critique.md`](./internal/archive/road-to-1.0-critique.md)).
Dependencies point inward only: goinfer and ken import aikit; aikit imports
neither.

## The Backend seam

The most consequential decision in the repo
([`encoder/backend.go`](../encoder/backend.go)): the encoder's forward pass
abstracts exactly one primitive — `MatmulBT` (`dst[M,N] = a[M,K]·b[N,K]ᵀ`,
the safetensors `[out,in]` weight layout, so no transpose copy). Norms,
RoPE, softmax, and elementwise ops always run on CPU; only the big weight
matmuls route through the `Backend` interface.

The default `"cpu"` backend (pure-Go SIMD) is compiled in. GPU is an
*inversion*: an external package calls `encoder.RegisterBackend(name, …)`
from its `init()`, and aikit gains GPU acceleration without the cgo
dependency ever entering its module graph. Two implementations exist: an
external one (`goinfer/gpu`, registers `"webgpu"`) and aikit's own native
ones (`gpu/enccuda`/`gpu/encmetal`, in this repo, register `"cuda"`/`"metal"`
— see § Quarantines). `NewBackend(name)` on a build that never imported a
registering package degrades to CPU with an explanatory error rather than
failing. `ann` and `vision` have analogous inversions of their own
(`FlatI8.EnableGPU`, `vision.RegisterResident`) that aikit's `gpu/anncuda`
family also plugs into — same pattern, different interface, because `ann`'s
and `vision`'s hot paths aren't a single matmul the way `encoder`'s is.

Why only matmul for `encoder`: it's the hot path by a wide margin, and
keeping everything else on CPU avoids a host↔device round-trip per layer.

## Quarantines

Three kinds of dependency are deliberately kept out of the core module graph,
each by a different mechanism (CI enforces all three):

| Dependency | Why quarantined | Mechanism |
|---|---|---|
| `cogentcore/webgpu` (cgo) | would break the no-cgo promise | inverted behind `encoder.Backend`; lives in `goinfer/gpu`, not this repo |
| `gotreesitter` (pure Go, ~large embedded grammars; pre-1.0 upstream) | payload size + upstream churn risk | separate module `chunk/treesitter`, registers via `chunk.Register`; versioned in lockstep with the core |
| CUDA/Metal bindings (cgo) — aikit's own native GPU backends | would break the no-cgo promise | `gpu/*`: a shared device-substrate module (`gpu`) plus 8 one-backend-per-module leaves (`anncuda`/`annmetal` → `FlatI8.EnableGPU`; `enccuda`/`encmetal` → `encoder.RegisterBackend("cuda"/"metal", …)`, the same seam `goinfer/gpu` uses; `qwencuda`/`qwenmetal` and `visioncuda`/`visionmetal` → `vision.RegisterResident`); each versioned in lockstep with the core (the two-tag-series, `tools/gpupins`); full detail in [`docs/task-native-gpu.md`](task-native-gpu.md) |

## Retrieval pipeline (how a consumer composes it)

The packages are independent; the canonical composition
([`examples/rag/`](../examples/rag)) is:

```mermaid
graph LR
    docs["documents"] --> chunk2["chunk"]
    chunk2 --> embed2["embed"]
    embed2 --> ann2["ann"]
    chunk2 --> bm252["bm25"]
    ann2 --> fuse2["fuse (RRF)"]
    bm252 --> fuse2
    fuse2 --> enc2["encoder rerank"]
    enc2 --> topk2["top-K results"]
```

Nothing in aikit requires this shape — each stage is independently usable,
and `fuse` works on any rankings with comparable keys.

The `ann2 --> fuse2 <-- bm252` piece — retrieve both signals, fuse.RRF them —
is the same four lines in every hybrid-search example (`examples/rag`,
`examples/colbert`). [`hybrid.Retriever`](../hybrid) is a thin, opt-in wrapper
around exactly that piece, for callers who'd rather not repeat it; it composes
already-built indices (doesn't build them, embed, tokenize, chunk, or rerank)
and stays out of the way of anyone who prefers to hand-wire `fuse.RRF`
directly, which every example here still does.

## Where the invariants live

The correctness-critical contracts are documented at the point of use; this
is the index:

| Invariant | Lives at |
|---|---|
| `ann` requires L2-normalized inputs; normalization happens at the `embed` boundary | [README carry-over invariants](../README.md#carry-over-invariants-read-these-once), `ann/flat.go` doc |
| `embed` accumulates in float64 (f32 silently breaks golden parity on longer inputs) | `embed/model.go` precision contract, `embed/pool.go` |
| `bm25` tokenizer and `encoder` weights are code-tuned (a hidden assumption for general NLP) | README carry-over invariants, `bm25/tokenize.go` |
| mmap-backed tensors must not outlive their `Close()` | `embed` mmap loader docs |
| Quant kernels trust block-size-aligned K (caller contract) | `linalg/quant.go` |
| Parallel matmuls are bit-identical at any width (column partition) | `linalg/linalg.go`, CHANGELOG 0.5.1 |
| Kernel dispatch / CPU feature detection map | [`internal/cpu-acceleration.md`](./internal/cpu-acceleration.md) |
| Serialized blob formats: rebuild-per-minor pre-1.0; `Load*` rejects other versions with `ann.ErrFormat` (loud, never silent) | [README versioning](../README.md#serialized-blob-formats), `ann/hnsw_persist.go` + `ann/flat_i8_persist.go` format-bump checklist |

## ADR index

Code comments cite ADRs by number (e.g. `bm25/tokenize.go`, `chunk/registry.go`,
`topk/topk.go`). **The ADR documents live in the ken repo** (aikit's packages
originated there; the records stayed with the project journal). The numbers
cited from aikit code, with the decision each is invoked for:

| ADR | Cited from | Invoked for |
|---|---|---|
| ADR-005 | `chunk/treesitter` | the swap-out path if `gotreesitter` breaks |
| ADR-006 | `bm25/tokenize.go` | code-tuned tokenization is intentional |
| ADR-008 | `bm25/tokenize.go` | verbatim parity as the tokenizer contract |
| ADR-010 | `chunk/*` | chunker registry + graceful-degradation behavior |
| ADR-012 | `ann/flat.go` | exact flat scan as the baseline ANN |
| ADR-025 | `topk`, `bm25`, `ann` | K-sized stable sort for deterministic ties |
| ADR-026 | `topk/topk.go` | heap selector vs alternatives |
| ADR-027 / 028 | `bm25/tokenize.go` | byte-scanning tokenizer + buffer reuse |
| ADR-029 | `embed` race test | concurrent-encode safety contract |
| ADR-032 | `chunk/*` | public `Chunker` interface + registry as 1.0 surface |

If ken's ADR directory is ever published or relocated, update this table to
deep-link; until then this index is the in-repo resolution for those
references.

## Numbered-citation index

Code comments also cite several OTHER numbering schemes — `audit #18`, `item
27`, `perf-campaign A5`, `lens doc §4.3`, `plan §6.2`, an `M8c` milestone
tag — each resolvable a different way. This table is the same treatment the
ADR index above gives ADR-NNN: what a style means and where (if anywhere) to
go look, so a citation doesn't require guessing. Every row below was verified
against the actual archive docs (grepped every citation site in the tree and
checked it resolves), not assumed from the style alone — see the two that
turned out NOT to be one doc each.

| Style | Example | Resolves to |
|---|---|---|
| `audit #N` | `audit #18` | `docs/internal/archive/AUDIT-2026-07-25.md`, finding `### N.` — in-repo, exact. Every cited number (3, 5–24) has a matching heading; verified 2026-08-15, no gaps. |
| `perf-campaign item N` (plain, unlettered) | `perf-campaign item 27` | `docs/internal/archive/perf-campaign-2026-07/perf-campaign-2026-07-28.md` — in-repo, exact, always this one doc. Items ≤26 have their own `### N ·` heading; higher ones (up to 44+) live as a numbered table row plus later numbered prose discussion ("Item 27 is DONE — …") rather than a heading — Ctrl+F "item N" finds both. Verified every cited number (0–44) resolves; none are missing. |
| `perf-campaign AN` (lettered) | `perf-campaign A5` | **A DIFFERENT DOC**: `docs/internal/archive/perf-campaign-2026-07/task-perf-handoff-linux.md`, `### AN ·` (A1–A6 exist; all six are cited from real code — ann/flat.go, ann/flat_i8.go, embed/tokenize.go, fuse/rsf.go, fuse/fuse.go, bm25/query.go, sparse/sparse.go). This is the doc a reader following the plain `item N` row above will NOT find these in — the letter, not just the number, picks the file. |
| A bare `(B7)` with no `perf-campaign` prefix | `tools/scriptsguard/main.go` | **Does not resolve.** Grepped every archive doc in `perf-campaign-2026-07/`: no `B7` heading, table row, or mention anywhere. `task-perf-handoff-linux.md` mentions "A1–A6 and B landed" as a single unenumerated follow-on, and `task-perf-lens-scans.md` has an unrelated prose-numbered "Phase B" list (items 6–10, not "B6"–"B10") — neither is addressable as `B7`. Likely a stale reference to an earlier, since-renumbered version of these docs. Flagged rather than guessed at; the comment's own claim (rules derived by path, not hand-enumerated) doesn't depend on the citation resolving. |
| Other letter+number tags (`H2`, `H7`, `C5`, `F2`, `P1`, …) | `H2` (`embed/typed_accessors_test.go`, `vision/qwen_encoder.go`) | **Not indexed here — a different citation family from perf-campaign's, scope not yet mapped.** These cluster in `embed`/`vision`/`gpu`/`linalg`, not the packages perf-campaign touched, so they're very likely a separate review's findings-lettering, not a perf-campaign phase. Needs its own grep-and-verify pass before a row can be added truthfully. |
| `lens doc §N.M` | `lens doc §4.3` | `docs/internal/archive/perf-campaign-2026-07/task-perf-lens-scans.md`'s own §-numbered sections — in-repo. |
| `plan §N.M`, milestone tags (`M0`…`M24`, `M8c`) | `plan §6.2` (`encoder/attention.go`) | A pre-1.0 CodeRankEmbed build plan from before aikit's split out of ken. **Not present in this repo, goinfer, or ken's tree as far as this index can confirm** — historical color surviving in the comment, not a document to go find. Don't spend time looking for a `plan.md`. Note: this style overlaps in SHAPE with `M8`/`M24` etc. seen elsewhere (`gpu/metal.go`, `encoder/forward_q8.go`) — those are the SAME milestone family (encoder/CodeRankEmbed build milestones), not a coincidence, but still unresolvable the same way. |
| `migration-plan §N` | `migration-plan §7/§9` (`tools/preflight/main.go`) | The aikit⇄goinfer module-split plan. Its stub, `docs/internal/archive/aikit-module-split-plan.md`, used to point at `goinfer/docs/migration-plan.md` — that file is gone now that the split it described is done (confirmed absent from goinfer's current tree, 2026-08-14). Historical only, same as the milestone tags above. |

Update this table when a new citation STYLE appears (a new archive doc, a new
tag family) — it tracks styles, not individual numbers, so it shouldn't need
touching every time a comment cites one more `item N`. If you add a row,
verify it the way the rows above were: grep every site, don't infer the
target doc from the style alone — the `AN` row above looks like it should
live next to the plain `item N` row and it does not.

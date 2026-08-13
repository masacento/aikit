# Task: the gpu lint group flags staticcheck issues on one box and not another (and CI is green)

**Filed 2026-08-13**, during the gpu_gate.sh → `tools/gpugate` migration (`fc77906`). Small,
self-contained; not part of that migration, which only surfaced it.

## The discrepancy

`go run -C tools ./gpugate` (the `lint` group — `golangci-lint run` across the nine gpu
modules) disagrees by machine on the SAME commit, and neither matches CI:

| where | golangci-lint | `lint` group |
|---|---|---|
| this Mac (arm64) | v2.11.4, **built with go1.26.1** | **FAIL** — `gpu/annmetal`, `gpu/encmetal` |
| nobara (RTX 2070S) | v2.11.4, **built with go1.26.5** | PASS (clean) |
| main CI (`gpu-kernels`) | v2.11.4, `go install …@v2.11.4` | green |

The old `scripts/gpu_gate.sh` reported the identical FAIL on the Mac and PASS on nobara — so
this is **not** introduced by the migration; the Go gate and the shell agree, box for box.

## What the Mac flags (both staticcheck)

```
gpu/annmetal/backend.go:3:1: ST1000: at least one file in a package should have a package comment
gpu/annmetal/gemv_bench_test.go:72:12: ST1023: should omit type float64 from declaration; it will be inferred (var secs float64 = math.Inf(1))
```
(`gpu/encmetal` flags the analogous ST1000.)

## The likely cause

The pinned **version is identical (v2.11.4)** on all three; what differs is the **Go
toolchain golangci-lint was compiled against**. staticcheck vendors `go/analysis` passes
whose behavior tracks the compiler's `go/types`, so a golangci-lint built with go1.26.1
enables checks (or resolves them differently) than one built with go1.26.5. Pinning the
*golangci-lint version* does not pin this; `go install …@v2.11.4` builds it with whatever
`go` is on the box, so CI, nobara, and the Mac can each get a different staticcheck.

## The decision to make (either closes it)

1. **The findings are real** — then fix them: add the missing package comments to
   `gpu/annmetal` and `gpu/encmetal`, drop the redundant `float64` in
   `gemv_bench_test.go:72`. Cheap, and ST1000/ST1023 are reasonable.
2. **The gate must be deterministic across boxes** — then close the gap that lets CI stay
   green while a maintainer's box goes red: pin the Go toolchain golangci-lint is built with
   (or vendor a fixed staticcheck), so the `lint` group gives one answer everywhere. A gate
   whose verdict depends on the runner's incidental Go build is a gate that will surprise
   someone at tag time.

Both are worth doing; (1) is a few minutes and (2) is the durable fix. Neither belongs in
the migration commit.

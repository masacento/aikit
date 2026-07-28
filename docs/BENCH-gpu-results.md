# GPU benchmark results

> **GENERATED** by `bench report` from `records.jsonl` — do not edit by hand.
> Re-generate after each periodic run (`docs/BENCH-gpu.md`). Cross-machine
> absolute numbers are never placed in adjacent columns — the CPU baselines are
> different chips; compare within a machine, or via the normalized summary.

aikit `1a6722b` · 16 records · 1 machine(s)

## Per-machine tables (apples-to-apples, same box)

### apple-m1pro Apple M1 Pro arm64 · GPU Apple M1 Pro (Apple)

| workload | shape | precision | cpu-arm64 q/s | metal q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 5.1k queries/s | 648.3 queries/s | 0.13× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 6.2k queries/s | 2.8k queries/s | 0.44× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 6.2k queries/s | 3.2k queries/s | 0.52× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 6.2k queries/s | 3.3k queries/s | 0.54× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 1.6k queries/s | 288.8 queries/s | 0.18× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 1.5k queries/s | 330.3 queries/s | 0.21× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 1.5k queries/s | 331.1 queries/s | 0.22× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 1.5k queries/s | 311.3 queries/s | 0.20× | 0.9754 | ✅ |

## Normalized cross-platform summary (speedup over each box's own CPU)

The only honest all-backends view: absolute ms don't compare across machines, but
*speedup over the CPU each GPU ships next to* does — the decision-relevant number.

| workload | shape | precision | metal ×vs-cpu |
|---|---|---|--:|
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 0.13× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 0.44× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 0.52× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 0.54× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 0.18× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 0.21× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 0.22× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 0.20× |

## Dispatch thresholds (the crossover — the input to backend dispatch)

- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=100k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=10k)**: CPU wins at every measured batch (GPU never overtakes).

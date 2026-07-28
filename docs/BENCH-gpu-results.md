# GPU benchmark results

> **GENERATED** by `bench report` from `records.jsonl` — do not edit by hand.
> Re-generate after each periodic run (`docs/BENCH-gpu.md`). Cross-machine
> absolute numbers are never placed in adjacent columns — the CPU baselines are
> different chips; compare within a machine, or via the normalized summary.

aikit `e331284` · 16 records · 1 machine(s)

## Per-machine tables (apples-to-apples, same box)

### apple-m1pro Apple M1 Pro arm64 · GPU Apple M1 Pro (Apple)

| workload | shape | precision | cpu-arm64 q/s | metal q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 6.4k queries/s | 646.7 queries/s | 0.10× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 6.4k queries/s | 9.3k queries/s | 1.45× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 6.3k queries/s | 16.7k queries/s | 2.64× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 6.3k queries/s | 17.6k queries/s | 2.79× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 1.6k queries/s | 132.8 queries/s | 0.08× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 1.5k queries/s | 1.0k queries/s | 0.65× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 1.5k queries/s | 2.7k queries/s | 1.74× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 1.5k queries/s | 3.1k queries/s | 1.99× | 0.9754 | ✅ |

## Normalized cross-platform summary (speedup over each box's own CPU)

The only honest all-backends view: absolute ms don't compare across machines, but
*speedup over the CPU each GPU ships next to* does — the decision-relevant number.

| workload | shape | precision | metal ×vs-cpu |
|---|---|---|--:|
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 0.10× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 1.45× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 2.64× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 2.79× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 0.08× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 0.65× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 1.74× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 1.99× |

## Dispatch thresholds (the crossover — the input to backend dispatch)

- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 64**.
- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 8**.

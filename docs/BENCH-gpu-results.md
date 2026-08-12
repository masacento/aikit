# GPU benchmark results

> **GENERATED** by `bench report` from `records.jsonl` — do not edit by hand.
> Re-generate after each periodic run (`docs/BENCH-gpu.md`). Cross-machine
> absolute numbers are never placed in adjacent columns — the CPU baselines are
> different chips; compare within a machine, or via the normalized summary.

aikit `d9eda2b` · 36 records · 2 machine(s)

## Per-machine tables (apples-to-apples, same box)

### apple arm64 · GPU Apple M1 Pro (Apple)

| workload | shape | precision | cpu-arm64 q/s | metal q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 9.2k queries/s | 3.9k queries/s | 0.42× | 0.9922 | ✅ |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 2.4k queries/s | 1.7k queries/s | 0.70× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 9.4k queries/s | 1.3k queries/s | 0.14× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 9.4k queries/s | 9.4k queries/s | 1.00× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 9.2k queries/s | 22.9k queries/s | 2.48× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 9.1k queries/s | 24.8k queries/s | 2.72× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 2.7k queries/s | 132.2 queries/s | 0.05× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 2.5k queries/s | 1.0k queries/s | 0.40× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 2.3k queries/s | 2.7k queries/s | 1.16× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 2.3k queries/s | 3.1k queries/s | 1.36× | 0.9754 | ✅ |

### nvidia-rtx2070s AMD Ryzen 7 3700X amd64 · GPU NVIDIA GeForce RTX 2070 SUPER (sm_75)

| workload | shape | precision | cpu-amd64 q/s | cuda q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 3.8k queries/s | 4.6k queries/s | 1.21× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 3.7k queries/s | 9.4k queries/s | 2.56× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 3.8k queries/s | 11.7k queries/s | 3.05× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 3.9k queries/s | 12.9k queries/s | 3.34× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 794.9 queries/s | 587.5 queries/s | 0.74× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 796.8 queries/s | 3.1k queries/s | 3.83× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 797.7 queries/s | 8.3k queries/s | 10.38× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 726.8 queries/s | 11.1k queries/s | 15.25× | 0.9754 | ✅ |

## Normalized cross-platform summary (speedup over each box's own CPU)

The only honest all-backends view: absolute ms don't compare across machines, but
*speedup over the CPU each GPU ships next to* does — the decision-relevant number.

| workload | shape | precision | cuda ×vs-cpu | metal ×vs-cpu |
|---|---|---|--:|--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | — | 0.42× |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | — | 0.70× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 1.21× | 0.14× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 2.56× | 1.00× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 3.05× | 2.48× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 3.34× | 2.72× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 0.74× | 0.05× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 3.83× | 0.40× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 10.38× | 1.16× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 15.25× | 1.36× |

## Dispatch thresholds (the crossover — the input to backend dispatch)

- **apple/ann.FlatI8.Query (int8, N=100k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple/ann.FlatI8.Query (int8, N=10k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 64**.
- **apple/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 64**.
- **nvidia-rtx2070s/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 8**.
- **nvidia-rtx2070s/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 1**.

# GPU benchmark results

> **GENERATED** by `bench report` from `records.jsonl` — do not edit by hand.
> Re-generate after each periodic run (`docs/BENCH-gpu.md`). Cross-machine
> absolute numbers are never placed in adjacent columns — the CPU baselines are
> different chips; compare within a machine, or via the normalized summary.

aikit `b179acf` · 44 records · 3 machine(s)

## Per-machine tables (apples-to-apples, same box)

### apple arm64 · GPU Apple M1 Pro (Apple)

| workload | shape | precision | cpu-arm64 q/s | metal q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 9.2k queries/s | 3.9k queries/s | 0.42× | 0.9922 | ✅ |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 2.5k queries/s | 1.6k queries/s | 0.65× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 9.4k queries/s | 1.7k queries/s | 0.18× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 9.4k queries/s | 10.9k queries/s | 1.16× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 9.2k queries/s | 23.2k queries/s | 2.53× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 9.1k queries/s | 25.2k queries/s | 2.76× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 2.7k queries/s | 207.8 queries/s | 0.08× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 2.7k queries/s | 1.5k queries/s | 0.56× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 2.5k queries/s | 3.4k queries/s | 1.35× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 2.4k queries/s | 3.6k queries/s | 1.50× | 0.9754 | ✅ |

### apple-m1pro Apple M1 Pro arm64 · GPU Apple M1 Pro (Apple)

| workload | shape | precision | cpu-arm64 q/s | metal q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| vision.SigLIP.Forward | dim=512 patches=196 | int8 | 4.4 images/s | 6.1 images/s | 1.37× | 0.9999 | ✅ |
| vision.SigLIP.Forward | dim=768 patches=576 | int8 | 0.7 images/s | 1.1 images/s | 1.54× | 0.9999 | ✅ |

### nvidia-rtx2070s Ryzen 7 3700X amd64 · GPU NVIDIA GeForce RTX 2070 SUPER (sm_75)

| workload | shape | precision | cpu-amd64 q/s | cuda q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 10.0k queries/s | 10.4k queries/s | 1.04× | 0.9922 | ✅ |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 1.2k queries/s | 3.3k queries/s | 2.74× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 9.8k queries/s | 9.2k queries/s | 0.94× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 7.5k queries/s | 26.0k queries/s | 3.46× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 8.3k queries/s | 39.3k queries/s | 4.72× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 10.0k queries/s | 42.4k queries/s | 4.25× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 475.3 queries/s | 2.8k queries/s | 5.92× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 579.8 queries/s | 7.3k queries/s | 12.67× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 1.1k queries/s | 31.5k queries/s | 28.52× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 1.2k queries/s | 44.1k queries/s | 36.61× | 0.9754 | ✅ |

## Normalized cross-platform summary (speedup over each box's own CPU)

The only honest all-backends view: absolute ms don't compare across machines, but
*speedup over the CPU each GPU ships next to* does — the decision-relevant number.

| workload | shape | precision | cuda ×vs-cpu | metal ×vs-cpu |
|---|---|---|--:|--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 1.04× | 0.42× |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 2.74× | 0.65× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 0.94× | 0.18× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 3.46× | 1.16× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 4.72× | 2.53× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 4.25× | 2.76× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 5.92× | 0.08× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 12.67× | 0.56× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 28.52× | 1.35× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 36.61× | 1.50× |
| vision.SigLIP.Forward | dim=512 patches=196 | int8 | — | 1.37× |
| vision.SigLIP.Forward | dim=768 patches=576 | int8 | — | 1.54× |

## Dispatch thresholds (the crossover — the input to backend dispatch)

- **apple/ann.FlatI8.Query (int8, N=100k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple/ann.FlatI8.Query (int8, N=10k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 64**.
- **apple/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 8**.
- **nvidia-rtx2070s/ann.FlatI8.Query (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia-rtx2070s/ann.FlatI8.Query (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia-rtx2070s/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia-rtx2070s/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 8**.

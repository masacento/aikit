# GPU benchmark results

> **GENERATED** by `bench report` from `records.jsonl` — do not edit by hand.
> Re-generate after each periodic run (`docs/BENCH-gpu.md`). Cross-machine
> absolute numbers are never placed in adjacent columns — the CPU baselines are
> different chips; compare within a machine, or via the normalized summary.

aikit `4c34cc5` · 44 records · 2 machine(s)

## Per-machine tables (apples-to-apples, same box)

### apple-m1pro Apple M1 Pro arm64 · GPU Apple M1 Pro (Apple)

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
| vision.SigLIP.Forward | dim=512 patches=196 | int8 | 4.4 images/s | 6.1 images/s | 1.37× | 0.9999 | ✅ |
| vision.SigLIP.Forward | dim=768 patches=576 | int8 | 0.7 images/s | 1.1 images/s | 1.54× | 0.9999 | ✅ |

### nvidia-rtx2070s Ryzen 7 3700X amd64 · GPU NVIDIA GeForce RTX 2070 SUPER (sm_75)

| workload | shape | precision | cpu-amd64 q/s | cuda q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 9.8k queries/s | 12.6k queries/s | 1.28× | 0.9922 | ✅ |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 1.4k queries/s | 3.8k queries/s | 2.77× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 10.5k queries/s | 8.3k queries/s | 0.79× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 10.2k queries/s | 27.2k queries/s | 2.66× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 9.9k queries/s | 41.1k queries/s | 4.17× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 9.8k queries/s | 32.0k queries/s | 3.27× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 1.8k queries/s | 4.1k queries/s | 2.32× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 1.4k queries/s | 19.0k queries/s | 13.19× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 1.4k queries/s | 37.9k queries/s | 27.35× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 1.3k queries/s | 48.0k queries/s | 35.84× | 0.9754 | ✅ |

## Normalized cross-platform summary (speedup over each box's own CPU)

The only honest all-backends view: absolute ms don't compare across machines, but
*speedup over the CPU each GPU ships next to* does — the decision-relevant number.

| workload | shape | precision | cuda ×vs-cpu | metal ×vs-cpu |
|---|---|---|--:|--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 1.28× | 0.42× |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 2.77× | 0.65× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 0.79× | 0.18× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 2.66× | 1.16× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 4.17× | 2.53× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 3.27× | 2.76× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 2.32× | 0.08× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 13.19× | 0.56× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 27.35× | 1.35× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 35.84× | 1.50× |
| vision.SigLIP.Forward | dim=512 patches=196 | int8 | — | 1.37× |
| vision.SigLIP.Forward | dim=768 patches=576 | int8 | — | 1.54× |

## Dispatch thresholds (the crossover — the input to backend dispatch)

- **apple-m1pro/ann.FlatI8.Query (int8, N=100k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple-m1pro/ann.FlatI8.Query (int8, N=10k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 64**.
- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 8**.
- **nvidia-rtx2070s/ann.FlatI8.Query (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia-rtx2070s/ann.FlatI8.Query (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia-rtx2070s/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia-rtx2070s/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 8**.

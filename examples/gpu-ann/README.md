# gpu-ann — the native-GPU ANN path, made visible

The same `ann.FlatI8` int8 index, queried once on the CPU and once with
`EnableGPU()`, checking the two agree exactly and timing both. `gpu/`'s
~6,150 lines (device substrate + 8 backend modules) were otherwise only
discoverable via godoc. See `main.go`'s doc comment for the full design
rationale, including why this is its own Go module.

Needs no model — `FlatI8` just quantizes whatever vectors it's given, so this
runs on synthetic data with zero setup beyond the GPU itself:

```sh
go run ./examples/gpu-ann                     # defaults: n=1e6, 256 queries, k=10
go run ./examples/gpu-ann -n 50000 -queries 8  # a quick, smaller run
```

## Real output, both backends this repo ships

**Metal** (M1 Pro, an integrated GPU):

```
GPU backend "metal" enabled.

256 queries, k=10, corpus n=1000000 dim=256:
  CPU  QueryBatch: 1.07939925s
  GPU  QueryBatch: 956.771708ms (1.13x)
  results: IDENTICAL — every query's top-k matched exactly (index and score).
```

Repeated runs on the same machine landed anywhere from 0.97× to 1.2× — close
to parity, not the larger wins `docs/task-native-gpu.md` reports from its own
benchmark harness. At small scale (`-n 50000 -queries 8`) Metal **loses**
outright (dispatch overhead dominates a corpus this small).

**CUDA** (RTX 2070 SUPER, a discrete GPU, verified over SSH):

```
GPU backend "cuda" enabled.

256 queries, k=10, corpus n=1000000 dim=256:
  CPU  QueryBatch: 2.809285269s
  GPU  QueryBatch: 40.355864ms (69.61x)
  results: IDENTICAL — every query's top-k matched exactly (index and score).
```

And still ~20× at the small scale where Metal lost. A discrete GPU's dispatch
overhead is proportionally far smaller against its own much higher raw
throughput — "GPU loses below some N" turned out to be a property of the
*hardware class*, not a fixed threshold in the code.

Every run on both backends was bit-identical to the CPU result — this example
checks that itself, and `gpu/anncuda`'s own test suite (run on the same CUDA
box) adds a much more exhaustive parity gate on top.

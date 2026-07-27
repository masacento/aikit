package ann

// Backend is a device accelerator for FlatI8 scoring. It is registered by an
// aikit/gpu build (native Metal today, CUDA later) via RegisterBackend in an
// init(); the default pure-Go build registers none, and FlatI8 scores on the CPU
// (linalg.MatmulBTW8A8). This is the same inversion the encoder/vision packages
// use for their GPU backends — the core ann package never imports the device
// package, so the default build stays cgo-free and pure-Go.
type Backend interface {
	// NewI8Index makes an int8 index resident on the device: bq is [n*dim]
	// row-major int8 codes, scales is [n] per-row reconstruction scales. The
	// returned handle scores queries against those resident codes.
	NewI8Index(bq []int8, scales []float32, n, dim int) (I8Index, error)
	// Name identifies the backend (e.g. "metal") for diagnostics.
	Name() string
}

// I8Index is a device-resident int8 index. Score writes, for each row j, the same
// W8A8 value linalg.MatmulBTW8A8 computes on the CPU (dynamic-quantize q to int8,
// int8-dot against row j, rescale by the query and row scales) — so a GPU-scored
// top-k is rank-identical to the CPU one within the int8 quantization.
type I8Index interface {
	Score(q []float32, dst []float32) error
	Close() error
}

var backend Backend

// RegisterBackend installs the process-wide device backend for FlatI8 scoring.
// Called from an init() in the aikit/gpu build; the default build never calls it.
// Last registration wins (a process links at most one native backend).
func RegisterBackend(b Backend) { backend = b }

// HasBackend reports whether a device backend has been registered.
func HasBackend() bool { return backend != nil }

// BackendName returns the registered backend's name, or "" if none.
func BackendName() string {
	if backend == nil {
		return ""
	}
	return backend.Name()
}

// EnableGPU makes this index's codes resident on the registered device backend so
// subsequent Query calls score on the GPU. It returns an error if no backend is
// registered (default build), if the index is paged (LoadFlatI8MmapPaged — the
// budget-paged path is CPU-only until Phase 2), or if the device upload fails; on
// any error the index keeps scoring on the CPU. Idempotent-ish: a second call
// replaces the prior device index (the old one is closed).
func (f *FlatI8) EnableGPU() error {
	if backend == nil {
		return errNoBackend
	}
	if f.pager != nil {
		return errPagedGPU
	}
	if f.closed {
		panic("ann: EnableGPU on a closed FlatI8")
	}
	idx, err := backend.NewI8Index(f.bq, f.scales, f.n, f.dim)
	if err != nil {
		return err
	}
	if f.gpu != nil {
		_ = f.gpu.Close()
	}
	f.gpu = idx
	return nil
}

// GPUEnabled reports whether Query scores on the device backend.
func (f *FlatI8) GPUEnabled() bool { return f.gpu != nil }

type annError string

func (e annError) Error() string { return string(e) }

const (
	errNoBackend annError = "ann: no device backend registered (build with the aikit/gpu backend)"
	errPagedGPU  annError = "ann: EnableGPU unsupported on a paged index (CPU-only until Phase 2)"
)

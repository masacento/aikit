//go:build !darwin

// Package gpu is aikit's cgo-free native-GPU device substrate. The Metal
// implementation (metal.go, annbackend.go) is darwin-only; the CUDA implementation
// is Phase 1b. On other platforms this package is empty and registers no ann.Backend,
// so ann.FlatI8 scores on the CPU.
package gpu

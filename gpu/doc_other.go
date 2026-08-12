//go:build !darwin && !linux

// Package gpu is aikit's cgo-free native-GPU device substrate: Metal on darwin
// (metal.go) and CUDA on linux (cuda.go), presenting the same Device / Buffer /
// Queue / Pipeline / Encoder vocabulary on both. On every other platform this
// package is empty and registers no ann.Backend, so ann.FlatI8 scores on the CPU.

package gpu

//go:build linux

package main

// Blank import: anncuda's init() reaches CUDA and calls ann.RegisterBackend on
// success — see gpu/anncuda/backend.go, the Linux mirror of gpu/annmetal. If
// there's no CUDA GPU/driver, it registers nothing and main.go falls back to
// CPU-only, same as if this import weren't here at all.
//
// Verified only via gpu/annmetal on darwin (this repo's dev machine has no
// CUDA hardware) — anncuda mirrors annmetal's structure closely enough
// (its own doc comment: "the Linux mirror of gpu/annmetal, same...") that this
// should just work, but it hasn't been run.
import _ "github.com/townsendmerino/aikit/gpu/anncuda"

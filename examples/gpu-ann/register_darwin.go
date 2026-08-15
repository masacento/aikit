//go:build darwin

package main

// Blank import: annmetal's init() reaches Metal and calls ann.RegisterBackend
// on success — see gpu/annmetal/backend.go. If there's no Metal GPU (or the
// kernel fails to compile), it registers nothing and main.go falls back to
// CPU-only, same as if this import weren't here at all.
import _ "github.com/townsendmerino/aikit/gpu/annmetal"

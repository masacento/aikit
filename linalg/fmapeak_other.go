//go:build !amd64 && !arm64

package linalg

// measureFMAPeak has no probe on this architecture, so no ceiling is claimed.
func measureFMAPeak() (float64, bool) { return 0, false }

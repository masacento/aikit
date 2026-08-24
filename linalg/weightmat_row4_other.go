//go:build !arm64

package linalg

// RepackInt4Row4 is a no-op on non-arm64: the split-half + 4-row-interleave
// layout and its kernel (docs/task-w4a8-neon-bandwidth.md) are arm64/NEON
// only. Always returns false; w is never modified.
func (w *WeightMat) RepackInt4Row4() bool { return false }

// MatmulBTW4A8Into always takes the canonical path on non-arm64 — w.q4Row4
// is never set here (RepackInt4Row4 above is a no-op), so there is nothing
// to dispatch on.
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	MatmulBTW4A8Into(ws, a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
}

// row4Usable is always false on non-arm64 — the split-half + 4-row kernel is
// NEON-only. See the arm64 file's comment for why WrapInt4Row4 needs this.
func row4Usable() bool { return false }

//go:build !arm64 && !amd64

package linalg

// MatmulBTW4A8Into takes the canonical path on every target with no repacked layout of its own:
// q4Row4 is never set here (RepackInt4Row4 is a no-op) and neither is q4SplitHalf
// (RepackInt4SplitHalf likewise), so there is nothing to dispatch on.
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	MatmulBTW4A8Into(ws, a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
}

// RepackInt4SplitHalf is a no-op off amd64: the split-half layout's only consumer is the AVX2
// kernel. Always returns false; w is never modified.
func (w *WeightMat) RepackInt4SplitHalf() bool { return false }

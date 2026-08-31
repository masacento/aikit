//go:build !arm64

package linalg

// NOTE: MatmulBTW4A8Into is NOT defined here. amd64 has its own split-half-aware version
// (weightmat_splithalf_amd64.go); every other non-arm64 target gets the canonical one
// (weightmat_canonical_other.go). Splitting it this way keeps the row4 no-ops below shared.

// RepackInt4Row4 is a no-op on non-arm64: the split-half + 4-row-interleave
// layout and its kernel (docs/task-w4a8-neon-bandwidth.md) are arm64/NEON
// only. Always returns false; w is never modified.
func (w *WeightMat) RepackInt4Row4() bool { return false }

// row4Usable is always false on non-arm64 — the split-half + 4-row kernel is
// NEON-only. See the arm64 file's comment for why WrapInt4Row4 needs this.
func row4Usable() bool { return false }

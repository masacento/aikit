//go:build arm64

package linalg

// RepackInt4Row4 attempts to populate w's optional split-half +
// 4-row-interleaved layout (RepackW4A8Row4/RepackW4A8Row4Scales) from its
// canonical int4 storage, for a load-time caller to opt a tensor into the
// faster MatmulBTW4A8Into dispatch below. Explicit and caller-driven, never
// probed for automatically inside a matmul (docs/task-w4a8-neon-bandwidth.md's
// plumbing brief design constraint): a loader decides per-tensor whether to
// call this, e.g. skipping it for tensors expertPager will read directly off
// a read-only mmap span, where there is no load-time repack step at all.
//
// Returns false, leaving w unchanged (canonical q4/q4s stay the only
// representation), if: w isn't int4-resident, DotProd isn't available on
// this core, rows isn't a multiple of 4, or cols isn't a multiple of the
// int4 group size (currently only group=32 is supported — the kernel's own
// fixed contract). A false return is not an error; it means this tensor's
// shape or this core doesn't qualify, and MatmulBTW4A8Into below will keep
// using the canonical path transparently.
func (w *WeightMat) RepackInt4Row4() bool {
	if w.q4 == nil || !hasDotProd {
		return false
	}
	if w.group != 32 || w.rows%4 != 0 || w.cols%w.group != 0 {
		return false
	}
	w.q4Row4 = RepackW4A8Row4(w.q4, w.rows, w.cols, w.group)
	w.q4Row4Scales = RepackW4A8Row4Scales(w.q4s, w.rows, w.cols, w.group)
	return true
}

// MatmulBTW4A8Into is MatmulBTW4A8Into's WeightMat-method form for an
// int4-resident w, uniform for the caller: uses the row4-interleaved
// kernel when RepackInt4Row4 has populated it and M=1 (the only shape it
// applies to — batched/prefill gets nothing from this optimization and
// keeps routing through the canonical path, matching
// docs/prompts/w4a8-item3-harness.md's own "no M>1-specific work" scoping),
// falls back to the canonical per-row kernel otherwise. This is where the
// paged-MoE carve-out becomes automatic rather than a call-site special
// case: a WeightMat that was never repacked (paged tensors, by
// construction, since a read-only mmap span has no load-time repack step)
// simply always takes the fallback branch here.
//
// Bit-identical either way, for the same logical weights
// (TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical) — this method chooses
// a kernel, never a numeric result.
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	if M == 1 && w.q4Row4 != nil {
		MatmulBTW4A8Row4Into(ws, a, w.q4Row4, w.q4Row4Scales, dst, M, w.cols, w.rows, w.group)
		return
	}
	MatmulBTW4A8Into(ws, a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
}

// row4Usable reports whether this CPU can safely dispatch the split-half +
// 4-row-interleaved kernel — the same hasDotProd gate RepackInt4Row4 already
// applies before ever setting q4Row4. WrapInt4Row4 (weightmat.go, portable)
// calls this to apply the identical gate to EXTERNALLY-supplied bytes (a
// .giw kind-4 load reading an already-repacked layout off disk) — a case
// RepackInt4Row4's own gate never covered, since before WrapInt4Row4 existed
// the only way q4Row4 got populated was RepackInt4Row4 itself.
func row4Usable() bool { return hasDotProd }

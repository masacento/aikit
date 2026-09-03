//go:build arm64

package linalg

// sdotIssuePeak runs iters iterations of 32 independent SDOTs over 16
// accumulators with no memory traffic, and returns one accumulator so the work
// cannot be elided. See sdot_peak_arm64.s. Only call it on a DotProd core.
//
//go:noescape
func sdotIssuePeak(iters int) int32

// sdotPerIter is the SDOT count in one sdotIssuePeak loop iteration, kept next to
// the reader rather than buried in the assembly's shape.
const sdotPerIter = 32

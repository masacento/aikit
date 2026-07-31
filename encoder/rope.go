package encoder

import (
	"math"
	"sync"
	"sync/atomic"
)

// ropeTable holds precomputed cosθ/sinθ for every position up to seqLen
// at the model's configured base (1000 for CodeRankEmbed — NOT the
// usual 10000; see plan §1 and config.json).
//
// Layout: cos[m, d] for m ∈ [0, seqLen), d ∈ [0, headDim/2). Each entry
// covers the (d, d+halfDim) pair under rotate_half.
type ropeTable struct {
	headDim int
	halfDim int
	seqLen  int
	base    float64
	cos     []float32 // [seqLen, halfDim]
	sin     []float32 // [seqLen, halfDim]
}

// ropeCache memoizes one ropeTable across forwards.
//
// A table is a pure function of (headDim, base) and position: row m holds
// cos/sin of m·invFreq[d], with NO dependence on the table's own length. So a
// table built for a longer sequence is a strict superset of a shorter one, and a
// view over its first seqLen rows is BIT-IDENTICAL to newRopeTable(seqLen, …) —
// not approximately, the same arithmetic on the same inputs.
//
// The zero value is ready to use. Safe for concurrent forwards: the fast path is
// a single atomic load, and growth publishes a NEW table rather than resizing
// the old one, so a view another goroutine is still reading stays valid.
//
// This replaces a per-Encode rebuild (newRopeTable's doc called that "trivially
// cheaper than reusing a global cache with sync", which holds at CodeRankEmbed's
// L=512/headDim=64 but not at GTE's 8192 max — and the sync here is one atomic
// load on the hit path, not a mutex).
type ropeCache struct {
	mu  sync.Mutex
	tab atomic.Pointer[ropeTable]
}

func (c *ropeCache) get(seqLen, headDim int, base float64) *ropeTable {
	if t := c.tab.Load(); t.fits(seqLen, headDim, base) {
		return t.view(seqLen)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if t := c.tab.Load(); t.fits(seqLen, headDim, base) {
		return t.view(seqLen) // another goroutine grew it while we waited
	}
	t := newRopeTable(seqLen, headDim, base)
	c.tab.Store(t)
	return t
}

// fits reports whether t covers a request. A nil receiver (empty cache) does not.
func (t *ropeTable) fits(seqLen, headDim int, base float64) bool {
	return t != nil && t.headDim == headDim && t.base == base && t.seqLen >= seqLen
}

// view narrows t to its first seqLen positions, sharing the backing arrays.
// The result is read-only, as apply() requires.
func (t *ropeTable) view(seqLen int) *ropeTable {
	if t.seqLen == seqLen {
		return t
	}
	n := seqLen * t.halfDim
	return &ropeTable{
		headDim: t.headDim, halfDim: t.halfDim, seqLen: seqLen, base: t.base,
		cos: t.cos[:n], sin: t.sin[:n],
	}
}

// newRopeTable precomputes the rotary frequencies for one forward call.
// Cheap at CodeRankEmbed's shape (512×32 f32 each = 64 KB), which is why the
// Nomic forwards still build one per call. GTE goes through ropeCache instead:
// its max is 8192, and the table costs 2·seqLen·halfDim transcendental
// evaluations to build, which is not free once per Encode.
func newRopeTable(seqLen, headDim int, base float64) *ropeTable {
	if headDim%2 != 0 {
		panic("encoder: rope headDim must be even")
	}
	half := headDim / 2
	t := &ropeTable{
		headDim: headDim,
		halfDim: half,
		seqLen:  seqLen,
		base:    base,
		cos:     make([]float32, seqLen*half),
		sin:     make([]float32, seqLen*half),
	}
	// inv_freq[d] = 1 / base^(2d/headDim)  for d ∈ [0, half)
	// Note: NeoX convention; matches HF rotary_emb_interleaved=false.
	invFreq := make([]float64, half)
	for d := range half {
		invFreq[d] = 1.0 / math.Pow(base, float64(2*d)/float64(headDim))
	}
	for m := range seqLen {
		row := m * half
		mf := float64(m)
		for d := range half {
			theta := mf * invFreq[d]
			t.cos[row+d] = float32(math.Cos(theta))
			t.sin[row+d] = float32(math.Sin(theta))
		}
	}
	return t
}

// applyRoPE rotates x in place. x layout: [seqLen, heads, headDim]
// flattened row-major. Rotate_half on the last axis per position m:
//
//	x1 = x[:half]; x2 = x[half:]
//	out[:half] = x1 * cos[m] - x2 * sin[m]
//	out[half:] = x2 * cos[m] + x1 * sin[m]
//
// This is the same shape PyTorch's NomicBert applies via
// rotary_emb_interleaved=false.
func (t *ropeTable) apply(x []float32, heads int) { t.applyRows(x, heads, t.seqLen) }

// applyRows is apply restricted to the first `rows` positions, for a caller
// holding fewer rows than the table was built for — the trimmed final layer of a
// CLS-only forward computes Q for row 0 alone (lens doc §4.1). Position m still
// indexes the same cos/sin row, so row 0 gets exactly the rotation it gets in a
// full-length pass.
func (t *ropeTable) applyRows(x []float32, heads, rows int) {
	half := t.halfDim
	hd := t.headDim
	stride := heads * hd // per-position stride
	for m := range rows {
		cosRow := t.cos[m*half : (m+1)*half]
		sinRow := t.sin[m*half : (m+1)*half]
		base := m * stride
		for h := range heads {
			off := base + h*hd
			for d := range half {
				x1 := x[off+d]
				x2 := x[off+half+d]
				c := cosRow[d]
				s := sinRow[d]
				x[off+d] = x1*c - x2*s
				x[off+half+d] = x2*c + x1*s
			}
		}
	}
}

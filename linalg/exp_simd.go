//go:build goexperiment.simd

package linalg

import (
	"math"

	"simd"
)

// softmaxRowIntoRaw (SIMD build) — perf-campaign item 13, validated as a
// prototype (~/tmcode/go127-simd-audit, 2026-08-20) before landing here: on
// apple-m1pro (128-bit NEON) softmax measured ~2.5-2.6x faster than the
// scalar build; on nvidia-rtx2070s (Ryzen 7 3700X, AVX2) ~3.0-3.2x at the
// default 128-bit width and ~3.7-4.6x with GODEBUG=simd='+256' (AVX2's
// native width — Go 1.27's simd package defaults conservatively to 128-bit
// even on AVX2 hardware; 256 needs the explicit override past a safety
// check). Both are CPU-vs-CPU numbers — scalar Go arithmetic vs vector CPU
// instructions (NEON / AVX2) — nothing to do with the GPU (CUDA/Metal)
// kernels, which are a separate code path entirely.
//
// Vectorizes the independent axes only (max-scan, exp, scale); the f64 sum
// stays a scalar sequential fold in the SAME order the non-SIMD build uses
// (exp_scalar.go) — the one part of softmax that is a genuine reduction, per
// this codebase's standing rule that reductions re-tree and elementwise
// passes don't (dead-ends §8.4 applied the same discipline to layerNorm).
//
// NOT bit-identical to the non-SIMD build: expF32CoreVec's MulAdd is a fused
// op, and Go's scalar compiler already fuses p*r+c on arm64 but not amd64 —
// so this can differ from the scalar softmaxRowIntoRaw by up to 1 ULP per
// exponential (TestExpF32CoreVec_matchesScalarULP gates the bound). That is
// strictly a GOEXPERIMENT=simd-only difference: the default build (this
// file excluded) is untouched.
func softmaxRowIntoRaw(dst, src []float32) {
	n := len(src)
	c := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	// Pass 1: max — same scalar fold the non-SIMD kernel uses over the lane
	// partials, so maxV is bit-identical to it (max is order-invariant absent
	// NaN, which is outside softmax's contract either way).
	maxV := src[0]
	i := 0
	if n >= L {
		m := simd.LoadFloat32s(src[0:L])
		for i = L; i+L <= n; i += L {
			m = m.Max(simd.LoadFloat32s(src[i : i+L]))
		}
		var buf [16]float32 // Len() <= 512/32 = 16
		m.Store(buf[:L])
		maxV = buf[0]
		for _, v := range buf[1:L] {
			if v > maxV {
				maxV = v
			}
		}
	}
	for ; i < n; i++ { // tail (and the whole row when n < L)
		if v := src[i]; v > maxV {
			maxV = v
		}
	}

	// Pass 2: exp(v - max), vectorized.
	bcMax := simd.BroadcastFloat32s(maxV)
	i = 0
	for ; i+L <= n; i += L {
		expF32CoreVec(simd.LoadFloat32s(src[i:i+L]).Sub(bcMax), &c).Store(dst[i : i+L])
	}
	if i < n {
		// Part-load fills the tail lanes with zero — in-domain (e^0=1) —
		// and StorePart writes back only the real elements.
		v, _ := simd.LoadFloat32sPart(src[i:])
		expF32CoreVec(v.Sub(bcMax), &c).StorePart(dst[i:])
	}

	// Pass 3: f64 sum — scalar on purpose, same tree as the non-SIMD kernel.
	var sum float64
	for _, e := range dst {
		sum += float64(e)
	}
	if sum == 0 {
		u := 1.0 / float32(n)
		for j := range dst {
			dst[j] = u
		}
		return
	}

	// Pass 4: scale.
	inv := simd.BroadcastFloat32s(float32(1.0 / sum))
	i = 0
	for ; i+L <= n; i += L {
		simd.LoadFloat32s(dst[i : i+L]).Mul(inv).Store(dst[i : i+L])
	}
	if i < n {
		v, _ := simd.LoadFloat32sPart(dst[i:])
		v.Mul(inv).StorePart(dst[i:])
	}
}

// expSIMDConsts holds the broadcast constants so a hot loop hoists them once.
type expSIMDConsts struct {
	log2e, magic, ln2Hi, ln2Lo             simd.Float32s
	c0, c1, c2, c3, c4, c5, one, underflow simd.Float32s
	two, overflow, posInf                  simd.Float32s
	i127, i1, i255                         simd.Int32s
}

func newExpSIMDConsts() expSIMDConsts {
	return expSIMDConsts{
		log2e:     simd.BroadcastFloat32s(log2eF32),
		magic:     simd.BroadcastFloat32s(roundMagicF32),
		ln2Hi:     simd.BroadcastFloat32s(ln2HiF32),
		ln2Lo:     simd.BroadcastFloat32s(ln2LoF32),
		c0:        simd.BroadcastFloat32s(1.9875691500e-4),
		c1:        simd.BroadcastFloat32s(1.3981999507e-3),
		c2:        simd.BroadcastFloat32s(8.3334519073e-3),
		c3:        simd.BroadcastFloat32s(4.1665795894e-2),
		c4:        simd.BroadcastFloat32s(1.6666665459e-1),
		c5:        simd.BroadcastFloat32s(5.0000001201e-1),
		one:       simd.BroadcastFloat32s(1),
		underflow: simd.BroadcastFloat32s(expUnderflowF32),
		two:       simd.BroadcastFloat32s(2),
		overflow:  simd.BroadcastFloat32s(expOverflowF32),
		posInf:    simd.BroadcastFloat32s(float32(math.Inf(1))),
		i127:      simd.BroadcastInt32s(127),
		i1:        simd.BroadcastInt32s(1),
		i255:      simd.BroadcastInt32s(255),
	}
}

// expF32CoreVec is expF32Core (exp.go), vectorized — same constants, same
// round-magic trick, same Cephes minimax chain, same op order, including the
// e>=255 two-step scale expF32Core uses for x close enough to the overflow
// bound that k reaches 128 while the true result is still finite (2^128
// would encode +Inf and poison it — see expF32Core's own comment). Valid for
// the same domain ExpF32 delegates to the scalar core: x in
// [expUnderflowF32, expOverflowF32]. x > expOverflowF32 or NaN is NOT this
// function's job — see expF32Vec, which adds those two guards on top for a
// caller (SiLU) whose argument is unbounded, unlike softmax's x <= 0.
func expF32CoreVec(x simd.Float32s, c *expSIMDConsts) simd.Float32s {
	z := x.Mul(c.log2e)
	kf := z.Add(c.magic).Sub(c.magic) // round-to-nearest-even via 1.5*2^23
	r := x.Sub(kf.Mul(c.ln2Hi)).Sub(kf.Mul(c.ln2Lo))

	p := c.c0.MulAdd(r, c.c1)
	p = p.MulAdd(r, c.c2)
	p = p.MulAdd(r, c.c3)
	p = p.MulAdd(r, c.c4)
	p = p.MulAdd(r, c.c5)
	res := p.Mul(r).Mul(r).Add(r).Add(c.one) // p*r^2 + r + 1, same op order

	// 2^k by bit construction: (k+127)<<23 reinterpreted as float32. For
	// softmax's x <= 0 domain, e is always < 255 (see the comment this used
	// to carry) and the branch below is inert; for SiLU's e^-x with x very
	// negative (e.g. x=-88), e CAN reach 255+, which is why the two-step
	// path exists at all — both branches are computed and mask-selected,
	// the two-branch pattern this codebase already uses for Erf/Tanh.
	e := kf.ConvertToInt32().Add(c.i127)
	scaleLo := e.ShiftAllLeft(23).ToBits().BitsToFloat32()
	scaleHi := e.Sub(c.i1).ShiftAllLeft(23).ToBits().BitsToFloat32()
	resLo := res.Mul(scaleLo)
	resHi := res.Mul(scaleHi).Mul(c.two)
	res = resHi.IfElse(e.GreaterEqual(c.i255), resLo)

	// Lanes with x < underflow have e <= 0; e=0 self-flushes via a zero
	// scale but e<0 shifts sign bits into garbage — mask keeps only
	// in-domain lanes (Masked zeroes where the mask is false).
	return res.Masked(x.GreaterEqual(c.underflow))
}

// expF32Vec is ExpF32 (exp.go), vectorized: expF32CoreVec plus the two guards
// it doesn't cover — overflow saturates to +Inf, NaN propagates — the full
// contract a caller with an UNBOUNDED argument needs (unlike softmax, whose
// domain after subtracting the row max is always x <= 0). This is the guard
// work ExpF32Into declined to build for its caller-less state; SiLU's real
// caller is what justifies building it now.
func expF32Vec(x simd.Float32s, c *expSIMDConsts) simd.Float32s {
	res := expF32CoreVec(x, c)
	res = c.posInf.IfElse(x.Greater(c.overflow), res)
	// NaN != NaN is the one IEEE comparison that's true for NaN and only
	// NaN, so this reproduces ExpF32's `if x != x { return x }` branch-free.
	return x.IfElse(x.NotEqual(x), res)
}

// siluIntoRaw is SiLUInto (exp.go) with the length/empty checks already done
// — SiLU's autoresearch T1 target (docs/prompts/simd-elementwise-autoresearch.md).
// x/(1+e^-x) is where item 7's SoftmaxRowInto could stay narrow: exp's
// argument here is UNBOUNDED, so this is built on expF32Vec (full guards),
// not the softmax-only expF32CoreVec. The saturating ends need no branch of
// their own — they fall out of expF32Vec's own guards exactly as the scalar
// SiLUF32's doc comment says: x far below the exp overflow bound makes
// e^-x=+Inf and x/(1+Inf) a signed zero; x far above it makes e^-x flush to
// 0 and the result exactly x (Div's IEEE semantics do the rest).
func siluIntoRaw(dst, src []float32) {
	n := len(src)
	c := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	i := 0
	for ; i+L <= n; i += L {
		x := simd.LoadFloat32s(src[i : i+L])
		e := expF32Vec(x.Neg(), &c)
		x.Div(c.one.Add(e)).Store(dst[i : i+L])
	}
	if i < n {
		x, _ := simd.LoadFloat32sPart(src[i:])
		e := expF32Vec(x.Neg(), &c)
		x.Div(c.one.Add(e)).StorePart(dst[i:])
	}
}

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

// erfSIMDConsts holds ErfF32's broadcast constants — separate from
// expSIMDConsts (which erfVec still uses internally, for the tail branch's
// exp(-x^2)) so softmax/SiLU's hot path doesn't carry erf's ~20 extra
// broadcasts it never needs.
type erfSIMDConsts struct {
	zero, one, negOne, four, half, invSqrt2, erfScale      simd.Float32s
	sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10 simd.Float32s // series (|x|<1), Cephes
	tc0, tc1, tc2, tc3, tc4                                simd.Float32s // A&S tail (|x|>=1)
	a1                                                     simd.Float32s
}

func newErfSIMDConsts() erfSIMDConsts {
	return erfSIMDConsts{
		zero: simd.BroadcastFloat32s(0),
		one:  simd.BroadcastFloat32s(1), negOne: simd.BroadcastFloat32s(-1),
		four: simd.BroadcastFloat32s(4), half: simd.BroadcastFloat32s(0.5),
		invSqrt2: simd.BroadcastFloat32s(0.7071067811865476),
		erfScale: simd.BroadcastFloat32s(1.1283791670955126),
		sc0:      simd.BroadcastFloat32s(1.3122532963802806e-08),
		sc1:      simd.BroadcastFloat32s(-1.4503852223150468e-07),
		sc2:      simd.BroadcastFloat32s(1.4589169000933706e-06),
		sc3:      simd.BroadcastFloat32s(-1.3227513227513228e-05),
		sc4:      simd.BroadcastFloat32s(1.0683760683760684e-04),
		sc5:      simd.BroadcastFloat32s(-7.5757575757575758e-04),
		sc6:      simd.BroadcastFloat32s(4.6296296296296294e-03),
		sc7:      simd.BroadcastFloat32s(-2.3809523809523808e-02),
		sc8:      simd.BroadcastFloat32s(1.0000000000000001e-01),
		sc9:      simd.BroadcastFloat32s(-3.3333333333333331e-01),
		sc10:     simd.BroadcastFloat32s(1),
		tc0:      simd.BroadcastFloat32s(1.061405429),
		tc1:      simd.BroadcastFloat32s(-1.453152027),
		tc2:      simd.BroadcastFloat32s(1.421413741),
		tc3:      simd.BroadcastFloat32s(-0.284496736),
		tc4:      simd.BroadcastFloat32s(0.254829592),
		a1:       simd.BroadcastFloat32s(0.3275911),
	}
}

// erfVec is ErfF32 (exp.go), vectorized: both branches (series for |x|<1,
// A&S tail for |x|>=1) are computed unconditionally and mask-selected — the
// two-branch pattern this file's own comments describe for Erf/Tanh, now
// built. NOT full ExpF32 domain: the tail branch's exp(-x^2) argument is in
// [-16, -1] for the |x| in [1,4] it's ever evaluated at (x>4 saturates
// separately, before reaching it), comfortably inside expF32CoreVec's
// validated range with no overflow risk — so this reuses the narrow core,
// not expF32Vec, exactly as softmax does and for the same reason.
//
// NaN is NOT specially guarded, unlike expF32Vec's guard for SiLU's real
// caller: no production caller here has ever produced NaN input (GELU feeds
// on a forward pass's own activations), and the existing scalar ErfF32 has
// no stated/tested NaN contract either — this only vectorizes the domain
// both the scalar function and its real callers actually see.
func erfVec(x simd.Float32s, ec *erfSIMDConsts, xc *expSIMDConsts) simd.Float32s {
	absX := x.Abs()
	sgn := ec.negOne.IfElse(x.Less(ec.zero), ec.one)

	// Series branch, |x| < 1: erf(x) = (2/sqrt(pi))*x*P(x^2), no cancellation.
	z := absX.Mul(absX)
	p := ec.sc0.MulAdd(z, ec.sc1)
	p = p.MulAdd(z, ec.sc2)
	p = p.MulAdd(z, ec.sc3)
	p = p.MulAdd(z, ec.sc4)
	p = p.MulAdd(z, ec.sc5)
	p = p.MulAdd(z, ec.sc6)
	p = p.MulAdd(z, ec.sc7)
	p = p.MulAdd(z, ec.sc8)
	p = p.MulAdd(z, ec.sc9)
	p = p.MulAdd(z, ec.sc10)
	seriesMag := ec.erfScale.Mul(absX).Mul(p)

	// A&S 7.1.26 tail, |x| >= 1: erf ~ 1, so 1-(small) is stable.
	t := ec.one.Div(ec.one.Add(ec.a1.Mul(absX)))
	pt := ec.tc0.MulAdd(t, ec.tc1)
	pt = pt.MulAdd(t, ec.tc2)
	pt = pt.MulAdd(t, ec.tc3)
	pt = pt.MulAdd(t, ec.tc4)
	negX2 := absX.Mul(absX).Neg()
	tailMag := ec.one.Sub(pt.Mul(t).Mul(expF32CoreVec(negX2, xc)))

	mag := seriesMag.IfElse(absX.Less(ec.one), tailMag)
	mag = ec.one.IfElse(absX.Greater(ec.four), mag) // saturated: 1-erf(4) below f32 resolution
	return mag.Mul(sgn)
}

// geluIntoRaw is GELUInto (exp.go) with the length/empty checks already
// done — T1's second target, the exact (erf) GELU. Built directly on erfVec;
// no extra guard work of its own (GELU's own saturating ends, per GELUF32's
// doc comment, fall out of erf's ±1 saturation with no separate branch).
func geluIntoRaw(dst, src []float32) {
	n := len(src)
	ec := newErfSIMDConsts()
	xc := newExpSIMDConsts()
	var probe simd.Float32s
	L := probe.Len()

	i := 0
	for ; i+L <= n; i += L {
		x := simd.LoadFloat32s(src[i : i+L])
		erf := erfVec(x.Mul(ec.invSqrt2), &ec, &xc)
		ec.half.Mul(x).Mul(ec.one.Add(erf)).Store(dst[i : i+L])
	}
	if i < n {
		x, _ := simd.LoadFloat32sPart(src[i:])
		erf := erfVec(x.Mul(ec.invSqrt2), &ec, &xc)
		ec.half.Mul(x).Mul(ec.one.Add(erf)).StorePart(dst[i:])
	}
}

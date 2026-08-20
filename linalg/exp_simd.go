//go:build goexperiment.simd

package linalg

import "simd"

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
	i127                                   simd.Int32s
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
		i127:      simd.BroadcastInt32s(127),
	}
}

// expF32CoreVec is expF32Core (exp.go), vectorized — same constants, same
// round-magic trick, same Cephes minimax chain, same op order. Same contract
// as the scalar expF32Core: caller has bounded x <= 0 (softmax's domain
// after subtracting the row max), so the e>=255 overflow branch expF32Core
// itself guards is unreachable here and not replicated — this is NOT a
// vectorization of the general-purpose ExpF32 (which still has zero
// production callers as of this writing and is intentionally left
// unvectorized rather than replicating its NaN/overflow guards for no real
// caller).
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

	// 2^k by bit construction: (k+127)<<23 reinterpreted as float32.
	// x <= 0 => k <= 0 => e <= 127: the e>=255 overflow branch is unreachable.
	e := kf.ConvertToInt32().Add(c.i127)
	scale := e.ShiftAllLeft(23).ToBits().BitsToFloat32()
	res = res.Mul(scale)

	// Lanes with x < underflow have e <= 0; e=0 self-flushes via a zero
	// scale but e<0 shifts sign bits into garbage — mask keeps only
	// in-domain lanes (Masked zeroes where the mask is false).
	return res.Masked(x.GreaterEqual(c.underflow))
}

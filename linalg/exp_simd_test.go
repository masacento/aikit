//go:build goexperiment.simd

package linalg

import (
	"math"
	"math/rand"
	"testing"

	"simd"
)

// ulpDiff32 is the distance in float32 representation order (0 = bit-equal).
func ulpDiff32(a, b float32) uint32 {
	ia, ib := int64(orderedBits32(a)), int64(orderedBits32(b))
	d := ia - ib
	if d < 0 {
		d = -d
	}
	return uint32(d)
}

func orderedBits32(f float32) uint32 {
	b := math.Float32bits(f)
	if b&0x80000000 != 0 {
		return ^b
	}
	return b | 0x80000000
}

// TestExpF32CoreVec_matchesScalarULP is expF32CoreVec's accuracy gate: the
// bound this GOEXPERIMENT=simd build depends on. expF32CoreVec's MulAdd is a
// fused op, and Go's scalar compiler already fuses p*r+c on arm64 but not
// amd64, so the vector kernel can differ from the scalar expF32Core by up to
// 1 ULP — validated against the prototype's own sweep
// (~/tmcode/go127-simd-audit, 2026-08-20: max 1 ULP over the full softmax
// domain on both apple-m1pro and nvidia-rtx2070s).
func TestExpF32CoreVec_matchesScalarULP(t *testing.T) {
	const n = 1 << 20
	c := newExpSIMDConsts()
	src := make([]float32, n)
	for i := range src {
		// Dense sweep of [-87.4, 0], crossing the underflow flush at -87.33654.
		src[i] = -87.4 * float32(i) / float32(n-1)
	}
	copy(src, []float32{0, -0.0, -1e-30, -87.33, -87.336, -87.337, -87.34, -50, -1, -0.5})

	got := make([]float32, n)
	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= n; i += L {
		expF32CoreVec(simd.LoadFloat32s(src[i:i+L]), &c).Store(got[i : i+L])
	}
	if i < n {
		v, _ := simd.LoadFloat32sPart(src[i:])
		expF32CoreVec(v, &c).StorePart(got[i:])
	}

	var maxULP uint32
	var argMax float32
	for i, x := range src {
		want := expF32Core(x)
		if x < expUnderflowF32 {
			if got[i] != 0 {
				t.Fatalf("x=%g: flush-to-zero expected, got %g", x, got[i])
			}
			continue
		}
		if d := ulpDiff32(got[i], want); d > maxULP {
			maxULP, argMax = d, x
		}
	}
	t.Logf("max vector-vs-scalar diff: %d ULP at x=%g", maxULP, argMax)
	if maxULP > 2 {
		t.Fatalf("max ULP diff %d at x=%g exceeds bound 2", maxULP, argMax)
	}
}

// TestSoftmaxRowIntoRaw_tail checks every remainder length against the
// non-SIMD reference (exp_scalar.go's algorithm, reproduced here since that
// file is excluded from this build) — the SIMD kernel's tail handling
// (LoadFloat32sPart/StorePart) is exactly the part a fixed-size sweep like
// TestSoftmaxRowInto_matchesReference (exp_test.go, runs in both builds)
// under-covers, since its n values don't walk every lane-count remainder.
func TestSoftmaxRowIntoRaw_tail(t *testing.T) {
	var probe simd.Float32s
	L := probe.Len()
	rng := rand.New(rand.NewSource(29))
	for n := 1; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64()) * 6
		}
		got := make([]float32, n)
		softmaxRowIntoRaw(got, src)

		maxV := float64(src[0])
		for _, v := range src[1:] {
			if float64(v) > maxV {
				maxV = float64(v)
			}
		}
		var sum float64
		ref := make([]float64, n)
		for i, v := range src {
			ref[i] = math.Exp(float64(v) - maxV)
			sum += ref[i]
		}
		for i := range ref {
			ref[i] /= sum
			if d := math.Abs(float64(got[i]) - ref[i]); d > 1e-6 {
				t.Fatalf("n=%d i=%d: got %g want %g (diff %g)", n, i, got[i], ref[i], d)
			}
		}
	}
}

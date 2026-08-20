//go:build goexperiment.simd

package encoder

import (
	"math"
	"math/rand"
	"testing"

	"simd"
)

// log1pF32Core is the scalar form of the same algorithm log1pF32CoreVec
// vectorizes — reproduced here (not shared with log1p_simd.go's vector code)
// so this test can isolate "does the algorithm match math.Log1p" from "does
// the vector code match the scalar form of the same algorithm", the same
// two-question shape linalg's exp_simd_test.go uses.
func log1pF32Core(x float32) float32 {
	const (
		threshold = 1.0 / (1 << 12)
		sqrtHF    = 0.70710678
	)
	if x < threshold {
		return x - 0.5*x*x
	}
	u := x + 1.0
	bits := math.Float32bits(u)
	e := int32((bits>>23)&0xFF) - 126
	mBits := (bits &^ 0x7F800000) | (126 << 23)
	m := math.Float32frombits(mBits)
	if m < sqrtHF {
		e--
		m = m + m - 1.0
	} else {
		m = m - 1.0
	}
	z := m * m
	p := 7.0376836292e-2*m - 1.1514610310e-1
	p = p*m + 1.1676998740e-1
	p = p*m - 1.2420140846e-1
	p = p*m + 1.4249322787e-1
	p = p*m - 1.6668057665e-1
	p = p*m + 2.0000714765e-1
	p = p*m - 2.4999993993e-1
	p = p*m + 3.3333331174e-1
	y := p * m * z
	if e != 0 {
		fe := float32(e)
		y += -2.12194440e-4 * fe
	}
	y += -0.5 * z
	result := m + y
	if e != 0 {
		result += 0.693359375 * float32(e)
	}
	return result
}

// TestLog1pF32Core_matchesMathLog1p is log1pF32Core's accuracy gate: the
// contract this GOEXPERIMENT=simd build depends on. Validated externally
// before landing (a throwaway sweep script) at ~3e-7 max absolute error over
// x in [0, 1e10] — the bound here is set with headroom over that measurement.
// pooled[v] is always >= 0 (seeded at 0, only ever raised by max), so the
// SPLADE domain never exercises negative x; this only tests x >= 0.
func TestLog1pF32Core_matchesMathLog1p(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	var worstAbs float64
	var atAbs float32
	check := func(x float32) {
		got := log1pF32Core(x)
		want := math.Log1p(float64(x))
		d := math.Abs(float64(got) - want)
		if d > worstAbs {
			worstAbs, atAbs = d, x
		}
	}
	check(0)
	for range 2_000_000 {
		var x float32
		switch rng.Intn(5) {
		case 0:
			x = float32(rng.Float64()) * 1e-3
		case 1:
			x = float32(rng.Float64()) * 0.5
		case 2:
			x = float32(rng.Float64()) * 5
		case 3:
			x = float32(rng.Float64()) * 50
		case 4:
			x = float32(rng.Float64()) * 1000
		}
		check(x)
	}
	t.Logf("max abs error vs math.Log1p: %.3e at x=%v", worstAbs, atAbs)
	if worstAbs > 1e-6 {
		t.Errorf("max abs error %.3e at x=%v exceeds 1e-6", worstAbs, atAbs)
	}
}

// TestLog1pF32CoreVec_matchesScalar isolates the vector-vs-scalar gap for
// the SAME algorithm, over the same domain, plus every SIMD lane-count tail
// remainder.
func TestLog1pF32CoreVec_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	var probe simd.Float32s
	L := probe.Len()

	var worstAbs float64
	var atAbs float32
	for n := 0; n <= 3*L+1; n++ {
		src := make([]float32, n)
		for i := range src {
			switch rng.Intn(5) {
			case 0:
				src[i] = float32(rng.Float64()) * 1e-3
			case 1:
				src[i] = float32(rng.Float64()) * 0.5
			case 2:
				src[i] = float32(rng.Float64()) * 5
			case 3:
				src[i] = float32(rng.Float64()) * 50
			case 4:
				src[i] = float32(rng.Float64()) * 1000
			}
		}
		got := append([]float32(nil), src...)
		log1pPoolInto(got)
		for i, x := range src {
			want := log1pF32Core(x)
			d := math.Abs(float64(got[i] - want))
			if d > worstAbs {
				worstAbs, atAbs = d, x
			}
			if d > 1e-6 {
				t.Fatalf("n=%d i=%d x=%v: vec=%v scalar=%v diff=%v", n, i, x, got[i], want, d)
			}
		}
	}
	t.Logf("max abs diff vec vs scalar: %.3e at x=%v", worstAbs, atAbs)
}

// TestLog1pPoolInto_zeroIsIdentity confirms the mask-free design's load-
// bearing claim: log1pF32Core(0) == 0 exactly, so applying it unconditionally
// (SIMD build) matches the default build's `x > 0` skip.
func TestLog1pPoolInto_zeroIsIdentity(t *testing.T) {
	if got := log1pF32Core(0); got != 0 {
		t.Fatalf("log1pF32Core(0) = %v, want exactly 0", got)
	}
	pooled := make([]float32, 100)
	log1pPoolInto(pooled)
	for i, v := range pooled {
		if v != 0 {
			t.Fatalf("pooled[%d] = %v, want 0 (all-zero input)", i, v)
		}
	}
}

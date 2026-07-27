package encoder

import (
	"math/rand"
	"testing"
)

// Benchmarks for the MoE FFN's expert projections (audit #10). Per token, per
// selected expert, the forward applies W1 [D→intermediate] and W2 [intermediate→D];
// before #10 W2 was a scalar triple-loop. Shapes match nomic-embed-text-v2-moe
// (E=8, I=3072, D=768, top-2), so each projection is M=1, K·N ≈ 2.36M FLOPs.
//
// The three W2 micro-benchmarks document why moeMLP calls the register-blocked
// kernel DIRECTLY rather than matmulBTInto: at 2.36M FLOPs the shape is under
// matmulBTInto's 4M blocking threshold, so it dispatches to the naive dot-product
// — which is slower than even the old scalar AXPY. The blocked kernel is the actual
// win. On an M-series dev box:
//
//	MoEW2_scalar     ~1075 µs/op   the pre-#10 hand-written AXPY
//	MoEW2_dispatch   ~2940 µs/op   what matmulBTInto picks here (naive dot-product)
//	MoEW2_blocked     ~300 µs/op   what moeMLP now uses — ~3.5× the AXPY, ~10× naive

const (
	moeI = 3072
	moeD = 768
)

func randMoEVec(seed int64, n int) []float32 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64()) * 0.1
	}
	return v
}

// transpose [I,D] → [D,I], the layout the loader stores (transposeExpertsW2).
func transposeID(w2 []float32, I, D int) []float32 {
	out := make([]float32, I*D)
	for i := range I {
		for j := range D {
			out[j*I+i] = w2[i*D+j]
		}
	}
	return out
}

// BenchmarkMoEW2_scalar is the pre-#10 kernel: out_j += best·x1_i·W2[i,j], scalar.
func BenchmarkMoEW2_scalar(b *testing.B) {
	x1 := randMoEVec(1, moeI)
	w2 := randMoEVec(2, moeI*moeD) // [I, D]
	out := make([]float32, moeD)
	const best = float32(0.6)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		clear(out)
		for i := range moeI {
			wRow := w2[i*moeD : (i+1)*moeD]
			xi := x1[i]
			if xi == 0 {
				continue
			}
			for j := range moeD {
				out[j] += best * xi * wRow[j]
			}
		}
	}
	sink(out)
}

// BenchmarkMoEW2_dispatch is what matmulBTInto selects at this shape: because
// M·K·N < 4M it takes the naive dot-product path, not the blocked kernel.
func BenchmarkMoEW2_dispatch(b *testing.B) {
	x1 := randMoEVec(1, moeI)
	w2t := transposeID(randMoEVec(2, moeI*moeD), moeI, moeD) // [D, I]
	out := make([]float32, moeD)
	contrib := make([]float32, moeD)
	const best = float32(0.6)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		clear(out)
		matmulBTInto(x1, w2t, contrib, 1, moeI, moeD)
		for j := range moeD {
			out[j] += best * contrib[j]
		}
	}
	sink(out)
}

// BenchmarkMoEW2_blocked is the #10 kernel moeMLP uses: the register-blocked GEMM
// on the stored transpose, then a scaled add — same FLOPs, ~3.5× the scalar AXPY.
func BenchmarkMoEW2_blocked(b *testing.B) {
	x1 := randMoEVec(1, moeI)
	w2t := transposeID(randMoEVec(2, moeI*moeD), moeI, moeD)
	out := make([]float32, moeD)
	contrib := make([]float32, moeD)
	const best = float32(0.6)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		clear(out)
		matmulBTBlockedInto(x1, w2t, contrib, 1, moeI, moeD)
		for j := range moeD {
			out[j] += best * contrib[j]
		}
	}
	sink(out)
}

// BenchmarkMoEMLP runs the whole per-layer MoE FFN over L tokens to confirm the
// scratch-arena reuse (audit #10): after warmup it must be 0 allocs/op — the
// router scores, x1, and per-expert output all come from the pooled scratch.
func BenchmarkMoEMLP(b *testing.B) {
	const E, topK, L = 8, 2, 32
	router := randMoEVec(3, E*moeD)
	w1 := randMoEVec(4, E*moeI*moeD)
	w2t := randMoEVec(5, E*moeI*moeD) // transposed layout is irrelevant to timing/allocs
	bias := randMoEVec(6, moeD)
	h := randMoEVec(7, L*moeD)
	work := make([]float32, L*moeD)
	s := &scratch{}
	s.ensureLayer(L, moeD, moeI, 12, moeD/12, L)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		copy(work, h) // moeMLP adds into h in place; restore each iteration
		moeMLP(work, router, w1, w2t, bias, E, topK, moeD, moeI, L, s)
	}
	sink(work)
}

func sink(v []float32) {
	var s float32
	for _, x := range v {
		s += x
	}
	if s == 12345.678 { // never true; defeats dead-code elimination
		panic(s)
	}
}

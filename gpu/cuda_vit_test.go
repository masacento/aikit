//go:build linux

package gpu

import (
	"math"
	"math/rand"
	"testing"
)

// cuda_vit_test.go gates each encoder kernel INDIVIDUALLY against a Go reference that
// reimplements vision/encoder.go's arithmetic. Gating the kernels one at a time is the
// point: a whole-tower cosine can stay high while a single op is subtly wrong (a
// softmax that forgot the max-subtract still normalizes; an erf-GELU still looks like a
// GELU), and then the bug only shows on some other checkpoint. Each kernel here is
// compared against the exact formulation the CPU tower uses, at f32-vs-double
// tolerances tight enough that a wrong FORMULA cannot pass.

func vitSetup(t *testing.T) (*Device, Queue, ViT) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(d.ReleaseObjects)
	v, err := d.NewViT()
	if err != nil {
		t.Fatalf("NewViT: %v", err)
	}
	return d, d.NewCommandQueue(), v
}

func randF32(rng *rand.Rand, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64()) * scale
	}
	return out
}

// maxAbsDiff reports the worst absolute deviation between two vectors.
func maxAbsDiff(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// TestCUDA_vitLayerNorm gates LayerNorm against layerNormInto's exact formulation:
// double-accumulated mean and variance, eps added to the VARIANCE, then scale+shift.
func TestCUDA_vitLayerNorm(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(1))
	for _, shape := range []struct{ rows, dim int }{{1, 32}, {16, 32}, {7, 257}, {64, 1152}} {
		x := randF32(rng, shape.rows*shape.dim, 2)
		w := randF32(rng, shape.dim, 1)
		b := randF32(rng, shape.dim, 1)
		const eps = 1e-6

		want := make([]float32, len(x))
		for r := range shape.rows {
			xr := x[r*shape.dim : (r+1)*shape.dim]
			var mean float64
			for _, val := range xr {
				mean += float64(val)
			}
			mean /= float64(shape.dim)
			var variance float64
			for _, val := range xr {
				dv := float64(val) - mean
				variance += dv * dv
			}
			variance /= float64(shape.dim)
			inv := 1.0 / math.Sqrt(variance+eps)
			for i := range shape.dim {
				want[r*shape.dim+i] = float32((float64(xr[i])-mean)*inv)*w[i] + b[i]
			}
		}

		dx, dw, db := NewBufferOf(d, x), NewBufferOf(d, w), NewBufferOf(d, b)
		out := NewBufferLenOf[float32](d, len(x))
		if err := q.Launch(v.LayerNorm, RowGrid(shape.rows),
			Arg(dx), Arg(dw), Arg(db), Arg(out),
			ArgValue(int32(shape.rows)), ArgValue(int32(shape.dim)), ArgValue(float32(eps))); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, len(x))
		if err := Download(out, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(got, want); dmax > 1e-5 {
			t.Fatalf("rows=%d dim=%d: worst Δ %.3g", shape.rows, shape.dim, dmax)
		}
		for _, bb := range []Buffer{dx, dw, db, out} {
			d.ReleaseBuf(bb)
		}
	}
	t.Log("layernorm ≡ CPU layerNormInto across 4 shapes")
}

// TestCUDA_vitGELU gates the TANH-approximation GELU. The break-it-first here is
// implicit but real: erf-GELU agrees with tanh-GELU to ~1e-3, so the 1e-6 bound below
// would reject it — this test distinguishes the two formulations, which is the whole
// reason to pin it.
func TestCUDA_vitGELU(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(2))
	x := randF32(rng, 4096, 3)
	const c = 0.7978845608028654
	want := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		want[i] = float32(0.5 * vv * (1.0 + math.Tanh(c*(vv+0.044715*vv*vv*vv))))
	}
	dx := NewBufferOf(d, x)
	if err := q.Launch(v.GELUTanh, Grid1D(len(x), 256), Arg(dx), ArgValue(int32(len(x)))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, len(x))
	if err := Download(dx, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dmax := maxAbsDiff(got, want); dmax > 1e-6 {
		t.Fatalf("worst Δ %.3g — wrong GELU formulation?", dmax)
	}

	// erf-GELU must NOT pass this bound, proving the test discriminates between the
	// two formulations rather than just "looks like a GELU".
	erf := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		erf[i] = float32(0.5 * vv * (1.0 + math.Erf(vv/math.Sqrt2)))
	}
	if dmax := maxAbsDiff(erf, want); dmax <= 1e-6 {
		t.Error("break-it-first vacuous: erf-GELU is indistinguishable at this tolerance")
	} else {
		t.Logf("gelu_tanh ≡ CPU geluTanh; erf-GELU rejected at Δ %.3g", dmax)
	}
}

// TestCUDA_vitQuantRows gates the per-row int8 quantizer against linalg's
// quantizeRowInt8, including the all-zero row (scale 0) edge case. Quantization must
// be byte-exact — it is integer output, so any deviation is a real bug.
func TestCUDA_vitQuantRows(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(3))
	const rows, dim = 12, 97
	x := randF32(rng, rows*dim, 2)
	for i := range dim { // row 5 is all zeros
		x[5*dim+i] = 0
	}
	wantQ := make([]int8, rows*dim)
	wantS := make([]float32, rows)
	for r := range rows {
		xr := x[r*dim : (r+1)*dim]
		var maxAbs float32
		for _, val := range xr {
			if val < 0 {
				val = -val
			}
			if val > maxAbs {
				maxAbs = val
			}
		}
		if maxAbs == 0 {
			continue
		}
		scale := maxAbs / 127
		wantS[r] = scale
		inv := 1.0 / scale
		for i, val := range xr {
			vv := math.Round(float64(val * inv))
			if vv > 127 {
				vv = 127
			} else if vv < -127 {
				vv = -127
			}
			wantQ[r*dim+i] = int8(vv)
		}
	}

	dx := NewBufferOf(d, x)
	dq := NewBufferLenOf[int8](d, rows*dim)
	ds := NewBufferLenOf[float32](d, rows)
	if err := q.Launch(v.QuantRows, RowGrid(rows),
		Arg(dx), Arg(dq), Arg(ds), ArgValue(int32(rows)), ArgValue(int32(dim))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	gotQ := make([]int8, rows*dim)
	gotS := make([]float32, rows)
	if err := Download(dq, gotQ); err != nil {
		t.Fatalf("Download q: %v", err)
	}
	if err := Download(ds, gotS); err != nil {
		t.Fatalf("Download s: %v", err)
	}
	for i := range wantQ {
		if gotQ[i] != wantQ[i] {
			t.Fatalf("q[%d] = %d, want %d (row %d)", i, gotQ[i], wantQ[i], i/dim)
		}
	}
	for r := range rows {
		if gotS[r] != wantS[r] {
			t.Fatalf("scale[%d] = %v, want %v", r, gotS[r], wantS[r])
		}
	}
	if gotS[5] != 0 {
		t.Errorf("all-zero row scale = %v, want 0", gotS[5])
	}
	t.Log("quant_rows byte-exact vs linalg quantizeRowInt8, incl. the all-zero row")
}

// TestCUDA_vitGEMMs gates both GEMMs against exact references. The W8A8 one is sharp:
// the int8 dot is exact integer arithmetic, so only the final rescale rounds.
func TestCUDA_vitGEMMs(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(4))
	for _, shape := range []struct{ M, N, K int }{{1, 8, 16}, {16, 32, 32}, {17, 33, 65}} {
		// --- W8A8 ---
		A := make([]int8, shape.M*shape.K)
		B := make([]int8, shape.N*shape.K)
		for i := range A {
			A[i] = int8(rng.Intn(255) - 127)
		}
		for i := range B {
			B[i] = int8(rng.Intn(255) - 127)
		}
		as := randF32(rng, shape.M, 0.01)
		bs := randF32(rng, shape.N, 0.01)
		want := make([]float32, shape.M*shape.N)
		for m := range shape.M {
			for n := range shape.N {
				var acc int32
				for k := range shape.K {
					acc += int32(A[m*shape.K+k]) * int32(B[n*shape.K+k])
				}
				want[m*shape.N+n] = float32(acc) * as[m] * bs[n]
			}
		}
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
		dC := NewBufferLenOf[float32](d, shape.M*shape.N)
		if err := q.Launch(v.GEMMW8A8, Grid1D(shape.M*shape.N, 256),
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(dC),
			ArgValue(int32(shape.M)), ArgValue(int32(shape.N)), ArgValue(int32(shape.K))); err != nil {
			t.Fatalf("W8A8 Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, shape.M*shape.N)
		if err := Download(dC, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(got, want); dmax > 1e-4 {
			t.Fatalf("W8A8 M=%d N=%d K=%d: worst Δ %.3g", shape.M, shape.N, shape.K, dmax)
		}

		// --- f32 ---
		Af := randF32(rng, shape.M*shape.K, 1)
		Bf := randF32(rng, shape.N*shape.K, 1)
		wantF := make([]float32, shape.M*shape.N)
		for m := range shape.M {
			for n := range shape.N {
				var acc float32
				for k := range shape.K {
					acc += Af[m*shape.K+k] * Bf[n*shape.K+k]
				}
				wantF[m*shape.N+n] = acc
			}
		}
		dAf, dBf := NewBufferOf(d, Af), NewBufferOf(d, Bf)
		dCf := NewBufferLenOf[float32](d, shape.M*shape.N)
		if err := q.Launch(v.GEMMF32, Grid1D(shape.M*shape.N, 256),
			Arg(dAf), Arg(dBf), Arg(dCf),
			ArgValue(int32(shape.M)), ArgValue(int32(shape.N)), ArgValue(int32(shape.K))); err != nil {
			t.Fatalf("f32 Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		gotF := make([]float32, shape.M*shape.N)
		if err := Download(dCf, gotF); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(gotF, wantF); dmax > 1e-3 {
			t.Fatalf("f32 M=%d N=%d K=%d: worst Δ %.3g", shape.M, shape.N, shape.K, dmax)
		}
		for _, bb := range []Buffer{dA, dB, dAs, dBs, dC, dAf, dBf, dCf} {
			d.ReleaseBuf(bb)
		}
	}
	t.Log("gemm_w8a8 ≡ exact int32 reference; gemm_f32 ≡ f32 reference (3 shapes each)")
}

// TestCUDA_vitAttention gates bidirectional multi-head attention against a Go
// reference using the CPU tower's exact softmax (max-subtract, double sum) and its
// 1/sqrt(hd) pre-softmax scaling.
func TestCUDA_vitAttention(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(5))
	for _, shape := range []struct{ np, nH, hd int }{{16, 2, 16}, {5, 1, 8}, {64, 4, 24}} {
		hidden := shape.nH * shape.hd
		qv := randF32(rng, shape.np*hidden, 1)
		kv := randF32(rng, shape.np*hidden, 1)
		vv := randF32(rng, shape.np*hidden, 1)
		scale := float32(1.0 / math.Sqrt(float64(shape.hd)))

		want := make([]float32, shape.np*hidden)
		scores := make([]float64, shape.np)
		for h := range shape.nH {
			off := h * shape.hd
			for i := range shape.np {
				maxv := math.Inf(-1)
				for j := range shape.np {
					var acc float32
					for dd := range shape.hd {
						acc += qv[i*hidden+off+dd] * kv[j*hidden+off+dd]
					}
					s := float64(acc * scale)
					scores[j] = s
					if s > maxv {
						maxv = s
					}
				}
				var sum float64
				for j := range shape.np {
					e := math.Exp(scores[j] - maxv)
					scores[j] = e
					sum += e
				}
				for dd := range shape.hd {
					var acc float64
					for j := range shape.np {
						acc += (scores[j] / sum) * float64(vv[j*hidden+off+dd])
					}
					want[i*hidden+off+dd] = float32(acc)
				}
			}
		}

		dq, dk, dv := NewBufferOf(d, qv), NewBufferOf(d, kv), NewBufferOf(d, vv)
		out := NewBufferLenOf[float32](d, shape.np*hidden)
		if err := q.Launch(v.Attention, AttentionGrid(shape.np, shape.nH),
			Arg(dq), Arg(dk), Arg(dv), Arg(out),
			ArgValue(int32(shape.np)), ArgValue(int32(shape.nH)), ArgValue(int32(shape.hd)),
			ArgValue(scale)); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, shape.np*hidden)
		if err := Download(out, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(got, want); dmax > 1e-5 {
			t.Fatalf("np=%d nH=%d hd=%d: worst Δ %.3g", shape.np, shape.nH, shape.hd, dmax)
		}
		for _, bb := range []Buffer{dq, dk, dv, out} {
			d.ReleaseBuf(bb)
		}
	}
	t.Log("attention ≡ CPU bidirectional MHA across 3 shapes")
}

// TestCUDA_vitAddOps gates the two broadcast/elementwise adds.
func TestCUDA_vitAddOps(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(6))
	const rows, dim = 9, 40
	x := randF32(rng, rows*dim, 1)
	bias := randF32(rng, dim, 1)
	vec := randF32(rng, rows*dim, 1)

	wantBias := make([]float32, len(x))
	for r := range rows {
		for i := range dim {
			wantBias[r*dim+i] = x[r*dim+i] + bias[i]
		}
	}
	wantVec := make([]float32, len(x))
	for i := range x {
		wantVec[i] = x[i] + vec[i]
	}

	dx := NewBufferOf(d, x)
	if err := q.Launch(v.AddBias, Grid1D(rows*dim, 256), Arg(dx), Arg(NewBufferOf(d, bias)),
		ArgValue(int32(rows)), ArgValue(int32(dim))); err != nil {
		t.Fatalf("add_bias Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, len(x))
	if err := Download(dx, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dmax := maxAbsDiff(got, wantBias); dmax > 1e-6 {
		t.Fatalf("add_bias worst Δ %.3g", dmax)
	}

	dx2 := NewBufferOf(d, x)
	if err := q.Launch(v.AddVec, Grid1D(rows*dim, 256), Arg(dx2), Arg(NewBufferOf(d, vec)),
		ArgValue(int32(rows*dim))); err != nil {
		t.Fatalf("add_vec Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := Download(dx2, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dmax := maxAbsDiff(got, wantVec); dmax > 1e-6 {
		t.Fatalf("add_vec worst Δ %.3g", dmax)
	}
	t.Log("add_bias / add_vec ≡ CPU references")
}

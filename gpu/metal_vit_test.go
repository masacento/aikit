//go:build darwin

package gpu

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// metal_vit_test.go gates each Metal encoder kernel INDIVIDUALLY against a Go reference
// that reimplements vision/encoder.go's arithmetic — the same discipline as
// cuda_vit_test.go. Gating one op at a time is the point: a tower cosine stays high while
// a single op is subtly wrong. The tolerances differ from the CUDA bars where a kernel's
// reduction ran in double on CUDA (MSL has no double); each such case states the achieved
// f32 number and the reason. The exact ones (int8 dot, elementwise adds, byte-exact quant)
// hold the same tight bounds.

func vitSetupM(t *testing.T) (*Device, Queue, ViT) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })
	v, err := d.NewViT()
	if err != nil {
		t.Fatalf("NewViT: %v", err)
	}
	return d, d.NewCommandQueue(), v
}

func randF32M(rng *rand.Rand, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64()) * scale
	}
	return out
}

func maxAbsDiffM(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// i32b/f32b bind an int/float scalar as a 1-element buffer — Metal has no by-value arg.
func i32b(d *Device, v int) Buffer     { return d.NewBufferU32(uint32(int32(v))) }
func f32b(d *Device, v float32) Buffer { return d.NewBufferFloats([]float32{v}) }

func mtg(n int) int {
	if n < ViTBlock {
		return n
	}
	return ViTBlock
}

// run1d / run1dTG pin the OS thread across the dispatch: Run1D allocs+drains an
// NSAutoreleasePool, which objc requires on one thread (a migrating goroutine crashes).
func run1d(q Queue, p Pipeline, n, tg int, bufs ...Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	q.Run1D(p, n, tg, bufs...)
}
func run1dTG(q Queue, p Pipeline, n, tg, tgBytes int, bufs ...Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	q.Run1DTG(p, n, tg, tgBytes, bufs...)
}

func TestMetal_vitLayerNorm(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(1))
	worstAll := 0.0
	for _, shape := range []struct{ rows, dim int }{{1, 32}, {16, 32}, {7, 257}, {64, 1152}} {
		x := randF32M(rng, shape.rows*shape.dim, 2)
		w := randF32M(rng, shape.dim, 1)
		b := randF32M(rng, shape.dim, 1)
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
		dx, dw, db := d.NewBufferFloats(x), d.NewBufferFloats(w), d.NewBufferFloats(b)
		out := d.NewBufferLen(len(x))
		run1d(q, v.LayerNorm, shape.rows*ViTBlock, ViTBlock,
			dx, dw, db, out, i32b(d, shape.rows), i32b(d, shape.dim), f32b(d, eps))
		if dmax := maxAbsDiffM(out.Floats(), want); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("layernorm f32-reduction worst Δ %.3g vs double CPU (CUDA bar was 1e-5)", worstAll)
	// f32 pairwise reduction, not double — allow a looser, stated bound.
	if worstAll > 5e-5 {
		t.Errorf("layernorm worst Δ %.3g exceeds the stated f32 bound 5e-5", worstAll)
	}
}

func TestMetal_vitGELU(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(2))
	x := randF32M(rng, 4096, 3)
	const c = 0.7978845608028654
	want := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		want[i] = float32(0.5 * vv * (1.0 + math.Tanh(c*(vv+0.044715*vv*vv*vv))))
	}
	dx := d.NewBufferFloats(x)
	run1d(q, v.GELUTanh, len(x), mtg(len(x)), dx, i32b(d, len(x)))
	got := append([]float32(nil), dx.Floats()...)
	dmax := maxAbsDiffM(got, want)
	t.Logf("gelu_tanh f32 worst Δ %.3g vs double CPU (CUDA bar was 1e-6)", dmax)
	if dmax > 1e-5 {
		t.Errorf("gelu worst Δ %.3g exceeds the stated f32 bound 1e-5", dmax)
	}
	// Break-it-first: erf-GELU must be rejected by the discriminating bound — proving
	// the kernel is tanh-GELU, not just "a GELU". erf vs tanh differ ~5e-4.
	erf := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		erf[i] = float32(0.5 * vv * (1.0 + math.Erf(vv/math.Sqrt2)))
	}
	if dmax := maxAbsDiffM(erf, want); dmax <= 1e-5 {
		t.Error("break-it-first vacuous: erf-GELU indistinguishable at 1e-5")
	} else {
		t.Logf("erf-GELU rejected at Δ %.3g — the gate is tanh-specific", dmax)
	}
}

func TestMetal_vitQuantRows(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(3))
	const rows, dim = 12, 97
	x := randF32M(rng, rows*dim, 2)
	for i := range dim {
		x[5*dim+i] = 0 // all-zero row
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
	dx := d.NewBufferFloats(x)
	dq := d.NewBufferInt8(make([]int8, rows*dim))
	ds := d.NewBufferLen(rows)
	run1d(q, v.QuantRows, rows*ViTBlock, ViTBlock, dx, dq, ds, i32b(d, rows), i32b(d, dim))
	gotQ, gotS := dq.Int8s(), ds.Floats()
	for i := range wantQ {
		if gotQ[i] != wantQ[i] {
			t.Fatalf("q[%d]=%d want %d (row %d)", i, gotQ[i], wantQ[i], i/dim)
		}
	}
	for r := range rows {
		if gotS[r] != wantS[r] {
			t.Fatalf("scale[%d]=%v want %v", r, gotS[r], wantS[r])
		}
	}
	if gotS[5] != 0 {
		t.Errorf("all-zero row scale=%v want 0", gotS[5])
	}
	t.Log("quant_rows byte-exact vs linalg quantizeRowInt8 (incl. all-zero row)")
}

func TestMetal_vitGEMMs(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(4))
	for _, s := range []struct{ M, N, K int }{{1, 8, 16}, {16, 32, 32}, {17, 33, 65}} {
		A := make([]int8, s.M*s.K)
		B := make([]int8, s.N*s.K)
		for i := range A {
			A[i] = int8(rng.Intn(255) - 127)
		}
		for i := range B {
			B[i] = int8(rng.Intn(255) - 127)
		}
		as := randF32M(rng, s.M, 0.01)
		bs := randF32M(rng, s.N, 0.01)
		want := make([]float32, s.M*s.N)
		for m := range s.M {
			for n := range s.N {
				var acc int32
				for k := range s.K {
					acc += int32(A[m*s.K+k]) * int32(B[n*s.K+k])
				}
				want[m*s.N+n] = float32(acc) * as[m] * bs[n]
			}
		}
		dC := d.NewBufferLen(s.M * s.N)
		run1d(q, v.GEMMW8A8, s.M*s.N, mtg(s.M*s.N),
			d.NewBufferInt8(A), d.NewBufferFloats(as), d.NewBufferInt8(B), d.NewBufferFloats(bs), dC,
			i32b(d, s.M), i32b(d, s.N), i32b(d, s.K))
		if dmax := maxAbsDiffM(dC.Floats(), want); dmax > 1e-4 {
			t.Fatalf("W8A8 %dx%dx%d worst Δ %.3g", s.M, s.N, s.K, dmax)
		}
		Af := randF32M(rng, s.M*s.K, 1)
		Bf := randF32M(rng, s.N*s.K, 1)
		wantF := make([]float32, s.M*s.N)
		for m := range s.M {
			for n := range s.N {
				var acc float32
				for k := range s.K {
					acc += Af[m*s.K+k] * Bf[n*s.K+k]
				}
				wantF[m*s.N+n] = acc
			}
		}
		dCf := d.NewBufferLen(s.M * s.N)
		run1d(q, v.GEMMF32, s.M*s.N, mtg(s.M*s.N),
			d.NewBufferFloats(Af), d.NewBufferFloats(Bf), dCf, i32b(d, s.M), i32b(d, s.N), i32b(d, s.K))
		if dmax := maxAbsDiffM(dCf.Floats(), wantF); dmax > 1e-3 {
			t.Fatalf("f32 %dx%dx%d worst Δ %.3g", s.M, s.N, s.K, dmax)
		}
	}
	t.Log("gemm_w8a8 ≡ exact int32; gemm_f32 ≡ f32 reference (3 shapes each)")
}

func TestMetal_vitAttention(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(5))
	worstAll := 0.0
	for _, s := range []struct{ np, nH, hd int }{{16, 2, 16}, {5, 1, 8}, {64, 4, 24}} {
		hidden := s.nH * s.hd
		qv := randF32M(rng, s.np*hidden, 1)
		kv := randF32M(rng, s.np*hidden, 1)
		vv := randF32M(rng, s.np*hidden, 1)
		scale := float32(1.0 / math.Sqrt(float64(s.hd)))
		want := make([]float32, s.np*hidden)
		scores := make([]float64, s.np)
		for h := range s.nH {
			off := h * s.hd
			for i := range s.np {
				maxv := math.Inf(-1)
				for j := range s.np {
					var acc float32
					for dd := range s.hd {
						acc += qv[i*hidden+off+dd] * kv[j*hidden+off+dd]
					}
					sv := float64(acc * scale)
					scores[j] = sv
					if sv > maxv {
						maxv = sv
					}
				}
				var sum float64
				for j := range s.np {
					e := math.Exp(scores[j] - maxv)
					scores[j] = e
					sum += e
				}
				for dd := range s.hd {
					var acc float64
					for j := range s.np {
						acc += (scores[j] / sum) * float64(vv[j*hidden+off+dd])
					}
					want[i*hidden+off+dd] = float32(acc)
				}
			}
		}
		out := d.NewBufferLen(s.np * hidden)
		run1dTG(q, v.Attention, s.np*s.nH*ViTBlock, ViTBlock, s.np*4,
			d.NewBufferFloats(qv), d.NewBufferFloats(kv), d.NewBufferFloats(vv), out,
			i32b(d, s.np), i32b(d, s.nH), i32b(d, s.hd), f32b(d, scale))
		if dmax := maxAbsDiffM(out.Floats(), want); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("attention f32-softmax worst Δ %.3g vs double CPU (CUDA bar was 1e-5)", worstAll)
	if worstAll > 5e-5 {
		t.Errorf("attention worst Δ %.3g exceeds the stated f32 bound 5e-5", worstAll)
	}
}

func TestMetal_vitAddOps(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(6))
	const rows, dim = 9, 40
	x := randF32M(rng, rows*dim, 1)
	bias := randF32M(rng, dim, 1)
	vec := randF32M(rng, rows*dim, 1)
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
	dx := d.NewBufferFloats(x)
	run1d(q, v.AddBias, rows*dim, mtg(rows*dim), dx, d.NewBufferFloats(bias), i32b(d, rows), i32b(d, dim))
	if dmax := maxAbsDiffM(dx.Floats(), wantBias); dmax > 1e-6 {
		t.Fatalf("add_bias worst Δ %.3g", dmax)
	}
	dx2 := d.NewBufferFloats(x)
	run1d(q, v.AddVec, rows*dim, mtg(rows*dim), dx2, d.NewBufferFloats(vec), i32b(d, rows*dim))
	if dmax := maxAbsDiffM(dx2.Floats(), wantVec); dmax > 1e-6 {
		t.Fatalf("add_vec worst Δ %.3g", dmax)
	}
	t.Log("add_bias / add_vec ≡ CPU references")
}

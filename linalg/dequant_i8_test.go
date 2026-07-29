package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestDequantizeRowsInt8_bitIdentical is the gate the whole kernel rests on: the
// vectorized widen must equal the scalar loop BIT for bit, not approximately.
// It can, because float32(int8) is exact and the scale is a single f32 multiply
// in both paths — so any difference means a real bug (a mis-sized tail, a
// wrong lane order, a sign-extension that zero-extended), not rounding.
//
// Lengths deliberately straddle the 32-element main loop and the 8-element
// secondary loop, and include non-multiples of 8 so the Go tail runs.
func TestDequantizeRowsInt8_bitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, cols := range []int{1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 96, 127, 128, 768, 3072} {
		for _, rows := range []int{1, 3} {
			q := make([]int8, rows*cols)
			for i := range q {
				q[i] = int8(rng.Intn(256) - 128) // full range incl. -128
			}
			scales := make([]float32, rows)
			for i := range scales {
				scales[i] = float32(rng.NormFloat64()) * 0.01
			}

			want := make([]float32, rows*cols)
			for r := range rows {
				dequantRowInt8Scalar(want[r*cols:r*cols+cols], q[r*cols:r*cols+cols], scales[r])
			}
			got := make([]float32, rows*cols)
			DequantizeRowsInt8Into(got, q, scales, rows, cols)

			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("rows=%d cols=%d elem %d (row %d, col %d): got %v want %v (q=%d scale=%v)",
						rows, cols, i, i/cols, i%cols, got[i], want[i], q[i], scales[i/cols])
				}
			}
		}
	}
}

// TestDequantizeRowsInt8_extremes pins the values most likely to expose a
// sign-extension bug: -128 (only representable negative with no positive twin),
// -1 (all bits set), 0, and 127.
func TestDequantizeRowsInt8_extremes(t *testing.T) {
	const cols = 64
	q := make([]int8, cols)
	for i := range q {
		switch i % 4 {
		case 0:
			q[i] = -128
		case 1:
			q[i] = -1
		case 2:
			q[i] = 0
		case 3:
			q[i] = 127
		}
	}
	scales := []float32{0.5}
	got := make([]float32, cols)
	DequantizeRowsInt8Into(got, q, scales, 1, cols)
	for i := range q {
		want := float32(q[i]) * 0.5
		if got[i] != want {
			t.Fatalf("elem %d: q=%d got %v want %v — sign extension?", i, q[i], got[i], want)
		}
	}
}

// TestDequantizeRowsInt8_doesNotOverrun checks the kernel writes exactly
// rows*cols floats and not one more, which a 32-wide main loop makes easy to
// get wrong.
func TestDequantizeRowsInt8_doesNotOverrun(t *testing.T) {
	for _, cols := range []int{8, 33, 96, 100} {
		const guard = 8
		dst := make([]float32, cols+guard)
		for i := range dst {
			dst[i] = float32(math.NaN())
		}
		q := make([]int8, cols)
		for i := range q {
			q[i] = 1
		}
		DequantizeRowsInt8Into(dst[:cols], q, []float32{2}, 1, cols)
		for i := cols; i < len(dst); i++ {
			if !math.IsNaN(float64(dst[i])) {
				t.Fatalf("cols=%d: wrote past the end at index %d (%v)", cols, i, dst[i])
			}
		}
	}
}

func BenchmarkDequantizeRowsInt8(b *testing.B) {
	// One CodeRankEmbed fc11 weight: [3072, 768] int8 + per-row scales.
	const rows, cols = 3072, 768
	rng := rand.New(rand.NewSource(4))
	q := make([]int8, rows*cols)
	for i := range q {
		q[i] = int8(rng.Intn(256) - 128)
	}
	scales := make([]float32, rows)
	for i := range scales {
		scales[i] = 0.01
	}
	dst := make([]float32, rows*cols)

	b.Run("scalar", func(b *testing.B) {
		for b.Loop() {
			for r := range rows {
				dequantRowInt8Scalar(dst[r*cols:r*cols+cols], q[r*cols:r*cols+cols], scales[r])
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*rows*cols), "ns/elem")
	})
	b.Run("dispatch", func(b *testing.B) {
		for b.Loop() {
			DequantizeRowsInt8Into(dst, q, scales, rows, cols)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*rows*cols), "ns/elem")
	})
}

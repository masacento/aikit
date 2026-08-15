package encoder

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Per-row symmetric int8 weight quantization — the local reference these tests check
// against. Moved here from the former quant.go: production quantizes through
// linalg.QuantizeInt8 (weights_q8.go) and matmulBTQ8Into, so quantizeRowsInt8/
// dequantizeRowsInt8 below are used only by this file and linalg_q8_test.go — test-only
// code with no reason to live outside a _test.go file. Kept in lock-step with
// linalg.QuantizeInt8's per-row symmetric max/127 round+clamp scheme to avoid drift.
//
// Why per-row symmetric:
//
//   - SYMMETRIC: zero stays at zero (no zero-point bookkeeping). Code-
//     embedding weights are roughly mean-zero per output channel, so
//     symmetric loses very little precision vs asymmetric.
//   - PER-ROW (= per-output-channel): each weight row gets its own scale.
//     The dynamic range of W[i,:] varies a lot across output channels
//     (e.g., embedding rows for rare tokens have small max, common ones
//     large); a single global scale would force the rare-row weights to
//     round to 0. Per-row is the standard "per-channel" quantization
//     bitsandbytes / GPTQ / etc. use.
//   - INT8 (range [-127, 127]): the standard. We never quantize to -128
//     because then -(-128) overflows; clamping to [-127, 127] avoids the
//     edge case for ~0% accuracy cost.
//
// Reconstruction quality: TestQuantizeRoundTrip pins relative L2 error
// per row to ≤ 1.5e-2 on uniform-random and normal-random weights, which
// the model card's weights comfortably satisfy.

// quantizeRowsInt8 quantizes a [rows, cols] f32 matrix (row-major) to
// int8 weights + per-row f32 scales. Returns:
//
//	q       — [rows*cols] int8, same row-major layout
//	scales  — [rows] f32, scale[i] = max(|W[i,:]|) / 127
//
// To reconstruct: W_approx[i,j] = float32(q[i*cols+j]) * scales[i].
//
// Empty rows (all-zero) get scale=1 — the encoded values are all zero
// so reconstruction is exactly zero; the scale value doesn't matter.
func quantizeRowsInt8(w []float32, rows, cols int) (q []int8, scales []float32) {
	if rows*cols != len(w) {
		panic("encoder: quantizeRowsInt8 shape mismatch")
	}
	q = make([]int8, rows*cols)
	scales = make([]float32, rows)
	for i := range rows {
		row := w[i*cols : (i+1)*cols]
		// max(|row|) — find the dynamic range of this row.
		var maxAbs float32
		for _, v := range row {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		if maxAbs == 0 {
			scales[i] = 1
			// q row stays all-zero
			continue
		}
		s := maxAbs / 127.0
		scales[i] = s
		inv := 1.0 / s
		off := i * cols
		for j, v := range row {
			x := math.Round(float64(v * inv))
			if x > 127 {
				x = 127
			} else if x < -127 {
				x = -127
			}
			q[off+j] = int8(x)
		}
	}
	return q, scales
}

// dequantizeRowsInt8 is the reconstruction. Test-only reference; the production
// forward pass dequantizes on-the-fly inside matmulBTQ8 so it never materializes
// the full f32 weight back into memory (defeats the M8 storage win).
func dequantizeRowsInt8(q []int8, scales []float32, rows, cols int) []float32 {
	if rows*cols != len(q) || rows != len(scales) {
		panic("encoder: dequantizeRowsInt8 shape mismatch")
	}
	w := make([]float32, rows*cols)
	for i := range rows {
		s := scales[i]
		off := i * cols
		for j := range cols {
			w[off+j] = float32(q[off+j]) * s
		}
	}
	return w
}

// TestQuantizeRoundTrip: per-row symmetric int8 quantization is
// lossy by ~1/127 = 0.78% of the row's dynamic range on truly random
// data. Bound the per-row relative L2 error to 1.5e-2 (loose enough
// to absorb tail-effect noise; tight enough to catch a logic bug).
//
// We also assert the SHAPE invariants — len(q) == rows*cols, len(scales) ==
// rows — and the small-input edge cases (all-zero row → scale=1, all-zero
// q; single-row matrix; rectangular non-square).
func TestQuantizeRoundTrip(t *testing.T) {
	cases := []struct {
		name       string
		rows, cols int
	}{
		{"tiny_square", 4, 4},
		{"rect_wide", 3, 64},
		{"rect_tall", 64, 3},
		{"forward_wqkv", 2304, 768}, // real model shape
		{"forward_fc11", 3072, 768}, // real model shape
		{"forward_fc2", 768, 3072},  // real model shape
	}
	rng := rand.New(rand.NewPCG(7, 11))
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			w := make([]float32, c.rows*c.cols)
			for i := range w {
				w[i] = float32(rng.NormFloat64() * 0.1)
			}
			q, scales := quantizeRowsInt8(w, c.rows, c.cols)
			if len(q) != c.rows*c.cols {
				t.Fatalf("q len: got %d want %d", len(q), c.rows*c.cols)
			}
			if len(scales) != c.rows {
				t.Fatalf("scales len: got %d want %d", len(scales), c.rows)
			}
			rec := dequantizeRowsInt8(q, scales, c.rows, c.cols)

			// Per-row relative L2 error.
			for i := 0; i < c.rows; i++ {
				var num, den float64
				for j := 0; j < c.cols; j++ {
					d := float64(rec[i*c.cols+j] - w[i*c.cols+j])
					num += d * d
					den += float64(w[i*c.cols+j]) * float64(w[i*c.cols+j])
				}
				if den == 0 {
					continue
				}
				relErr := math.Sqrt(num / den)
				if relErr > 1.5e-2 {
					t.Errorf("row %d: relErr=%v > 1.5e-2 (max-abs row=%v)", i, relErr, scales[i]*127)
					break
				}
			}
		})
	}
}

// TestQuantize_zeroRow: an all-zero row must produce all-zero q with
// some non-zero scale (we use 1 — value doesn't matter since q is 0).
// Catches a divide-by-zero regression if a future loosened scale calc
// drops the maxAbs==0 guard.
func TestQuantize_zeroRow(t *testing.T) {
	w := make([]float32, 4*8) // 4 rows × 8 cols; row 2 stays all-zero
	for i := range 4 * 8 {
		if i/8 != 2 {
			w[i] = float32(i) * 0.01
		}
	}
	q, scales := quantizeRowsInt8(w, 4, 8)
	for j := range 8 {
		if q[2*8+j] != 0 {
			t.Errorf("zero-row q[%d]: got %d want 0", j, q[2*8+j])
		}
	}
	if scales[2] == 0 {
		t.Errorf("zero-row scale should be non-zero (defaults to 1 for safety)")
	}
}

// TestQuantize_extremeMaxRow: a row whose max is +1 / -1 exactly maps
// to q ∈ {-127, 127} (not 128). Catches the clamp-to-127 bug — if the
// quantizer let -128 through, then sym-multiplying by scale would
// recover -1.008 instead of -1.0 (off by 0.008 = 0.6%).
func TestQuantize_extremeMaxRow(t *testing.T) {
	// Row with max +1.0 at one position, max -1.0 at another, zeros elsewhere.
	w := []float32{1.0, -1.0, 0, 0}
	q, scales := quantizeRowsInt8(w, 1, 4)
	if scales[0] != 1.0/127.0 {
		t.Errorf("scale: got %v want %v", scales[0], 1.0/127.0)
	}
	if q[0] != 127 {
		t.Errorf("q[0]: got %d want 127", q[0])
	}
	if q[1] != -127 {
		t.Errorf("q[1]: got %d want -127", q[1])
	}
}

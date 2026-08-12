//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestDotI8x8_matchesScalar gates the multi-column int8 kernel against the SCALAR
// reference, not against another SIMD kernel.
//
// That choice matters. For the f32 kernels the reference had to be dotFMA8, because
// f32 addition is not associative and only an identically-ordered kernel can be
// expected to match bit-for-bit. Integer arithmetic has no such constraint: addition
// is associative and int8×int8→int32 cannot overflow for any K this library sees
// (|Σ| ≤ K·127², so K would have to exceed 133,000). So the blocked kernel must equal
// the plain scalar loop EXACTLY, and comparing against scalar tests the thing that is
// actually true rather than an accident of matching instruction order.
func TestDotI8x8_matchesScalar(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(23, 29))
	for _, n := range []int{16, 32, 48, 64, 128, 768, 3072} {
		a := make([]int8, n)
		cols := make([][]int8, 8)
		for i := range a {
			a[i] = int8(rng.IntN(256) - 128) // FULL int8 range, -128 included
		}
		for c := range cols {
			cols[c] = make([]int8, n)
			for i := range cols[c] {
				cols[c][i] = int8(rng.IntN(256) - 128)
			}
		}
		var got [8]int32
		dotI8x8AVX2(&a[0], &cols[0][0], &cols[1][0], &cols[2][0], &cols[3][0],
			&cols[4][0], &cols[5][0], &cols[6][0], &cols[7][0], n, &got)
		for c := range cols {
			var want int32
			for i := range a {
				want += int32(a[i]) * int32(cols[c][i])
			}
			if got[c] != want {
				t.Errorf("n=%d col%d: kernel %d, scalar %d", n, c, got[c], want)
			}
		}
	}
}

// TestDotI8x8_columnsAreIndependent guards the failure a shared-a kernel invites:
// an accumulator wired to the wrong column, or one column's product landing in
// another's accumulator. Every column gets a distinct constant, so a crossed wire
// changes the value rather than producing a plausible-looking alternative.
func TestDotI8x8_columnsAreIndependent(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	const n = 64
	a := make([]int8, n)
	for i := range a {
		a[i] = 1
	}
	cols := make([][]int8, 8)
	for c := range cols {
		cols[c] = make([]int8, n)
		for i := range cols[c] {
			cols[c][i] = int8(c + 1)
		}
	}
	var got [8]int32
	dotI8x8AVX2(&a[0], &cols[0][0], &cols[1][0], &cols[2][0], &cols[3][0],
		&cols[4][0], &cols[5][0], &cols[6][0], &cols[7][0], n, &got)
	for c := range cols {
		if want := int32(n * (c + 1)); got[c] != want {
			t.Errorf("col%d = %d, want %d — accumulators are crossed", c, got[c], want)
		}
	}
}

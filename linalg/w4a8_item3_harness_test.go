package linalg

import (
	"math/rand"
	"testing"
)

// docs/prompts/w4a8-item3-harness.md (goinfer) — harness phase only. Repacks
// canonical int4-packed weights (QuantizeGroupInt4Row's interleaved
// even/odd-nibble layout) into a candidate "split-half" layout IN BENCHMARK
// CODE ONLY: no change to quant.go's canonical packer, no WeightMat/.giw
// change. See dot_w4a8_arm64.s's dotW4A8SplitHalfSDOT for why this layout is
// worth a kernel: canonical packing interleaves k and k+1 per byte, so
// unpacking needs VZIP1/VZIP2 to restore sequential order before the nibbles
// can feed SDOT. Split-half packs k and k+16 per byte instead (low nibble =
// block 0, k_local 0..15; high nibble = block 1, k_local 16..31) — each
// half is ALREADY sequential once masked/shifted out, so the unpack prologue
// drops straight from {AND, SHR, ZIP1, ZIP2} to {AND, SHR}.

// canonicalNibble returns the unsigned nibble value (1..15, zero=8) at global
// position k in the canonical packed layout — the same extraction
// dotW4A8Scalar does inline, factored out so the repack and both scalar
// oracles share one definition of "what canonical packing means".
func canonicalNibble(packed []byte, k int) byte {
	b := packed[k/2]
	if k&1 == 0 {
		return b & 0x0F
	}
	return (b >> 4) & 0x0F
}

// repackSplitHalfRow converts one canonical-packed row (K a multiple of 32,
// matching dotW4A8's own fast-path contract) into the split-half layout.
// Same byte count as the input; only the bit arrangement changes.
func repackSplitHalfRow(packed []byte, K int) []byte {
	if K%32 != 0 {
		panic("repackSplitHalfRow: K must be a multiple of 32")
	}
	nGroups := K / 32
	out := make([]byte, len(packed))
	for g := 0; g < nGroups; g++ {
		gk := g * 32
		obase := g * 16
		for i := 0; i < 16; i++ {
			lo := canonicalNibble(packed, gk+i)
			hi := canonicalNibble(packed, gk+i+16)
			out[obase+i] = lo | (hi << 4)
		}
	}
	return out
}

// splitHalfNibble is canonicalNibble's counterpart for the split-half layout.
func splitHalfNibble(packed []byte, k int) byte {
	g := k / 32
	kl := k % 32
	base := g * 16
	if kl < 16 {
		return packed[base+kl] & 0x0F
	}
	return (packed[base+(kl-16)] >> 4) & 0x0F
}

// dotW4A8SplitHalfScalar is dotW4A8Scalar's split-half-layout twin. Same
// per-k values, same group/k iteration order, same int32 accumulator type —
// so this must be BIT-IDENTICAL to dotW4A8Scalar for the same logical
// weights, and the test below asserts exactly that (== not tolerance).
func dotW4A8SplitHalfScalar(act []int8, packedSH []byte, scales []float32, group, K int) float32 {
	var total float32
	nGroups := (K + group - 1) / group
	for g := range nGroups {
		ks := g * group
		ke := min(ks+group, K)
		var acc int32
		for k := ks; k < ke; k++ {
			nib := splitHalfNibble(packedSH, k)
			acc += int32(act[k]) * int32(int(nib)-8)
		}
		total += float32(acc) * scales[g]
	}
	return total
}

func TestRepackSplitHalf_roundtripsCanonical(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 50; trial++ {
		nGroups := 1 + rng.Intn(20)
		K := nGroups * 32
		w := make([]float32, K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, _ := QuantizeGroupsInt4(w, 1, K, 32)
		sh := repackSplitHalfRow(packed, K)
		for k := 0; k < K; k++ {
			want := canonicalNibble(packed, k)
			got := splitHalfNibble(sh, k)
			if got != want {
				t.Fatalf("trial %d K=%d k=%d: canonical nibble %d, split-half roundtrip gave %d", trial, K, k, want, got)
			}
		}
	}
}

// repackSplitHalf4RowBlock interleaves 4 rows' split-half-packed data,
// group-contiguous: for each 32-wide K-group, the 4 rows' 16 split-half
// bytes are stored back-to-back (row0's 16, row1's 16, row2's 16, row3's
// 16), repeated per group — a "Q4_0_4x4-style" repack (item 4,
// docs/prompts/w4a8-item3-harness.md): unlike a per-byte interleave, this
// keeps each row's own unpack sequence intact (no new shuffle cost) while
// making the 4 rows' data for one group contiguous for streaming, and lets
// a kernel load one activation chunk per group and reuse it across all 4
// rows' SDOT instead of reloading it per row.
func repackSplitHalf4RowBlock(row0, row1, row2, row3 []byte, K int) []byte {
	if K%32 != 0 {
		panic("repackSplitHalf4RowBlock: K must be a multiple of 32")
	}
	nGroups := K / 32
	bpr := K / 2
	sh0 := repackSplitHalfRow(row0, K)
	sh1 := repackSplitHalfRow(row1, K)
	sh2 := repackSplitHalfRow(row2, K)
	sh3 := repackSplitHalfRow(row3, K)
	out := make([]byte, 4*bpr)
	for g := 0; g < nGroups; g++ {
		base := g * 16
		obase := g * 64
		copy(out[obase:obase+16], sh0[base:base+16])
		copy(out[obase+16:obase+32], sh1[base:base+16])
		copy(out[obase+32:obase+48], sh2[base:base+16])
		copy(out[obase+48:obase+64], sh3[base:base+16])
	}
	return out
}

// interleaveScales4Row is repackSplitHalf4RowBlock's counterpart for the
// per-group f32 scales: 4 scales per group (row0..row3), contiguous, same
// locality reasoning.
func interleaveScales4Row(s0, s1, s2, s3 []float32, nGroups int) []float32 {
	out := make([]float32, 4*nGroups)
	for g := 0; g < nGroups; g++ {
		out[4*g] = s0[g]
		out[4*g+1] = s1[g]
		out[4*g+2] = s2[g]
		out[4*g+3] = s3[g]
	}
	return out
}

// dotW4A8SplitHalf4RowScalar is the scalar oracle for the interleaved
// 4-row-block layout: computes all 4 rows' dots, must match
// dotW4A8SplitHalfScalar called separately per row (same values, same
// group/k order, same per-row accumulator — this only changes memory
// layout, not arithmetic), which the test below asserts exactly.
func dotW4A8SplitHalf4RowScalar(act []int8, packed4 []byte, scales4 []float32, group, K int) [4]float32 {
	nGroups := (K + group - 1) / group
	bpr := K / 2
	var out [4]float32
	for row := 0; row < 4; row++ {
		var total float32
		for g := 0; g < nGroups; g++ {
			ks := g * group
			ke := min(ks+group, K)
			rowBytes := packed4[g*64+row*16 : g*64+row*16+16]
			var acc int32
			for k := ks; k < ke; k++ {
				kl := k - ks // 0..31 within this group
				var nib byte
				if kl < 16 {
					nib = rowBytes[kl] & 0x0F
				} else {
					nib = (rowBytes[kl-16] >> 4) & 0x0F
				}
				acc += int32(act[k]) * int32(int(nib)-8)
			}
			total += float32(acc) * scales4[4*g+row]
		}
		_ = bpr
		out[row] = total
	}
	return out
}

func TestRepackSplitHalf4RowBlock_matchesPerRow(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	for trial := 0; trial < 30; trial++ {
		nGroups := 1 + rng.Intn(15)
		K := nGroups * 32
		rows := make([][]byte, 4)
		scales := make([][]float32, 4)
		acts := make([]int8, K)
		for i := range acts {
			acts[i] = int8(rng.Intn(255) - 128)
		}
		for r := 0; r < 4; r++ {
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			p, s := QuantizeGroupsInt4(w, 1, K, 32)
			rows[r] = p
			scales[r] = s
		}
		packed4 := repackSplitHalf4RowBlock(rows[0], rows[1], rows[2], rows[3], K)
		scales4 := interleaveScales4Row(scales[0], scales[1], scales[2], scales[3], nGroups)

		got := dotW4A8SplitHalf4RowScalar(acts, packed4, scales4, 32, K)
		for r := 0; r < 4; r++ {
			want := dotW4A8Scalar(acts, rows[r], scales[r], 32, K)
			if got[r] != want {
				t.Fatalf("trial %d K=%d row=%d: got %v want %v", trial, K, r, got[r], want)
			}
		}
	}
}

func TestDotW4A8SplitHalfScalar_matchesCanonical_exact(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 200; trial++ {
		K := 32 * (1 + rng.Intn(20))
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(255) - 128)
		}
		w := make([]float32, K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, 1, K, 32)
		sh := repackSplitHalfRow(packed, K)

		want := dotW4A8Scalar(act, packed, scales, 32, K)
		got := dotW4A8SplitHalfScalar(act, sh, scales, 32, K)
		if got != want {
			t.Fatalf("trial %d K=%d: got %v want %v (bit mismatch)", trial, K, got, want)
		}
	}
}

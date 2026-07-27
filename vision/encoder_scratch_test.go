package vision

import (
	"math"
	"math/rand"
	"testing"
)

// synthEncoder builds a small deterministic SigLIP encoder (random weights) so the
// CPU forward can be exercised without a checkpoint.
func synthEncoder(seed int64, quant bool) ([]float32, *Encoder) {
	rng := rand.New(rand.NewSource(seed))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.1
		}
		return s
	}
	const hidden, nH, inter, nLayers = 16, 2, 32, 3
	const patch, imgSize, chans = 4, 8, 3
	cpp := chans * patch * patch
	grid := imgSize / patch
	np := grid * grid
	e := &Encoder{
		Cfg: EncoderConfig{
			HiddenSize: hidden, NumAttentionHeads: nH, IntermediateSize: inter,
			NumHiddenLayers: nLayers, NumChannels: chans, ImageSize: imgSize,
			PatchSize: patch, LayerNormEps: 1e-6,
		},
		grid: grid, numPatches: np,
		patchW: rnd(hidden * cpp), patchB: rnd(hidden), posEmb: rnd(np * hidden),
		postLNw: rnd(hidden), postLNb: rnd(hidden),
	}
	for range nLayers {
		e.layers = append(e.layers, encLayer{
			ln1w: rnd(hidden), ln1b: rnd(hidden),
			qw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), qb: rnd(hidden),
			kw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), kb: rnd(hidden),
			vw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), vb: rnd(hidden),
			ow: newQMat(rnd(hidden*hidden), hidden, hidden, quant), ob: rnd(hidden),
			ln2w: rnd(hidden), ln2b: rnd(hidden),
			fc1w: newQMat(rnd(inter*hidden), inter, hidden, quant), fc1b: rnd(inter),
			fc2w: newQMat(rnd(hidden*inter), hidden, inter, quant), fc2b: rnd(hidden),
		})
	}
	return rnd(chans * imgSize * imgSize), e
}

// TestEncoder_scratchBitIdentical is the bit-identity gate for AUDIT #5: allocating
// the per-layer scratch once per Forward (instead of a fresh set every layer) must
// not change any output value. The expected numbers were captured from the
// pre-#5 code; the SigLIP parity test (against a real checkpoint) skips in CI, so
// this synthetic forward is the local guard.
func TestEncoder_scratchBitIdentical(t *testing.T) {
	// Values captured from the pre-refactor implementation, per weight precision.
	// #5 (scratch reuse) and #12 (Workspace threading) must not move any of them.
	cases := []struct {
		name      string
		quant     bool
		wantFirst []float32
		wantSumSq float64
	}{
		{"f32", false, []float32{-0.12652746, -0.0008432368, -0.1994251, 0.014424909}, 1.136807045},
		{"int8(W8A8)", true, []float32{-0.12667489, -0.00064121914, -0.19966985, 0.014695232}, 1.136773173},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pixels, e := synthEncoder(1234, tc.quant)
			out, err := e.Forward(pixels)
			if err != nil {
				t.Fatal(err)
			}
			var sum float64
			for _, v := range out {
				sum += float64(v) * float64(v)
			}
			for i, w := range tc.wantFirst {
				if out[i] != w {
					t.Errorf("out[%d] = %v, want %v (scratch/Workspace reuse changed the numerics)", i, out[i], w)
				}
			}
			if got := math.Sqrt(sum); math.Abs(got-tc.wantSumSq) > 1e-7 {
				t.Errorf("checksum(sumsq) = %.10g, want %.10g", got, tc.wantSumSq)
			}
		})
	}
}

// benchEncoder builds a mid-size encoder (np=256, hidden=128, 12 layers) to show
// the per-layer scratch reuse (#5).
func benchEncoder(seed int64, quant bool) ([]float32, *Encoder) {
	rng := rand.New(rand.NewSource(seed))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.1
		}
		return s
	}
	const hidden, nH, inter, nLayers = 128, 8, 512, 12
	const patch, imgSize, chans = 4, 64, 3
	cpp := chans * patch * patch
	grid := imgSize / patch
	np := grid * grid // 256
	e := &Encoder{
		Cfg: EncoderConfig{HiddenSize: hidden, NumAttentionHeads: nH, IntermediateSize: inter,
			NumHiddenLayers: nLayers, NumChannels: chans, ImageSize: imgSize, PatchSize: patch, LayerNormEps: 1e-6},
		grid: grid, numPatches: np,
		patchW: rnd(hidden * cpp), patchB: rnd(hidden), posEmb: rnd(np * hidden),
		postLNw: rnd(hidden), postLNb: rnd(hidden),
	}
	for range nLayers {
		e.layers = append(e.layers, encLayer{
			ln1w: rnd(hidden), ln1b: rnd(hidden),
			qw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), qb: rnd(hidden),
			kw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), kb: rnd(hidden),
			vw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), vb: rnd(hidden),
			ow: newQMat(rnd(hidden*hidden), hidden, hidden, quant), ob: rnd(hidden),
			ln2w: rnd(hidden), ln2b: rnd(hidden),
			fc1w: newQMat(rnd(inter*hidden), inter, hidden, quant), fc1b: rnd(inter),
			fc2w: newQMat(rnd(hidden*inter), hidden, inter, quant), fc2b: rnd(hidden),
		})
	}
	return rnd(chans * imgSize * imgSize), e
}

func BenchmarkForward(b *testing.B) {
	pixels, e := benchEncoder(9, false)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = e.Forward(pixels)
	}
}

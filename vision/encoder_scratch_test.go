package vision

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// synthEncoder builds a small deterministic SigLIP encoder (random weights) so the
// CPU forward can be exercised without a checkpoint.
func synthEncoder(seed int64) ([]float32, *Encoder) {
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
			qw: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), qb: rnd(hidden),
			kw: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), kb: rnd(hidden),
			vw: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), vb: rnd(hidden),
			ow: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), ob: rnd(hidden),
			ln2w: rnd(hidden), ln2b: rnd(hidden),
			fc1w: linalg.WrapF32(rnd(inter*hidden), inter, hidden), fc1b: rnd(inter),
			fc2w: linalg.WrapF32(rnd(hidden*inter), hidden, inter), fc2b: rnd(hidden),
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
	pixels, e := synthEncoder(1234)
	out, err := e.Forward(pixels)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, v := range out {
		sum += float64(v) * float64(v)
	}
	// Exact float32 values from the pre-refactor implementation.
	wantFirst := []float32{-0.12652746, -0.0008432368, -0.1994251, 0.014424909}
	for i, w := range wantFirst {
		if out[i] != w {
			t.Errorf("out[%d] = %v, want %v (scratch reuse changed the numerics)", i, out[i], w)
		}
	}
	if got := math.Sqrt(sum); math.Abs(got-1.136807045) > 1e-7 {
		t.Errorf("checksum(sumsq) = %.10g, want 1.136807045", got)
	}
}

// benchEncoder builds a mid-size encoder (np=256, hidden=128, 12 layers) to show
// the per-layer scratch reuse (#5).
func benchEncoder(seed int64) ([]float32, *Encoder) {
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
			qw: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), qb: rnd(hidden),
			kw: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), kb: rnd(hidden),
			vw: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), vb: rnd(hidden),
			ow: linalg.WrapF32(rnd(hidden*hidden), hidden, hidden), ob: rnd(hidden),
			ln2w: rnd(hidden), ln2b: rnd(hidden),
			fc1w: linalg.WrapF32(rnd(inter*hidden), inter, hidden), fc1b: rnd(inter),
			fc2w: linalg.WrapF32(rnd(hidden*inter), hidden, inter), fc2b: rnd(hidden),
		})
	}
	return rnd(chans * imgSize * imgSize), e
}

func BenchmarkForward(b *testing.B) {
	pixels, e := benchEncoder(9)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = e.Forward(pixels)
	}
}

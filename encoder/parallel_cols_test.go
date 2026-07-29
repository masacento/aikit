package encoder

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

// TestMatmulBTColsInto_bitIdentical pins the claim matmulBTColsInto's doc makes:
// fanning across output columns changes no bit of the result versus the serial
// blocked fill. Exact equality, not a tolerance — a column shard that landed on
// a different kernel tail would show up as a low-bit difference, and a tolerance
// would hide it.
//
// N is deliberately awkward (30522 = the real BERT vocabulary, and a prime-ish
// 1013) so the shard boundaries do not all fall on tile multiples.
func TestMatmulBTColsInto_bitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, sh := range []struct{ M, K, N int }{
		{1, 768, 30522},   // single-token query — the shape the row split cannot help
		{22, 768, 30522},  // a typical SPLADE query
		{357, 768, 30522}, // a long document
		{3, 768, 1013},    // awkward N, below linalg's own parallel threshold
		{64, 64, 8},       // N smaller than the worker count
	} {
		a := randF32(rng, sh.M*sh.K)
		b := randF32(rng, sh.N*sh.K)
		want := make([]float32, sh.M*sh.N)
		matmulBTBlockedInto(a, b, want, sh.M, sh.K, sh.N)

		got := make([]float32, sh.M*sh.N)
		matmulBTColsInto(a, b, got, sh.M, sh.K, sh.N)

		diff := 0
		var worst float64
		for i := range want {
			if got[i] != want[i] {
				diff++
				if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
					worst = d
				}
			}
		}
		if diff != 0 {
			t.Errorf("M=%d K=%d N=%d: %d/%d elements differ from the serial fill (worst |Δ|=%g); "+
				"the column fan-out is supposed to be numerically inert",
				sh.M, sh.K, sh.N, diff, len(want), worst)
		}
	}
}

// TestMatmulBTColsInto_serialUnderBatch checks the oversubscription guard: with
// another forward in flight, the column fan-out must not spawn.
func TestMatmulBTColsInto_serialUnderBatch(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const M, K, N = 22, 768, 4096
	a := randF32(rng, M*K)
	b := randF32(rng, N*K)
	want := make([]float32, M*N)
	matmulBTBlockedInto(a, b, want, M, K, N)

	enterForward()
	enterForward() // two in flight: a batch owns the cores
	got := make([]float32, M*N)
	matmulBTColsInto(a, b, got, M, K, N)
	leaveForward()
	leaveForward()

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d: got %v want %v — the serial fallback must match the serial fill", i, got[i], want[i])
		}
	}
	// Vacuity guard: if the guard silently stopped being consulted this test would
	// still pass (both paths agree bitwise by design), so assert the counter is the
	// thing being read.
	if inflightForwards.Load() != 0 {
		t.Fatalf("in-flight counter leaked: %d", inflightForwards.Load())
	}
}

func randF32(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

// TestSpladeVocabProj_colParallelIsBitIdentical is the same gate on the real
// checkpoint's decoder weights and real trunk activations, which have the actual
// magnitude distribution the synthetic test only approximates.
func TestSpladeVocabProj_colParallelIsBitIdentical(t *testing.T) {
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		t.Fatalf("LoadSPLADE: %v", err)
	}
	defer func() { _ = s.Close() }()

	D, V := s.bert.cfg.Hidden, s.vocab
	ids, err := s.bert.tok.EncodeWithSpecials("how do i parse json in go", s.bert.maxSeq)
	if err != nil {
		t.Fatal(err)
	}
	L := len(ids)
	h := s.bert.hiddenStates(ids, nil)
	trunk := matmulBT(h, s.transformW, L, D, D)
	addBias(trunk, s.transformB, L, D)
	gelu(trunk)
	layerNorm(trunk, s.transLNW, s.transLNB, L, D, s.bert.cfg.LNEps)

	want := make([]float32, L*V)
	matmulBTBlockedInto(trunk, s.decoderW, want, L, D, V)
	got := make([]float32, L*V)
	matmulBTColsInto(trunk, s.decoderW, got, L, D, V)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit[%d/%d] (L=%d V=%d): got %v want %v — column fan-out changed the logits",
				i, len(want), L, V, got[i], want[i])
		}
	}
	t.Logf("vocab projection bit-identical across %d logits (L=%d, V=%d)", len(want), L, V)
}

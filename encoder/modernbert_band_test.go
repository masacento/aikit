package encoder

import (
	"math"
	"testing"
)

// TestModernBERT_bandedMatchesDense gates the banded sliding-window attention path
// (bandedAttnHead) against the dense L×L mask-and-softmax path it replaces, at
// lengths long enough to trip the bandedAttn gate. The two differ only in the QKᵀ
// reduction tiling, so the hidden states should agree to f32 GEMM noise.
func TestModernBERT_bandedMatchesDense(t *testing.T) {
	m, err := LoadModernBERT("../testdata/ruri-v3-30m")
	if err != nil {
		t.Skipf("no ruri-v3-30m: %v", err)
	}
	defer m.Close()

	for _, L := range []int{512, 1000, 2048, 5002} {
		if !bandedAttnOK(m.be, m.slidingW, L) {
			t.Fatalf("L=%d: bandedAttn gate is off, test would be vacuous", L)
		}
		ids := make([]int32, L)
		for i := range ids {
			ids[i] = int32((i*7919 + 13) % 4000)
		}

		got := m.forward(ids)
		// cpuBackend.MatmulBT is matmulBTInto — the exact call the no-backend path
		// makes — so attaching it changes nothing except closing bandedAttn's gate.
		m.be = &cpuBackend{}
		want := m.forward(ids)
		m.be = nil

		if len(got) != len(want) {
			t.Fatalf("L=%d: length %d vs %d", L, len(got), len(want))
		}
		var maxAbs, maxRel float64
		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			if d > maxAbs {
				maxAbs = d
			}
			if s := math.Abs(float64(want[i])); s > 1e-3 && d/s > maxRel {
				maxRel = d / s
			}
		}
		t.Logf("L=%4d  maxΔ %.3e  maxRelΔ %.3e", L, maxAbs, maxRel)
		if maxAbs > 1e-3 {
			t.Errorf("L=%d: banded vs dense maxΔ %.3e exceeds 1e-3", L, maxAbs)
		}
	}
}

// TestModernBERTQ8_bandedMatchesDense is the same gate for the int8 forward, which
// shares the banded helper: only the projections are quantized, so its attention is
// the same f32 code and must land in the same place.
func TestModernBERTQ8_bandedMatchesDense(t *testing.T) {
	m, err := LoadModernBERTQ8("../testdata/ruri-v3-30m")
	if err != nil {
		t.Skipf("no ruri-v3-30m: %v", err)
	}
	defer m.Close()

	for _, L := range []int{512, 2048} {
		if !bandedAttnOK(m.be, m.slidingW, L) {
			t.Fatalf("L=%d: bandedAttn gate is off, test would be vacuous", L)
		}
		ids := make([]int32, L)
		for i := range ids {
			ids[i] = int32((i*7919 + 13) % 4000)
		}

		got := m.forward(ids)
		m.be = &cpuBackend{} // closes the gate; cpuBackend.MatmulBT is the same call
		want := m.forward(ids)
		m.be = nil

		var maxAbs float64
		for i := range got {
			if d := math.Abs(float64(got[i] - want[i])); d > maxAbs {
				maxAbs = d
			}
		}
		t.Logf("L=%4d  maxΔ %.3e", L, maxAbs)
		if maxAbs > 1e-3 {
			t.Errorf("L=%d: banded vs dense maxΔ %.3e exceeds 1e-3", L, maxAbs)
		}
	}
}

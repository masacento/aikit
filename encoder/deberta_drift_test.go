package encoder

import (
	"fmt"
	"math"
	"testing"
)

// TestDeBERTa_layerDrift reports the worst sample deviation at EACH depth, which is
// how you tell accumulated float32 drift from a structural bug. Drift climbs
// smoothly with depth; a wrong bucket index, gather or scale shows up as a step at
// the layer that introduces it and stays. Diagnostic only — it asserts nothing
// beyond what TestDeBERTa_parity already does.
func TestDeBERTa_layerDrift(t *testing.T) {
	d, g := loadDeBERTaFixture(t)
	D := d.HiddenDim()

	worst := make([]float64, len(g.Cases[0].Layers))
	worstLen := make([]int, len(worst))
	for _, c := range g.Cases {
		all := d.AllHiddenStates(c.InputIDs)
		for li, want := range c.Layers {
			h := all[li]
			k := 0
			for _, r := range c.Rows {
				for _, dim := range c.Dims {
					diff := math.Abs(float64(h[r*D+dim]) - float64(want.Sample[k]))
					if diff > worst[li] {
						worst[li] = diff
						worstLen[li] = c.L
					}
					k++
				}
			}
		}
	}
	for li, w := range worst {
		ratio := ""
		if li > 0 && worst[li-1] > 0 {
			ratio = fmt.Sprintf("  x%.1f", w/worst[li-1])
		}
		t.Logf("hidden[%2d] worst %.3e (at L=%d)%s", li, w, worstLen[li], ratio)
	}
}

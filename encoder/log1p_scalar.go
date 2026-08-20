//go:build !goexperiment.simd

package encoder

import "math"

// log1pPoolInto (default build) — byte-for-byte the loop splade.go's pooling
// step had before log1p_simd.go existed: log(1+x) applied only where x > 0
// (pooled is seeded at 0 and only ever raised by max, so it is never
// negative — the check just skips the no-op v==0 case, a call-count
// optimization, not a correctness requirement).
func log1pPoolInto(pooled []float32) {
	for v, x := range pooled {
		if x > 0 {
			pooled[v] = float32(math.Log1p(float64(x)))
		}
	}
}

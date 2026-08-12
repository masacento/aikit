//go:build arm64

package linalg

import "time"

// fmaPeakARM64 runs a register-saturating f32 NEON FMA loop (no memory traffic) to
// measure the core's achievable FMA throughput — used by the GEMM peak-fraction gate
// to ground the "fraction of peak" denominator in a measured ceiling, not a spec sheet.
//
//go:noescape
func fmaPeakARM64(iters int64)

// fmaPeakFlopsPerIterARM64 is one fmaPeakARM64 iteration: 20 FMLAs × 4 f32 lanes × 2 flops.
const fmaPeakFlopsPerIterARM64 = 20 * 4 * 2

// measureFMAPeak times the NEON probe and converts to GFLOP/s, taking the max of
// several short runs. See the amd64 twin for why a ceiling must be a maximum.
func measureFMAPeak() (float64, bool) {
	const (
		warm  = int64(2_000_000)
		iters = int64(10_000_000)
		runs  = 5
	)
	fmaPeakARM64(warm)
	best := 0.0
	for range runs {
		t0 := time.Now()
		fmaPeakARM64(iters)
		el := time.Since(t0)
		if el <= 0 {
			continue
		}
		if g := float64(iters) * fmaPeakFlopsPerIterARM64 / el.Seconds() / 1e9; g > best {
			best = g
		}
	}
	return best, best > 0
}

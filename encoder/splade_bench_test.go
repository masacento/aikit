package encoder

import (
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// spladeBenchText builds a query of roughly n WordPiece tokens. SPLADE's cost
// is dominated by the [L,V] vocabulary projection, so L is the knob.
func spladeBenchText(n int) string {
	var b strings.Builder
	words := []string{"how", "do", "i", "parse", "json", "in", "go", "with", "generic", "structs",
		"machine", "learning", "for", "semantic", "search", "over", "code", "repositories"}
	for i := range n {
		b.WriteString(words[i%len(words)])
		b.WriteByte(' ')
	}
	return b.String()
}

func loadBenchSPLADE(b *testing.B) *SPLADE {
	b.Helper()
	const dir = "../testdata/splade-model"
	if _, err := os.Stat(dir); err != nil {
		b.Skip("testdata/splade-model/ not present; see scripts/README.md")
	}
	s, err := LoadSPLADE(dir)
	if err != nil {
		b.Fatalf("LoadSPLADE: %v", err)
	}
	return s
}

// BenchmarkSPLADEExpand is the end-to-end number the perf campaign's SPLADE
// items are judged against: trunk forward + MLM transform + the [L,D]x[V,D]ᵀ
// vocabulary projection + pooling.
func BenchmarkSPLADEExpand(b *testing.B) {
	s := loadBenchSPLADE(b)
	for _, L := range []int{16, 64, 256} {
		text := spladeBenchText(L)
		b.Run(shapeName("L", L), func(b *testing.B) {
			for b.Loop() {
				if _, err := s.Expand(text); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func shapeName(prefix string, n int) string {
	return prefix + itoa(n)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// spladeBenchLogits produces a real [L,V] logit matrix from the checkpoint, so
// the pooling benchmarks below run on the actual positive density of a trained
// SPLADE (mostly-negative by construction) rather than a synthetic fixture that
// would flatter the hoisted form.
func spladeBenchLogits(b *testing.B, s *SPLADE, L int) ([]float32, int) {
	b.Helper()
	ids, err := s.bert.tok.EncodeWithSpecials(spladeBenchText(L), s.bert.maxSeq)
	if err != nil {
		b.Fatal(err)
	}
	D, V, n := s.bert.cfg.Hidden, s.vocab, len(ids)
	h := s.bert.hiddenStates(ids, nil)
	t := matmulBT(h, s.transformW, n, D, D)
	addBias(t, s.transformB, n, D)
	gelu(t)
	layerNorm(t, s.transLNW, s.transLNB, n, D, s.bert.cfg.LNEps)
	logits := matmulBT(t, s.decoderW, n, D, V)
	addBias(logits, s.decoderB, n, V)
	return logits, n
}

// poolPerElement is the pre-item-2 shape: math.Log1p once per POSITIVE element
// of the [L,V] logit matrix, before the max.
func poolPerElement(logits []float32, L, V int) []float32 {
	pooled := make([]float32, V)
	for i := range L {
		row := logits[i*V : (i+1)*V]
		for v, x := range row {
			if x > 0 {
				w := float32(math.Log1p(float64(x)))
				if w > pooled[v] {
					pooled[v] = w
				}
			}
		}
	}
	return pooled
}

// poolHoisted is the shipped shape: raw max, then Log1p once per vocab entry.
func poolHoisted(logits []float32, L, V int) []float32 {
	pooled := make([]float32, V)
	for i := range L {
		row := logits[i*V : (i+1)*V]
		for v, x := range row {
			if x > pooled[v] {
				pooled[v] = x
			}
		}
	}
	for v, x := range pooled {
		if x > 0 {
			pooled[v] = float32(math.Log1p(float64(x)))
		}
	}
	return pooled
}

// BenchmarkSpladePooling isolates perf-campaign item 2 on real logits.
func BenchmarkSpladePooling(b *testing.B) {
	s := loadBenchSPLADE(b)
	for _, L := range []int{16, 64, 256} {
		logits, n := spladeBenchLogits(b, s, L)
		V := s.vocab
		var pos int
		for _, x := range logits {
			if x > 0 {
				pos++
			}
		}
		b.Run(shapeName("L", n)+"/perElement", func(b *testing.B) {
			b.ReportMetric(float64(pos)/float64(len(logits))*100, "%positive")
			for b.Loop() {
				sinkF32 = poolPerElement(logits, n, V)
			}
		})
		b.Run(shapeName("L", n)+"/hoisted", func(b *testing.B) {
			for b.Loop() {
				sinkF32 = poolHoisted(logits, n, V)
			}
		})
	}
}

var sinkF32 []float32

// BenchmarkSpladeVocabProj isolates the [L,D]x[V,D]ᵀ vocabulary projection —
// perf-campaign item 7's target.
func BenchmarkSpladeVocabProj(b *testing.B) {
	s := loadBenchSPLADE(b)
	D, V := s.bert.cfg.Hidden, s.vocab
	for _, L := range []int{16, 64, 256} {
		t, n := spladeBenchTrunk(b, s, L)
		dst := make([]float32, n*V)
		b.Run(shapeName("L", n)+"/serial", func(b *testing.B) {
			for b.Loop() {
				matmulBTBlockedInto(t, s.decoderW, dst, n, D, V)
			}
		})
		b.Run(shapeName("L", n)+"/cols", func(b *testing.B) {
			for b.Loop() {
				matmulBTColsInto(t, s.decoderW, dst, n, D, V)
			}
		})
	}
}

// spladeBenchTrunk returns the [L,D] MLM-transform output feeding the
// vocabulary projection.
func spladeBenchTrunk(b *testing.B, s *SPLADE, L int) ([]float32, int) {
	b.Helper()
	ids, err := s.bert.tok.EncodeWithSpecials(spladeBenchText(L), s.bert.maxSeq)
	if err != nil {
		b.Fatal(err)
	}
	D, n := s.bert.cfg.Hidden, len(ids)
	h := s.bert.hiddenStates(ids, nil)
	t := matmulBT(h, s.transformW, n, D, D)
	addBias(t, s.transformB, n, D)
	gelu(t)
	layerNorm(t, s.transLNW, s.transLNB, n, D, s.bert.cfg.LNEps)
	return t, n
}

// BenchmarkSpladePhases times the three stages of expandIDs separately, with the
// vocabulary projection run serially vs column-parallel, in ONE process so the
// two variants see the same thermal and allocator state. The end-to-end
// before/after showed the projection getting 5x faster while Expand got *slower*
// at short L; this attributes the difference to a stage.
func BenchmarkSpladePhases(b *testing.B) {
	s := loadBenchSPLADE(b)
	D, V := s.bert.cfg.Hidden, s.vocab
	for _, L := range []int{16, 256} {
		ids, err := s.bert.tok.EncodeWithSpecials(spladeBenchText(L), s.bert.maxSeq)
		if err != nil {
			b.Fatal(err)
		}
		n := len(ids)
		for _, mode := range []string{"serial", "cols"} {
			b.Run(shapeName("L", n)+"/"+mode, func(b *testing.B) {
				logits := make([]float32, n*V)
				var trunk, proj, tail time.Duration
				iters := 0
				for b.Loop() {
					t0 := time.Now()
					h := s.bert.hiddenStates(ids, nil)
					t := matmulBT(h, s.transformW, n, D, D)
					addBias(t, s.transformB, n, D)
					gelu(t)
					layerNorm(t, s.transLNW, s.transLNB, n, D, s.bert.cfg.LNEps)
					t1 := time.Now()
					if mode == "serial" {
						matmulBTBlockedInto(t, s.decoderW, logits, n, D, V)
					} else {
						matmulBTColsInto(t, s.decoderW, logits, n, D, V)
					}
					t2 := time.Now()
					addBias(logits, s.decoderB, n, V)
					sinkF32 = poolHoisted(logits, n, V)
					t3 := time.Now()
					trunk += t1.Sub(t0)
					proj += t2.Sub(t1)
					tail += t3.Sub(t2)
					iters++
				}
				b.ReportMetric(float64(trunk.Nanoseconds())/float64(iters)/1e6, "trunk-ms")
				b.ReportMetric(float64(proj.Nanoseconds())/float64(iters)/1e6, "proj-ms")
				b.ReportMetric(float64(tail.Nanoseconds())/float64(iters)/1e6, "tail-ms")
			})
		}
	}
}

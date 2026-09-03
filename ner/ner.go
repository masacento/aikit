// Package ner runs named-entity recognition over raw text: GLiNER2 zero-shot
// boundary extraction (entity types named at call time, no fine-tune and no
// schema in the weights) and supervised BIO token classification on a
// DistilBERT trunk (TokenClassifier). Both report entities as byte offsets
// into the caller's text and are parity-pinned to their Python references.
package ner

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// linear is one affine layer: dst[M,out] = x[M,in]·Wᵀ + B, the HF nn.Linear layout.
type linear struct {
	W, B    []float32
	in, out int
}

func (l linear) apply(x []float32, rows int) []float32 {
	dst := make([]float32, rows*l.out)
	linalg.MatmulBT(x, l.W, dst, rows, l.in, l.out)
	for i := range rows {
		row := dst[i*l.out : (i+1)*l.out]
		for j := range row {
			row[j] += l.B[j]
		}
	}
	return dst
}

// Opts tunes decoding. The zero value means threshold 0.5, flat NER — GLiNER's own
// defaults.
type Opts struct {
	// Threshold is the minimum sigmoid score for a span to be considered. Zero
	// means 0.5; set a negative value to accept everything.
	Threshold float64
	// Nested allows overlapping spans (GLiNER's flat_ner=false).
	Nested bool
	// MultiLabel allows the same span to carry more than one label.
	MultiLabel bool
	// CJKSplit segments CJK word runs with the litsea word-segmentation
	// models instead of leaving 「権藤三峰は武将である」 as one opaque word.
	// Off (the default) keeps byte-for-byte parity with the reference
	// splitter, which does not segment CJK.
	CJKSplit bool
}

func (o Opts) threshold() float64 {
	if o.Threshold == 0 {
		return 0.5
	}
	return o.Threshold
}

func sigmoid(v float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(v))))
}

// Entity is one predicted span, located by BYTE offset into the text passed to
// Predict, so text[Start:End] slices the original string directly. The Python
// reference reports character offsets instead; the two agree on ASCII and differ
// everywhere else, which is why this is stated rather than assumed.
type Entity struct {
	Text  string
	Label string
	Start int // byte offset into the input text
	End   int
	Score float64
}

// charToByte converts a Python string index into a Go byte offset.
func charToByte(s string, char int) int {
	n := 0
	for i := range s { // ranges by rune, i is the byte offset
		if n == char {
			return i
		}
		n++
	}
	return len(s)
}

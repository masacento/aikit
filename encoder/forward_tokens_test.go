package encoder

import (
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestForwardInnerAndForwardTokens_agree cross-checks that forward.go's
// forwardInner and forward_tokens.go's forwardTokens compute the SAME
// per-layer transformer stack on the same input — not just that each
// separately matches its own golden fixture. forwardTokens' own doc comment
// claims this ("intentionally a near-mirror of forward... so any future
// change to the transformer block stays a one-place edit per path"), but
// nothing enforced it: the two could silently diverge (a stray change to one
// copy's layer loop, an argument passed in the wrong order) and every
// existing test would still pass, each against its own separately-pinned
// golden.
//
// forwardTokens returns the raw [L, D] hidden states pre-pooling;
// forwardInner returns those same hidden states pooled per the model's
// declared reduction. So "agree" here means: pooling forwardTokens' output
// the model's own way must reproduce forwardInner's output bit-for-bit — both
// run identical selfAttention/layerNorm/applyMLP calls in the same order, so
// there is no floating-point reason for them to differ at all.
func TestForwardInnerAndForwardTokens_agree(t *testing.T) {
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no model at %s — symlink testdata/encoder-model -> HF snapshot", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	w := m.weights
	D := w.Cfg.HiddenDim

	cases := []struct {
		name    string
		text    string
		isQuery bool
	}{
		{"query", "how do i parse json", true},
		{"doc", "def add(a, b):\n    return a + b", false},
		{"empty", "", true},
		{"long", "a much longer sentence meant to exercise more than a handful of token positions so attention actually has several rows of K and V to read across", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var (
				ids []int32
				err error
			)
			if c.isQuery {
				ids, err = EncodeQuery(m.tok, c.text, m.maxSeqLength)
			} else {
				ids, err = EncodeDoc(m.tok, c.text, m.maxSeqLength)
			}
			if err != nil {
				t.Fatalf("tokenize: %v", err)
			}
			L := len(ids)

			pooled := w.forwardInner(ids)
			tokens := w.forwardTokens(ids)
			if len(tokens) != L*D {
				t.Fatalf("forwardTokens len = %d, want %d (L=%d, D=%d)", len(tokens), L*D, L, D)
			}
			want := poolOne(tokens, L, D, w.Cfg.pooling)
			if len(pooled) != len(want) {
				t.Fatalf("forwardInner len = %d, want %d", len(pooled), len(want))
			}
			for i := range want {
				if pooled[i] != want[i] {
					t.Fatalf("forwardInner and pool(forwardTokens) disagree at [%d]: %v vs %v (L=%d)", i, pooled[i], want[i], L)
				}
			}
		})
	}
}

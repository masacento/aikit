package encoder

import (
	"os"
	"strings"
	"testing"
)

// BenchmarkCrossEncoderScore is item 13's third model, and the one the campaign
// doc predicted would gain most: at D=384 the FFN activation share GROWS
// (activation ÷ FFN-matmul = t_act/(k·D·t_mac)), which the doc sized at ~22%
// GELU plus ~14% softmax for MiniLM-L6.
func BenchmarkCrossEncoderScore(b *testing.B) {
	const dir = "../testdata/crossencoder-model"
	if _, err := os.Stat(dir); err != nil {
		b.Skip("testdata/crossencoder-model/ not present; see scripts/README.md")
	}
	ce, err := LoadCrossEncoder(dir)
	if err != nil {
		b.Fatalf("LoadCrossEncoder: %v", err)
	}
	defer func() { _ = ce.Close() }()

	const query = "how do i parse json in go with generic structs"
	const sent = "encoding/json unmarshals into a struct or a map depending on the target type. "

	// Sentence counts chosen so the PAIR lands near the named length. The
	// tokenizer truncates to maxSeq (512 here), so a doc that is merely "long"
	// silently becomes L=512 whatever the label says — the earlier version of
	// this benchmark was named L200 and ran at 512.
	for _, tc := range []struct {
		name  string
		sents int
	}{
		{"L200", 8},
		{"L512", 40}, // 897 raw tokens, truncated to maxSeq
	} {
		doc := strings.Repeat(sent, tc.sents)
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				s, err := ce.Score(query, doc)
				if err != nil {
					b.Fatal(err)
				}
				sinkScore = s
			}
		})
	}
}

var sinkScore float32

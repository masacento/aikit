package encoder

import (
	"os"
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
	doc := ""
	for range 40 {
		doc += "encoding/json unmarshals into a struct or a map depending on the target type. "
	}
	b.Run("L200", func(b *testing.B) {
		for b.Loop() {
			s, err := ce.Score(query, doc)
			if err != nil {
				b.Fatal(err)
			}
			sinkScore = s
		}
	})
}

var sinkScore float32

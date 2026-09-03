package embed

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

// unmarshalUnigramJSON is the generic decode scanUnigramJSON replaces; the tests
// below use it as the oracle.
func unmarshalUnigramJSON(t *testing.T, data []byte) (*unigramJSON, []string, []float64) {
	t.Helper()
	var raw unigramJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	pieces := make([]string, len(raw.Model.Vocab))
	scores := make([]float64, len(raw.Model.Vocab))
	for i, entry := range raw.Model.Vocab {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 {
			t.Fatalf("vocab[%d]: not a pair", i)
		}
		if err := json.Unmarshal(pair[0], &pieces[i]); err != nil {
			t.Fatalf("vocab[%d] piece: %v", i, err)
		}
		if err := json.Unmarshal(pair[1], &scores[i]); err != nil {
			t.Fatalf("vocab[%d] score: %v", i, err)
		}
	}
	return &raw, pieces, scores
}

func compareUnigramJSON(t *testing.T, what string, got *unigramJSON, gotPieces []string, gotScores []float64, want *unigramJSON, wantPieces []string, wantScores []float64) {
	t.Helper()
	if !reflect.DeepEqual(gotPieces, wantPieces) {
		for i := range wantPieces {
			if i >= len(gotPieces) {
				t.Fatalf("%s: pieces truncated at %d (%d of %d)", what, i, len(gotPieces), len(wantPieces))
			}
			if gotPieces[i] != wantPieces[i] {
				t.Fatalf("%s: piece[%d] = %q, want %q", what, i, gotPieces[i], wantPieces[i])
			}
		}
		t.Fatalf("%s: %d pieces, want %d", what, len(gotPieces), len(wantPieces))
	}
	if len(gotScores) != len(wantScores) {
		t.Fatalf("%s: %d scores, want %d", what, len(gotScores), len(wantScores))
	}
	for i := range wantScores {
		// Bit-exact: both sides go through strconv's correctly-rounded parse, so
		// any difference here is a decode bug, not float noise.
		if math.Float64bits(gotScores[i]) != math.Float64bits(wantScores[i]) {
			t.Fatalf("%s: score[%d] = %v, want %v", what, i, gotScores[i], wantScores[i])
		}
	}
	if !reflect.DeepEqual(got.AddedTokens, want.AddedTokens) {
		t.Errorf("%s: added_tokens = %+v, want %+v", what, got.AddedTokens, want.AddedTokens)
	}
	if !reflect.DeepEqual(got.PostProcessor, want.PostProcessor) {
		t.Errorf("%s: post_processor = %+v, want %+v", what, got.PostProcessor, want.PostProcessor)
	}
	if got.Model.Type != want.Model.Type {
		t.Errorf("%s: model.type = %q, want %q", what, got.Model.Type, want.Model.Type)
	}
	if got.Model.ByteFallback != want.Model.ByteFallback {
		t.Errorf("%s: byte_fallback = %v, want %v", what, got.Model.ByteFallback, want.Model.ByteFallback)
	}
	switch {
	case (got.Model.UnkID == nil) != (want.Model.UnkID == nil):
		t.Errorf("%s: unk_id = %v, want %v", what, got.Model.UnkID, want.Model.UnkID)
	case got.Model.UnkID != nil && *got.Model.UnkID != *want.Model.UnkID:
		t.Errorf("%s: unk_id = %d, want %d", what, *got.Model.UnkID, *want.Model.UnkID)
	}
	// Normalizer / PreTokenizer are raw subtrees handed on to buildNormalizer and
	// parseMetaspaceKind; compare them by what those consumers see.
	for _, f := range []struct {
		name      string
		got, want json.RawMessage
	}{
		{"normalizer", got.Normalizer, want.Normalizer},
		{"pre_tokenizer", got.PreTokenizer, want.PreTokenizer},
	} {
		var a, b any
		gotEmpty := len(f.got) == 0 || json.Unmarshal(f.got, &a) != nil
		wantEmpty := len(f.want) == 0 || json.Unmarshal(f.want, &b) != nil
		if gotEmpty != wantEmpty || !reflect.DeepEqual(a, b) {
			t.Errorf("%s: %s = %s, want %s", what, f.name, f.got, f.want)
		}
	}
}

// TestScanUnigramJSON_realTokenizers is the contract that lets
// parseUnigramTokenizer skip the whole-document unmarshal: on every real
// tokenizer.json in testdata the scanner must decode exactly what encoding/json
// decodes — including bit-exact scores, since the Viterbi search compares them.
func TestScanUnigramJSON_realTokenizers(t *testing.T) {
	entries, err := os.ReadDir("../testdata")
	if err != nil {
		t.Skip(err)
	}
	n := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		p := "../testdata/" + e.Name() + "/tokenizer.json"
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		if !isUnigramTokenizer(jsonProbeString(data, "model", "type"), jsonProbeString(data, "normalizer", "type")) {
			continue
		}
		got, gotPieces, gotScores, ok := scanUnigramJSON(data)
		if !ok {
			t.Errorf("%s: scanner declined a real Unigram tokenizer", p)
			continue
		}
		want, wantPieces, wantScores := unmarshalUnigramJSON(t, data)
		compareUnigramJSON(t, p, got, gotPieces, gotScores, want, wantPieces, wantScores)
		n++
	}
	if n == 0 {
		t.Skip("no Unigram tokenizer.json in testdata")
	}
	t.Logf("scanned %d Unigram tokenizer.json files", n)
}

// TestScanUnigramJSON_shapes covers the encodings a hand-written scanner can get
// wrong but the checkpoints in testdata happen not to contain.
func TestScanUnigramJSON_shapes(t *testing.T) {
	const head = `"normalizer":null,"pre_tokenizer":{"type":"Metaspace","replacement":"\u2581"},` +
		`"added_tokens":[{"id":0,"content":"<unk>","special":true}],`
	cases := []struct {
		name string
		doc  string
	}{
		{"plain", `{` + head + `"model":{"type":"Unigram","unk_id":0,"vocab":[["a",0.0],["b",-1.5]],"byte_fallback":true}}`},
		{"escapes", `{` + head + `"model":{"unk_id":0,"vocab":[["\"",0],["\\",-1],["\u2581x",-2],["a\/b",-3],["\ud83d\ude00",-4]]}}`},
		{"exponent and precision", `{` + head + `"model":{"unk_id":0,"vocab":[["a",-1e-3],["b",-12.345678901234567],["c",1E5],["d",-0]]}}`},
		{"integer scores", `{` + head + `"model":{"unk_id":0,"vocab":[["a",0],["b",-13]]}}`},
		{"empty vocab", `{` + head + `"model":{"unk_id":0,"vocab":[]}}`},
		{"empty piece", `{` + head + `"model":{"unk_id":0,"vocab":[["",0.0]]}}`},
		{"whitespace", "{\n " + head + "\n \"model\" : {\n \"unk_id\" : 0 ,\n \"vocab\" : [\n [ \"a\" , 0.0 ]\n ]\n }\n}"},
		{"vocab before unk_id", `{` + head + `"model":{"vocab":[["a",0.0]],"unk_id":0,"byte_fallback":false}}`},
		{"model first", `{"model":{"unk_id":0,"vocab":[["a",0.0]]},` + head[:len(head)-1] + `}`},
		{"brackets inside pieces", `{` + head + `"model":{"unk_id":0,"vocab":[["[",0.0],["]",-1.0],["{\"}",-2.0]]}}`},
		{"post_processor", `{` + head + `"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>"}},{"Sequence":{"id":"A"}}],` +
			`"special_tokens":{"<s>":{"ids":[1]}}},"model":{"unk_id":0,"vocab":[["a",0.0]]}}`},
		// encoding/json takes the last of a duplicated key, so the scanner must too.
		{"duplicate keys", `{` + head + `"model":{"unk_id":1,"vocab":[["a",0.0]],"unk_id":0,"vocab":[["b",-1.0],["c",-2.0]]}}`},
		{"null model fields", `{` + head + `"model":{"type":null,"unk_id":0,"byte_fallback":null,"vocab":[["a",0.0]]}}`},
		// encoding/json rewrites invalid UTF-8 to U+FFFD rather than failing, so
		// the scanner has to produce the same replacement, not the raw byte.
		{"invalid utf-8 in piece", "{" + head + "\"model\":{\"unk_id\":0,\"vocab\":[[\"\x9d\",0.0],[\"ok\",-1.0]]}}"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, gotPieces, gotScores, ok := scanUnigramJSON([]byte(tc.doc))
			if !ok {
				t.Fatalf("scanner declined %s", tc.doc)
			}
			want, wantPieces, wantScores := unmarshalUnigramJSON(t, []byte(tc.doc))
			compareUnigramJSON(t, tc.name, got, gotPieces, gotScores, want, wantPieces, wantScores)
		})
	}
}

// TestScanUnigramJSON_declines checks the scanner refuses anything it is not sure
// about, so parseUnigramTokenizer falls back to encoding/json and reports the
// original error rather than loading a half-decoded vocab.
func TestScanUnigramJSON_declines(t *testing.T) {
	cases := []struct{ name, doc string }{
		{"empty", ``},
		{"not json", `<html>`},
		{"array root", `[1,2]`},
		{"no model", `{"normalizer":null}`},
		{"model not an object", `{"model":"Unigram"}`},
		{"no vocab", `{"model":{"type":"Unigram","unk_id":0}}`},
		{"vocab not an array", `{"model":{"vocab":{}}}`},
		{"entry not an array", `{"model":{"vocab":["a"]}}`},
		{"piece not a string", `{"model":{"vocab":[[1,0.0]]}}`},
		{"score not a number", `{"model":{"vocab":[["a","x"]]}}`},
		{"missing score", `{"model":{"vocab":[["a"]]}}`},
		{"extra pair element", `{"model":{"vocab":[["a",0.0,"x"]]}}`},
		{"truncated vocab", `{"model":{"vocab":[["a",0.0],["b",`},
		{"unterminated string", `{"model":{"vocab":[["a`},
		{"unk_id not a number", `{"model":{"unk_id":"0","vocab":[["a",0.0]]}}`},
		{"added_tokens wrong shape", `{"added_tokens":{"a":1},"model":{"unk_id":0,"vocab":[["a",0.0]]}}`},
		// Numbers strconv.ParseFloat takes but JSON does not: without jsonNumberOK
		// the scanner would load vocabs encoding/json rejects.
		{"leading zero score", `{"model":{"unk_id":0,"vocab":[["a",00]]}}`},
		{"plus-signed score", `{"model":{"unk_id":0,"vocab":[["a",+1]]}}`},
		{"bare fraction score", `{"model":{"unk_id":0,"vocab":[["a",.5]]}}`},
		{"trailing dot score", `{"model":{"unk_id":0,"vocab":[["a",1.]]}}`},
		{"infinity score", `{"model":{"unk_id":0,"vocab":[["a",Inf]]}}`},
		{"nan score", `{"model":{"unk_id":0,"vocab":[["a",NaN]]}}`},
		{"hex score", `{"model":{"unk_id":0,"vocab":[["a",0x1p2]]}}`},
		{"empty exponent score", `{"model":{"unk_id":0,"vocab":[["a",1e]]}}`},
		{"plus-signed unk_id", `{"model":{"unk_id":+0,"vocab":[["a",0.0]]}}`},
		{"byte_fallback not a bool", `{"model":{"unk_id":0,"byte_fallback":1,"vocab":[["a",0.0]]}}`},
		{"model type not a string", `{"model":{"type":7,"unk_id":0,"vocab":[["a",0.0]]}}`},
		{"missing comma between pairs", `{"model":{"unk_id":0,"vocab":[["a",0.0]["b",-1.0]]}}`},
		{"trailing comma in vocab", `{"model":{"unk_id":0,"vocab":[["a",0.0],]}}`},
		{"missing comma between members", `{"model":{"unk_id":0 "vocab":[["a",0.0]]}}`},
		{"trailing comma in object", `{"model":{"unk_id":0,"vocab":[["a",0.0]],}}`},
		{"raw control byte in piece", "{\"model\":{\"unk_id\":0,\"vocab\":[[\"\x03\",0.0]]}}"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, _, ok := scanUnigramJSON([]byte(tc.doc)); ok {
				t.Fatalf("scanner accepted %s", tc.doc)
			}
		})
	}
}

// FuzzScanUnigramJSON checks the scanner cannot panic or run away: it reads the
// document before anything has validated it.
func FuzzScanUnigramJSON(f *testing.F) {
	f.Add(`{"model":{"type":"Unigram","unk_id":0,"vocab":[["a",0.0],["b",-1.5]]}}`)
	f.Add(`{"model":{"vocab":[["\ud83d\ude00",-1e-3]]}}`)
	f.Add(`{"model":{"vocab":[[`)
	f.Add(`{`)
	f.Fuzz(func(t *testing.T, doc string) {
		_, pieces, scores, ok := scanUnigramJSON([]byte(doc))
		if !ok {
			return
		}
		if len(pieces) != len(scores) {
			t.Fatalf("%d pieces vs %d scores", len(pieces), len(scores))
		}
		// The scanner only promises to be exact about what it decodes — a field it
		// skips (e.g. "decoder") is checked for bracket balance and nothing more.
		// So the oracle is the vocab subtree: if the scanner accepted it, it must
		// be valid JSON and decode to exactly these pieces and scores.
		top, ok := jsonObjectFields([]byte(doc), "model")
		if !ok || top[0] == nil {
			t.Fatalf("accepted a document with no model object: %q", doc)
		}
		model, ok := jsonObjectFields(top[0], "vocab")
		if !ok || model[0] == nil {
			t.Fatalf("accepted a model with no vocab: %q", doc)
		}
		var want [][]json.RawMessage
		if err := json.Unmarshal(model[0], &want); err != nil {
			t.Fatalf("accepted a vocab encoding/json rejects (%v): %q", err, doc)
		}
		if len(want) != len(pieces) {
			t.Fatalf("%d pieces, want %d: %q", len(pieces), len(want), doc)
		}
		for i, pair := range want {
			if len(pair) != 2 {
				t.Fatalf("accepted vocab[%d] with %d elements: %q", i, len(pair), doc)
			}
			var piece string
			var score float64
			if err := json.Unmarshal(pair[0], &piece); err != nil {
				t.Fatalf("accepted vocab[%d] piece (%v): %q", i, err, doc)
			}
			if err := json.Unmarshal(pair[1], &score); err != nil {
				t.Fatalf("accepted vocab[%d] score (%v): %q", i, err, doc)
			}
			if pieces[i] != piece {
				t.Fatalf("piece[%d] = %q, want %q: %q", i, pieces[i], piece, doc)
			}
			if math.Float64bits(scores[i]) != math.Float64bits(score) {
				t.Fatalf("score[%d] = %v, want %v: %q", i, scores[i], score, doc)
			}
		}
	})
}

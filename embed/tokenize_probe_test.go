package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// TestJSONProbeString_matchesUnmarshal is the contract that lets LoadTokenizer
// dispatch on the probe instead of a whole-document unmarshal: for the fields it
// probes, the byte scanner must answer exactly what encoding/json would.
func TestJSONProbeString_matchesUnmarshal(t *testing.T) {
	cases := []struct {
		name string
		doc  string
		path []string
		want string
	}{
		{"model type", `{"model":{"type":"Unigram","vocab":[["a",-1.0],["b",-2.5]]}}`, []string{"model", "type"}, "Unigram"},
		// The field wanted sits AFTER a large sibling — the case the probe exists
		// for (HF writes "model" last, past the whole vocab).
		{"after big sibling", `{"added_tokens":[{"id":0,"content":"<s>","special":true}],"model":{"type":"BPE"}}`, []string{"model", "type"}, "BPE"},
		{"nested braces in string", `{"a":{"t":"}{[\"x"},"model":{"type":"WordPiece"}}`, []string{"model", "type"}, "WordPiece"},
		{"escapes decoded", `{"model":{"type":"abc\n"}}`, []string{"model", "type"}, "abc\n"},
		{"null section", `{"normalizer":null,"model":{"type":"Unigram"}}`, []string{"normalizer", "type"}, ""},
		{"absent section", `{"model":{"type":"Unigram"}}`, []string{"normalizer", "type"}, ""},
		{"absent leaf", `{"model":{"vocab":[]}}`, []string{"model", "type"}, ""},
		{"not a string", `{"model":{"type":7}}`, []string{"model", "type"}, ""},
		{"section not an object", `{"model":"Unigram"}`, []string{"model", "type"}, ""},
		{"scalars skipped", `{"a":1e-3,"b":true,"c":null,"model":{"type":"BPE"}}`, []string{"model", "type"}, "BPE"},
		{"whitespace", "{\n\t\"model\" : {\n\t\t\"type\" : \"BPE\"\n\t}\n}", []string{"model", "type"}, "BPE"},
		{"empty", ``, []string{"model", "type"}, ""},
		{"truncated", `{"model":{"type":"Uni`, []string{"model", "type"}, ""},
		{"not json", `<html>`, []string{"model", "type"}, ""},
		{"array root", `["model"]`, []string{"model", "type"}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := jsonProbeString([]byte(tc.doc), tc.path...); got != tc.want {
				t.Fatalf("jsonProbeString(%s) = %q, want %q", tc.doc, got, tc.want)
			}
		})
	}
}

// TestJSONProbe_realTokenizers checks the probe against the checkpoints in
// testdata, where the answer is whatever the full unmarshal used to produce.
func TestJSONProbe_realTokenizers(t *testing.T) {
	paths, err := os.ReadDir("../testdata")
	if err != nil {
		t.Skip(err)
	}
	n := 0
	for _, e := range paths {
		if !e.IsDir() {
			continue
		}
		p := "../testdata/" + e.Name() + "/tokenizer.json"
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var want struct {
			Model        struct{ Type string } `json:"model"`
			Normalizer   struct{ Type string } `json:"normalizer"`
			PreTokenizer struct{ Type string } `json:"pre_tokenizer"`
		}
		if err := json.Unmarshal(data, &want); err != nil {
			t.Errorf("%s: %v", p, err)
			continue
		}
		for _, f := range []struct{ section, want string }{
			{"model", want.Model.Type},
			{"normalizer", want.Normalizer.Type},
			{"pre_tokenizer", want.PreTokenizer.Type},
		} {
			if got := jsonProbeString(data, f.section, "type"); got != f.want {
				t.Errorf("%s: %s.type = %q, want %q", p, f.section, got, f.want)
			}
		}
		n++
	}
	if n == 0 {
		t.Skip("no tokenizer.json in testdata")
	}
	t.Logf("probed %d tokenizer.json files", n)
}

// FuzzJSONProbe checks the scanner cannot panic or run away on arbitrary input —
// it runs before anything has validated the document.
func FuzzJSONProbe(f *testing.F) {
	f.Add(`{"model":{"type":"Unigram"},"normalizer":null}`)
	f.Add(`{"a":[{"b":"\""}],"model":{"type":"BPE"}}`)
	f.Add(`{`)
	f.Fuzz(func(t *testing.T, doc string) {
		jsonProbeString([]byte(doc), "model", "type")
		jsonProbeRaw([]byte(doc), "pre_tokenizer")
	})
}

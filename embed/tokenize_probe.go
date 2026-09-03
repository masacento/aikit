package embed

import "encoding/json"

// tokenize_probe.go — a minimal JSON field probe for the dispatch decisions
// LoadTokenizer makes before it knows which parser to run.
//
// The problem it solves is that a tokenizer.json is big and encoding/json is
// whole-document: `json.Unmarshal(data, &struct{ Model struct{ Type string } })`
// reads model.type by LEXING ALL 6.7 MB of ruri-v3-30m's file — every one of the
// 102400 [piece, score] vocab entries — and then throwing the result away, before
// the real parser reads the same bytes again. That probe was ~40% of the encoder's
// cold start (PERF_REPORT §5); the model type it wants sits in the first ~100
// bytes of the model object.
//
// jsonProbeString walks only the keys on the path it was given and skips every
// other value with a byte scan — no allocation, no unescaping, no number
// conversion — so it stops at the first "type" it needs instead of running to EOF.
// It is deliberately NOT a JSON parser: it is a locator whose answers are always
// re-validated by the real parse that follows. On anything malformed it returns
// "" / nil and the caller falls through to a full json.Unmarshal, which reports
// the syntax error exactly as before.

// jsonProbeString returns the string value at path (a chain of object keys) in
// data, or "" if the path is absent, is not a string, or the document does not
// parse far enough along that path to tell.
func jsonProbeString(data []byte, path ...string) string {
	raw := jsonProbeRaw(data, path...)
	if len(raw) < 2 || raw[0] != '"' {
		return ""
	}
	body := raw[1 : len(raw)-1]
	for _, c := range body {
		if c == '\\' { // rare in these fields; let the real decoder handle escapes
			var s string
			if json.Unmarshal(raw, &s) != nil {
				return ""
			}
			return s
		}
	}
	return string(body)
}

// jsonProbeRaw returns the raw (still-encoded) JSON value at path, or nil. The
// result aliases data.
func jsonProbeRaw(data []byte, path ...string) []byte {
	for _, key := range path {
		v, ok := jsonObjectField(data, key)
		if !ok {
			return nil
		}
		data = v
	}
	return data
}

// jsonObjectField returns the raw value of key in the JSON object at the start of
// data. Keys are compared raw, which is exact for the plain-ASCII keys this file
// probes (a key spelled with an escape simply won't match, and the caller falls
// back to the full parse).
func jsonObjectField(data []byte, key string) ([]byte, bool) {
	i := jsonSkipSpace(data, 0)
	if i >= len(data) || data[i] != '{' {
		return nil, false
	}
	i++
	for {
		i = jsonSkipSpace(data, i)
		if i >= len(data) {
			return nil, false
		}
		switch data[i] {
		case '}':
			return nil, false
		case ',':
			i++
			continue
		case '"':
		default:
			return nil, false
		}
		k, next, ok := jsonScanString(data, i)
		if !ok {
			return nil, false
		}
		i = jsonSkipSpace(data, next)
		if i >= len(data) || data[i] != ':' {
			return nil, false
		}
		i = jsonSkipSpace(data, i+1)
		end, ok := jsonSkipValue(data, i)
		if !ok {
			return nil, false
		}
		if string(k) == key {
			return data[i:end], true
		}
		i = end
	}
}

func jsonSkipSpace(data []byte, i int) int {
	for i < len(data) {
		switch data[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// jsonScanString scans the string literal starting at data[i] (which must be the
// opening quote) and returns its raw body (quotes stripped, escapes NOT decoded)
// and the index just past the closing quote.
func jsonScanString(data []byte, i int) (body []byte, next int, ok bool) {
	start := i + 1
	for j := start; j < len(data); j++ {
		switch data[j] {
		case '\\':
			j++ // the escaped byte cannot end the string; \uXXXX's digits are not '"'
		case '"':
			return data[start:j], j + 1, true
		}
	}
	return nil, 0, false
}

// jsonSkipValue returns the index just past the JSON value starting at data[i].
// Objects and arrays are skipped by bracket depth, with strings consumed whole so
// a bracket inside one cannot unbalance the count; scalars run to the first
// structural byte or space.
func jsonSkipValue(data []byte, i int) (int, bool) {
	if i >= len(data) {
		return 0, false
	}
	switch data[i] {
	case '"':
		_, next, ok := jsonScanString(data, i)
		return next, ok
	case '{', '[':
		depth := 0
		for j := i; j < len(data); j++ {
			switch data[j] {
			case '{', '[':
				depth++
			case '}', ']':
				if depth--; depth == 0 {
					return j + 1, true
				}
			case '"':
				_, next, ok := jsonScanString(data, j)
				if !ok {
					return 0, false
				}
				j = next - 1
			}
		}
		return 0, false
	default: // number, true, false, null
		for j := i; j < len(data); j++ {
			switch data[j] {
			case ',', '}', ']', ' ', '\t', '\n', '\r':
				return j, j > i
			}
		}
		return len(data), len(data) > i
	}
}

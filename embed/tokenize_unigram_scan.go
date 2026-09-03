package embed

import (
	"encoding/json"
	"strconv"
	"unicode/utf8"
)

// tokenize_unigram_scan.go — a direct scanner for the one part of a Unigram
// tokenizer.json that is actually big: model.vocab.
//
// `json.Unmarshal(data, &unigramJSON{})` costs ~130 ms on ruri-v3-30m (6.7 MB,
// 102400 [piece, score] pairs) and that was the entire cold start of
// encoder.LoadModernBERT — the weights are an mmap and take 0.1 ms. The cost is
// not the bytes; it is what the generic decoder does per entry:
//
//   - checkValid lexes all 6.7 MB once before decoding starts (~25% of the total),
//   - each vocab entry is decoded into a json.RawMessage, which copies it,
//   - each of those is then Unmarshal'd again into []json.RawMessage (two more
//     copies + their own checkValid), and the piece and score are Unmarshal'd a
//     third time.
//
// scanUnigramJSON walks the document once with the byte scanner already used by
// jsonProbeString, decodes pieces and scores straight out of the input, and hands
// the small subtrees (added_tokens, post_processor) to encoding/json unchanged.
// Every piece lives in one shared backing string instead of 102400 separate
// allocations.
//
// What it decodes it decodes exactly: pieces and scores are byte-for-byte what
// encoding/json produces, down to JSON's stricter number grammar, its rejection
// of raw control bytes in strings, and its U+FFFD substitution for invalid UTF-8
// (FuzzScanUnigramJSON pins all three). Anything it does not recognise makes it
// return ok=false, and the caller falls back to the full json.Unmarshal, which
// reports the error exactly as before.
//
// What it does NOT do is validate the fields it skips. Like jsonProbeString it is
// a locator there: "decoder" and friends are scanned for bracket balance and
// nothing else, so a syntax error buried inside one is no longer reported.

// scanUnigramJSON decodes a Unigram tokenizer.json without a whole-document
// unmarshal. ok is false if anything on the path it walks is not the shape it
// expects; the caller must then fall back to the generic decoder.
func scanUnigramJSON(data []byte) (raw *unigramJSON, pieces []string, scores []float64, ok bool) {
	top, ok := jsonObjectFields(data, "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "model")
	if !ok || top[4] == nil {
		return nil, nil, nil, false
	}
	model, ok := jsonObjectFields(top[4], "type", "unk_id", "byte_fallback", "vocab")
	if !ok || model[3] == nil {
		return nil, nil, nil, false
	}
	pieces, scores, ok = scanUnigramVocab(model[3])
	if !ok {
		return nil, nil, nil, false
	}

	raw = &unigramJSON{
		Normalizer:   top[1],
		PreTokenizer: top[2],
	}
	switch b := model[0]; {
	case b == nil, string(b) == "null":
	case b[0] == '"':
		raw.Model.Type = jsonProbeString(b)
	default: // not a string — the generic decoder would reject the document
		return nil, nil, nil, false
	}
	switch b := model[2]; {
	case b == nil, string(b) == "false", string(b) == "null":
	case string(b) == "true":
		raw.Model.ByteFallback = true
	default: // not a bool — the generic decoder would reject the document
		return nil, nil, nil, false
	}
	if model[1] != nil && string(model[1]) != "null" {
		if !jsonNumberOK(model[1]) {
			return nil, nil, nil, false
		}
		v, err := strconv.ParseInt(string(model[1]), 10, 32)
		if err != nil {
			return nil, nil, nil, false
		}
		unk := int32(v)
		raw.Model.UnkID = &unk
	}
	// added_tokens and post_processor are a few hundred bytes; the generic decoder
	// is fine there and keeps their (nested, optional) shapes in one place.
	if b := top[0]; b != nil {
		if json.Unmarshal(b, &raw.AddedTokens) != nil {
			return nil, nil, nil, false
		}
	}
	if b := top[3]; b != nil && string(b) != "null" {
		if json.Unmarshal(b, &raw.PostProcessor) != nil {
			return nil, nil, nil, false
		}
	}
	return raw, pieces, scores, true
}

// scanUnigramVocab decodes a `[[piece, score], ...]` array. The returned pieces
// are substrings of one shared backing string.
func scanUnigramVocab(data []byte) (pieces []string, scores []float64, ok bool) {
	i := jsonSkipSpace(data, 0)
	if i >= len(data) || data[i] != '[' {
		return nil, nil, false
	}
	i = jsonSkipSpace(data, i+1)
	// A pretty-printed vocab runs ~65 bytes per entry, of which ~7 are the piece;
	// these estimates size the two buffers in one shot for the common case.
	buf := make([]byte, 0, len(data)/8)
	type span struct{ off, end int }
	spans := make([]span, 0, len(data)/64)
	scores = make([]float64, 0, len(data)/64)

	for i < len(data) && data[i] != ']' {
		if data[i] != '[' {
			return nil, nil, false
		}
		// piece
		i = jsonSkipSpace(data, i+1)
		if i >= len(data) || data[i] != '"' {
			return nil, nil, false
		}
		body, next, sok := jsonScanString(data, i)
		if !sok {
			return nil, nil, false
		}
		off := len(buf)
		if plainJSONString(body) {
			buf = append(buf, body...)
		} else {
			// None of these cases occur in a real SentencePiece vocab — pieces are
			// unescaped, valid UTF-8 — so hand the literal to the stdlib rather
			// than reimplement \uXXXX decoding, the U+FFFD substitution for invalid
			// UTF-8, and the rejection of raw control bytes, all of which the
			// result has to match exactly.
			var s string
			if json.Unmarshal(data[i:next], &s) != nil {
				return nil, nil, false
			}
			buf = append(buf, s...)
		}
		spans = append(spans, span{off, len(buf)})
		// score
		i = jsonSkipSpace(data, next)
		if i >= len(data) || data[i] != ',' {
			return nil, nil, false
		}
		i = jsonSkipSpace(data, i+1)
		end, vok := jsonSkipValue(data, i)
		if !vok {
			return nil, nil, false
		}
		if !jsonNumberOK(data[i:end]) {
			return nil, nil, false
		}
		score, err := strconv.ParseFloat(string(data[i:end]), 64)
		if err != nil {
			return nil, nil, false
		}
		scores = append(scores, score)
		// close the pair, then require a separator before the next one (a missing
		// comma, or a trailing one before ']', is not JSON)
		i = jsonSkipSpace(data, end)
		if i >= len(data) || data[i] != ']' {
			return nil, nil, false
		}
		i = jsonSkipSpace(data, i+1)
		if i < len(data) && data[i] == ',' {
			i = jsonSkipSpace(data, i+1)
			if i < len(data) && data[i] == ']' {
				return nil, nil, false
			}
			continue
		}
		break
	}
	if i >= len(data) || data[i] != ']' {
		return nil, nil, false
	}

	all := string(buf)
	pieces = make([]string, len(spans))
	for n, sp := range spans {
		pieces[n] = all[sp.off:sp.end]
	}
	return pieces, scores, true
}

// plainJSONString reports whether body — the raw bytes between a JSON string's
// quotes — is already its own decoding, so it can be copied out of the input as
// is. It is not when it holds an escape, a raw control byte (which JSON forbids
// outright), or invalid UTF-8 (which encoding/json rewrites to U+FFFD).
func plainJSONString(body []byte) bool {
	for i := 0; i < len(body); {
		if c := body[i]; c < utf8.RuneSelf {
			if c == '\\' || c < 0x20 {
				return false
			}
			i++
			continue
		}
		r, size := utf8.DecodeRune(body[i:])
		if r == utf8.RuneError && size == 1 {
			return false
		}
		i += size
	}
	return true
}

// jsonNumberOK reports whether b is a JSON number:
//
//	-? (0 | [1-9][0-9]*) ('.' [0-9]+)? ([eE] [+-]? [0-9]+)?
//
// strconv.ParseFloat is laxer than JSON — it takes "00", "+1", ".5", "1.",
// "Inf", "NaN" and hex floats — so a score has to clear this first, or the
// scanner would accept vocabs encoding/json rejects.
func jsonNumberOK(b []byte) bool {
	i := 0
	if i < len(b) && b[i] == '-' {
		i++
	}
	// int: a single 0, or a nonzero digit followed by digits
	switch {
	case i < len(b) && b[i] == '0':
		i++
	case i < len(b) && b[i] >= '1' && b[i] <= '9':
		for i < len(b) && b[i] >= '0' && b[i] <= '9' {
			i++
		}
	default:
		return false
	}
	// frac
	if i < len(b) && b[i] == '.' {
		i++
		start := i
		for i < len(b) && b[i] >= '0' && b[i] <= '9' {
			i++
		}
		if i == start {
			return false
		}
	}
	// exp
	if i < len(b) && (b[i] == 'e' || b[i] == 'E') {
		i++
		if i < len(b) && (b[i] == '+' || b[i] == '-') {
			i++
		}
		start := i
		for i < len(b) && b[i] >= '0' && b[i] <= '9' {
			i++
		}
		if i == start {
			return false
		}
	}
	return i == len(b)
}

// jsonObjectFields returns the raw (still-encoded) values of keys in the JSON
// object at the start of data, in a single pass; out[i] is nil when keys[i] is
// absent. ok is false if data is not an object whose skeleton scans cleanly. The
// results alias data.
func jsonObjectFields(data []byte, keys ...string) (out [][]byte, ok bool) {
	out = make([][]byte, len(keys))
	i := jsonSkipSpace(data, 0)
	if i >= len(data) || data[i] != '{' {
		return nil, false
	}
	i = jsonSkipSpace(data, i+1)
	for i < len(data) && data[i] != '}' {
		if data[i] != '"' {
			return nil, false
		}
		k, next, sok := jsonScanString(data, i)
		if !sok {
			return nil, false
		}
		i = jsonSkipSpace(data, next)
		if i >= len(data) || data[i] != ':' {
			return nil, false
		}
		i = jsonSkipSpace(data, i+1)
		end, vok := jsonSkipValue(data, i)
		if !vok {
			return nil, false
		}
		for n, key := range keys {
			// Last occurrence wins, as in encoding/json: a duplicated key must not
			// decode differently here than it would in the fallback path.
			if string(k) == key {
				out[n] = data[i:end]
			}
		}
		// require a separator before the next member (a missing comma, or a
		// trailing one before '}', is not JSON)
		i = jsonSkipSpace(data, end)
		if i < len(data) && data[i] == ',' {
			i = jsonSkipSpace(data, i+1)
			if i < len(data) && data[i] == '}' {
				return nil, false
			}
			continue
		}
		break
	}
	if i >= len(data) || data[i] != '}' {
		return nil, false
	}
	return out, true
}

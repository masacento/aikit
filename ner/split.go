package ner

import (
	"unicode"
	"unicode/utf8"
)

// Word is one unit of GLiNER's word-level input: the token text plus its byte range
// in the original string, which is what lets a predicted span be reported as an
// offset into the caller's text rather than as word indices.
type Word struct {
	Text  string
	Start int // byte offset of the first byte
	End   int // byte offset one past the last
}

// SplitWords reproduces GLiNER's WhitespaceTokenSplitter, whose pattern is
//
//	\w+(?:[-_]\w+)*|\S
//
// It is written as a scanner rather than a Go regexp on purpose. Go's regexp
// `\w` is ASCII-only (`[0-9A-Za-z_]`) and its `\s` is `[\t\n\f\r ]`, while Python's
// are Unicode: `\w` matches anything `str.isalnum()` accepts plus underscore, and
// `\s` matches anything `str.isspace()` accepts. Transliterating the pattern would
// therefore split every non-ASCII input differently from the reference — silently,
// and worst on exactly the multilingual text this model exists for.
//
// Two consequences of the pattern that matter downstream and are not bugs here:
//
//   - It does NOT segment CJK. `アップルは2007年にiPhoneを発表した。` is a run of word
//     characters, so it comes back as very few words and the model can only predict
//     very coarse spans. That is the checkpoint's real behaviour; reproducing it is
//     the point, and "improving" it would put this port out of parity.
//   - Punctuation becomes one word per rune (`\S`), so "Hawaii." is two words.
func SplitWords(text string) []Word {
	var out []Word
	b := []byte(text)
	i := 0
	for i < len(b) {
		r, sz := decodeRune(b[i:])
		switch {
		case isWordRune(r):
			start := i
			i += sz
			i = scanWordRunes(b, i)
			// (?:[-_]\w+)* — a hyphen or underscore only extends the word when at
			// least one word rune follows. "a-b-c" is one word; "a-" is "a" then "-",
			// because the group needs \w+ and backtracks when it cannot have it.
			for i < len(b) && (b[i] == '-' || b[i] == '_') {
				next := scanWordRunes(b, i+1)
				if next == i+1 {
					break
				}
				i = next
			}
			out = append(out, Word{Text: string(b[start:i]), Start: start, End: i})
		case isPySpace(r):
			i += sz
		default:
			// \S — a single non-space, non-word rune.
			out = append(out, Word{Text: string(b[i : i+sz]), Start: i, End: i + sz})
			i += sz
		}
	}
	return out
}

// scanWordRunes advances over a maximal run of word runes starting at i.
func scanWordRunes(b []byte, i int) int {
	for i < len(b) {
		r, sz := decodeRune(b[i:])
		if !isWordRune(r) {
			return i
		}
		i += sz
	}
	return i
}

// isWordRune is Python's `\w` for str patterns: "alphanumeric characters (as defined
// by str.isalnum()) as well as the underscore". str.isalnum() is isalpha (Unicode
// L*) or isdecimal/isdigit/isnumeric (N*). Combining marks are NOT included — they
// are not alphanumeric in Python — so this is deliberately not unicode.IsMark.
func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}

// isPySpace is Python's `\s` for str patterns, i.e. str.isspace(). Go's
// unicode.IsSpace is very close but omits the C1 range U+001C..U+001F, which Python
// treats as whitespace. Those four are file/group/record/unit separators; they turn
// up in real scraped text, and disagreeing about them shifts every word offset after
// the first occurrence.
func isPySpace(r rune) bool {
	if r >= 0x1C && r <= 0x1F {
		return true
	}
	return unicode.IsSpace(r)
}

// decodeRune is utf8.DecodeRune with a guaranteed forward step, so malformed input
// cannot stall the scanner.
func decodeRune(b []byte) (rune, int) {
	r, sz := utf8.DecodeRune(b)
	if sz == 0 {
		return r, 1
	}
	return r, sz
}

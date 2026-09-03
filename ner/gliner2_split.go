package ner

import (
	cjkseg "github.com/townsendmerino/aikit/cjk"
	"strings"
	"unicode"
)

// gliner2_split.go — gliner2's WhitespaceTokenSplitter (fastino-ai/GLiNER2,
// gliner2/processing/word_splitter.py), the word splitter the boundary model runs
// before subword tokenization. The pattern is, in priority order:
//
//	(?:https?://[^\s]+|www\.[^\s]+)
//	|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}   (re.IGNORECASE)
//	|@[a-z0-9_]+
//	|\w+(?:[-_]\w+)*
//	|\S
//
// It is a scanner rather than a Go regexp for the same reason SplitWords is: Go's
// `\w`/`\s` are ASCII-only while Python's are Unicode, and the alternation has to
// be tried in exactly this order to reproduce the reference. Offsets are BYTE
// offsets into the original text (the Word convention); the reference reports
// code-point offsets, which the parity tests convert. The yielded token text is
// lowercased (re.IGNORECASE does not change what is matched, only the classes);
// lowercasing the source first would corrupt offsets via Unicode case folding
// (İ.lower() → "i̇") — the reference's own docstring warns about this.
//
// Two behaviours that look like bugs but are the checkpoint's reality:
//
//   - Like SplitWords, this does NOT segment CJK — 「権藤三峰は武将である」 is one
//     word. gliner2 ships a CharLevelSplitter for that, but the default pipeline
//     does not use it, and parity wins.
//   - An email-ish run like "John@Example.com" matches the EMAIL branch (not
//     \w+), because the branch is case-insensitive and tried before \w+.

// SplitWords2 is the reference splitter, byte-for-byte: like SplitWords, this
// does NOT segment CJK — 「権藤三峰は武将である」 is one word, and parity wins
// over niceness. SplitWords2CJK is the CJK-segmenting variant used by the live
// pipeline; the parity tests pin this one.
func SplitWords2(text string) []Word {
	return splitWords2(text, false)
}

// SplitWords2CJK is SplitWords2 with CJK word runs segmented by the litsea
// word-segmentation models instead of left as one opaque \w+ word. Latin and
// other scripts follow the reference exactly.
func SplitWords2CJK(text string) []Word {
	return splitWords2(text, true)
}

func splitWords2(text string, cjk bool) []Word {
	var out []Word
	b := []byte(text)
	i := 0
	for i < len(b) {
		// finditer skips unmatched characters (whitespace here — every non-space
		// rune matches one of the branches, \S at worst).
		r, sz := decodeRune(b[i:])
		if isPySpace(r) {
			i += sz
			continue
		}
		start := i
		emit := func(end int) {
			out = append(out, Word{Text: pyLower(string(b[start:end])), Start: start, End: end})
		}

		// Branch 1: URL — https?:// or www. then run of non-space.
		if n, ok := matchURL(b, i); ok {
			i = n
			emit(i)
			continue
		}

		// Branch 2: email — local@domain.tld, case-insensitive, with the regex
		// engine's greedy-then-backtrack domain split (see matchEmail).
		if n, ok := matchEmail(b, i); ok {
			i = n
			emit(i)
			continue
		}

		// Branch 3: @handle.
		if b[i] == '@' {
			n := scanClass(b, i+1, isHandleRune)
			if n > i+1 {
				i = n
				emit(i)
				continue
			}
		}

		// Branch 4: \w+(?:[-_]\w+)* — identical word-run logic to SplitWords.
		if isWordRune(r) {
			i += sz
			i = scanWordRunes(b, i)
			for i < len(b) && (b[i] == '-' || b[i] == '_') {
				next := scanWordRunes(b, i+1)
				if next == i+1 {
					break
				}
				i = next
			}
			// A CJK run is one opaque \w+ match to the regex, so when the
			// caller asked for it, segment the run with the litsea
			// word-segmentation model instead. Each piece is emitted with
			// its own byte offset; on any model failure the run falls back
			// to a single word.
			if cjk && cjkseg.HasCJK(string(b[start:i])) {
				pos := start
				for _, piece := range segmentCJKRun(string(b[start:i])) {
					pieceEnd := pos + len(piece)
					out = append(out, Word{Text: pyLower(string(b[pos:pieceEnd])), Start: pos, End: pieceEnd})
					pos = pieceEnd
				}
				continue
			}
			emit(i)
			continue
		}

		// Branch 5: \S — a single rune.
		i += sz
		emit(i)
	}
	return out
}

// matchURL matches the URL alternative at i: a case-insensitive scheme prefix
// (https://, http://, www.) followed by a maximal run of non-whitespace bytes.
func matchURL(b []byte, i int) (end int, ok bool) {
	for _, scheme := range []string{"https://", "http://", "www."} {
		if i+len(scheme) <= len(b) && asciiEqFold(b[i:i+len(scheme)], scheme) {
			j := i + len(scheme)
			for j < len(b) {
				r, sz := decodeRune(b[j:])
				if isPySpace(r) {
					break
				}
				j += sz
			}
			return j, true
		}
	}
	return i, false
}

// matchEmail matches the email alternative at i:
//
//	[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}     (re.IGNORECASE)
//
// The domain part backtracks: the regex engine takes the maximal [a-z0-9.-] run
// after '@', then walks a cut position right-to-left until the remainder is a
// literal '.' followed by at least two letters — the match ends where that
// letter run ends, not necessarily at the run's end ("a@b.co.uk" is one word;
// "a@b.c" fails the branch entirely and falls through to \w+ | @handle).
func matchEmail(b []byte, i int) (end int, ok bool) {
	local := scanClass(b, i, isEmailLocalRune)
	if local == i || local >= len(b) || b[local] != '@' {
		return i, false
	}
	domStart := local + 1
	domEnd := scanClass(b, domStart, isEmailDomainRune)
	if domEnd == domStart {
		return i, false
	}
	// Right-to-left over the domain run for the rightmost legal dot-TLD split.
	for c := domEnd - 2; c > domStart; c-- {
		if b[c] != '.' {
			continue
		}
		tldEnd := scanClass(b, c+1, isASCIILetter)
		if tldEnd-(c+1) >= 2 {
			return tldEnd, true
		}
	}
	return i, false
}

// scanClass advances over a maximal run of bytes satisfying pred, starting at i.
func scanClass(b []byte, i int, pred func(byte) bool) int {
	for i < len(b) && pred(b[i]) {
		i++
	}
	return i
}

func isASCIILetter(c byte) bool { return c|0x20 >= 'a' && c|0x20 <= 'z' }

// isEmailLocalRune is [a-z0-9._%+-] under re.IGNORECASE.
func isEmailLocalRune(c byte) bool {
	switch {
	case isASCIILetter(c), c >= '0' && c <= '9':
		return true
	}
	switch c {
	case '.', '_', '%', '+', '-':
		return true
	}
	return false
}

// isEmailDomainRune is [a-z0-9.-] under re.IGNORECASE.
func isEmailDomainRune(c byte) bool {
	switch {
	case isASCIILetter(c), c >= '0' && c <= '9':
		return true
	}
	return c == '.' || c == '-'
}

func isHandleRune(c byte) bool {
	switch {
	case isASCIILetter(c), c >= '0' && c <= '9':
		return true
	}
	return c == '_'
}

// asciiEqFold compares two same-length ASCII byte slices case-insensitively.
func asciiEqFold(a []byte, lit string) bool {
	if len(a) != len(lit) {
		return false
	}
	for i := range a {
		ca, cl := a[i], lit[i]
		if ca >= 'A' && ca <= 'Z' {
			ca |= 0x20
		}
		if cl >= 'A' && cl <= 'Z' {
			cl |= 0x20
		}
		if ca != cl {
			return false
		}
	}
	return true
}

// pyLower is Python's str.lower() for the one input where Go's strings.ToLower
// disagrees: U+0130 (LATIN CAPITAL LETTER I WITH DOT ABOVE) full-case-folds to
// "i" + U+0307 in Python, while Go's per-rune ToLower maps it to plain "i". The
// parity fixture covers it (İstanbul), so the special case is pinned, not
// speculative. Everything else agrees.
func pyLower(s string) string {
	if !strings.ContainsRune(s, 0x0130) {
		return strings.ToLower(s)
	}
	var b strings.Builder
	b.Grow(len(s) + 2)
	for _, r := range s {
		if r == 0x0130 {
			b.WriteString("i\u0307")
			continue
		}
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

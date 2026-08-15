package regex

import (
	"regexp"
	"regexp/syntax"
	"unicode/utf8"
)

// Prescreen precomputation: for every ^-anchored language-rule pattern, derive
// either a literal byte prefix or a first-byte set that a candidate line can be
// checked against before it ever reaches the regexp engine (anyMatch, in
// chunker.go, is the consumer). Split out of chunker.go — one file per concern
// in what was an 850+ line type; see chunker.go's doc comment.

// register installs a language's rules, precomputing the literal prefix of every
// pattern (lens doc §3.2).
//
// WHY THIS IS SOUND. Every pattern here is ^-anchored — asserted below, because
// the argument depends on it. Anchored means a match must begin at byte 0, so
// "the match begins with p" and "the line begins with p" are the same statement,
// and a line lacking the prefix provably cannot match. The prescreen therefore
// changes which lines reach the regexp engine, never which lines match.
//
// WHY IT IS WORTH IT. regexp.Match is a sync.Pool round-trip plus inputs.init,
// bitState.reset and onepass/backtrack dispatch before it looks at a single
// byte. For `^func\b` — a pattern whose entire meaning is "does this line start
// with func" — that is essentially all overhead: 219.3 ns/line for Go's four
// definition patterns against 22.8 ns for the byte-prefix equivalent.
func register(lang string, r LanguageRules) {
	pre := func(res []*regexp.Regexp) [][]byte {
		out := make([][]byte, len(res))
		for i, re := range res {
			out[i] = anchoredLiteralPrefix(re.String())
		}
		return out
	}
	r.defsPre, r.skipPre, r.attachPre = pre(r.defs), pre(r.skip), pre(r.attach)
	// First-byte sets only where the literal prefix gave nothing (so the stronger
	// multi-byte screen always wins when it exists).
	fb := func(res []*regexp.Regexp, pre [][]byte) []*[256]bool {
		out := make([]*[256]bool, len(res))
		for i, re := range res {
			if len(pre[i]) == 0 {
				out[i] = anchoredFirstByteSet(re.String())
			}
		}
		return out
	}
	r.defsFB, r.skipFB, r.attachFB = fb(r.defs, r.defsPre), fb(r.skip, r.skipPre), fb(r.attach, r.attachPre)
	languageRules[lang] = r
}

// anchoredLiteralPrefix returns the literal bytes that must begin any match of
// pattern, or nil if there are none to be had.
//
// NOT regexp.LiteralPrefix, which was the obvious thing to reach for and does
// not work here: it reads the COMPILED program's extracted prefix, and a `\b` or
// `\s` immediately after the literal blocks that extraction. It returns "" for
// `^func\b` and for `^class\s+\w+` — i.e. for almost every rule in this package,
// including all four of Go's definition patterns. Only the pure-literal comment
// rules (`^//`, `^/\*`) got a prefix from it.
//
// The syntax tree has what the compiled program lost. syntax.Parse(`^func\b`)
// simplifies to a concatenation of \A, the literal "func", and a word boundary,
// so walking the leading literals of that concatenation gives "func" directly.
//
// It bails to nil rather than guessing whenever the argument would not hold:
//
//   - no leading \A (an unanchored pattern would need a substring prescreen, and
//     more importantly would mean the rule style changed — hence the panic);
//   - a case-folded literal, where a byte compare is not the same test;
//   - a non-ASCII rune, so the prefix stays a plain byte comparison;
//   - anything that is not a literal in leading position — an optional group
//     (TypeScript's `^(export\s+)?…`), a character class, an alternation. Those
//     get no prescreen and run exactly as before. Bounding them needs a
//     first-byte SET rather than a prefix, which is left undone.
func anchoredLiteralPrefix(pattern string) []byte {
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil // already compiled by the caller, so unreachable
	}
	re = re.Simplify()
	subs := []*syntax.Regexp{re}
	if re.Op == syntax.OpConcat {
		subs = re.Sub
	}
	if len(subs) == 0 || subs[0].Op != syntax.OpBeginText {
		panic("chunk/regex: pattern " + pattern + " is not ^-anchored; " +
			"the literal-prefix prescreen assumes anchoring")
	}
	var out []byte
	for _, sub := range subs[1:] {
		if sub.Op != syntax.OpLiteral || sub.Flags&syntax.FoldCase != 0 {
			break
		}
		ascii := true
		for _, r := range sub.Rune {
			if r >= utf8.RuneSelf {
				ascii = false
				break
			}
		}
		if !ascii {
			break
		}
		for _, r := range sub.Rune {
			out = append(out, byte(r))
		}
	}
	return out
}

// anchoredFirstByteSet is the weaker prescreen for the ^-anchored patterns that
// anchoredLiteralPrefix can't handle because their leading element is an optional
// group or alternation, not a plain literal — TypeScript's `^(export\s+)?…`, where
// a match can begin with e/d/a/c/… (export/default/abstract/class). It returns the
// set of bytes any match can START with, so a line whose first byte is not in the
// set provably cannot match (same anchoring soundness as the literal prefix, one
// byte wide). nil ⇒ no useful screen: an all-nullable/`.`-leading pattern, anything
// non-ASCII or case-folded (a byte compare wouldn't be exact — bail rather than
// guess), or a set so large the prescreen would filter almost nothing (the method-
// modifier pattern, which can start with any identifier byte). The set is always a
// SUPERSET of the true first bytes, so an imprecise walk only screens less, never
// wrong.
func anchoredFirstByteSet(pattern string) *[256]bool {
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil
	}
	re = re.Simplify()
	subs := []*syntax.Regexp{re}
	if re.Op == syntax.OpConcat {
		subs = re.Sub
	}
	if len(subs) == 0 || subs[0].Op != syntax.OpBeginText {
		return nil // anchoredLiteralPrefix already panics on a non-anchored rule
	}
	var set [256]bool
	if _, ok := firstBytesConcat(subs[1:], &set); !ok {
		return nil
	}
	n := 0
	for _, b := range set {
		if b {
			n++
		}
	}
	// Empty ⇒ nothing to screen. >96 (past the ASCII printable span) ⇒ the screen
	// would reject almost no lines, so it is pure overhead — run unscreened.
	if n == 0 || n > 96 {
		return nil
	}
	return &set
}

// firstBytes adds re's possible first bytes to set and reports whether re can
// match the empty string (nullable). ok is false for a construct that cannot be
// safely reduced to an ASCII byte set (non-ASCII, `.`, case-folded) — the caller
// then bails to no prescreen.
func firstBytes(re *syntax.Regexp, set *[256]bool) (nullable, ok bool) {
	switch re.Op {
	case syntax.OpEmptyMatch, syntax.OpBeginLine, syntax.OpEndLine,
		syntax.OpBeginText, syntax.OpEndText, syntax.OpWordBoundary, syntax.OpNoWordBoundary:
		return true, true // zero-width: contributes no bytes, matches empty
	case syntax.OpLiteral:
		if re.Flags&syntax.FoldCase != 0 {
			return false, false
		}
		if len(re.Rune) == 0 {
			return true, true
		}
		if re.Rune[0] >= utf8.RuneSelf {
			return false, false
		}
		set[byte(re.Rune[0])] = true
		return false, true
	case syntax.OpCharClass:
		if re.Flags&syntax.FoldCase != 0 {
			return false, false
		}
		for i := 0; i+1 < len(re.Rune); i += 2 {
			if re.Rune[i] >= utf8.RuneSelf || re.Rune[i+1] >= utf8.RuneSelf {
				return false, false // a non-ASCII first byte we can't enumerate as bytes
			}
			for r := re.Rune[i]; r <= re.Rune[i+1]; r++ {
				set[byte(r)] = true
			}
		}
		return false, true
	case syntax.OpAnyChar, syntax.OpAnyCharNotNL:
		return false, false // any byte → useless set
	case syntax.OpStar, syntax.OpQuest:
		_, ok := firstBytes(re.Sub[0], set)
		return true, ok
	case syntax.OpPlus:
		_, ok := firstBytes(re.Sub[0], set)
		return false, ok
	case syntax.OpRepeat:
		_, ok := firstBytes(re.Sub[0], set)
		return re.Min == 0, ok
	case syntax.OpCapture:
		return firstBytes(re.Sub[0], set)
	case syntax.OpConcat:
		return firstBytesConcat(re.Sub, set)
	case syntax.OpAlternate:
		anyNull := false
		for _, s := range re.Sub {
			n, ok := firstBytes(s, set)
			if !ok {
				return false, false
			}
			anyNull = anyNull || n
		}
		return anyNull, true
	default:
		return false, false
	}
}

// firstBytesConcat walks a concatenation: each element contributes its first
// bytes until the first non-nullable one, which closes the set.
func firstBytesConcat(subs []*syntax.Regexp, set *[256]bool) (nullable, ok bool) {
	for _, s := range subs {
		n, ok := firstBytes(s, set)
		if !ok {
			return false, false
		}
		if !n {
			return false, true
		}
	}
	return true, true
}

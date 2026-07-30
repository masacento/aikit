// Package regex is the v1-default chunker (ken's DESIGN.md §2 Option C): one
// generic line-walking engine driven by per-language LanguageRules. It
// registers itself as "regex" via chunk.Register in init(); import it for
// side effects (internal/search does) — chunk must not import this package
// (import cycle), so registration is decoupled the database/sql way.
//
// This package moved here from ken (ADR-034); "DESIGN.md" refers to
// https://github.com/townsendmerino/ken/blob/main/docs/DESIGN.md.
//
// Algorithm (ken's DESIGN.md §2 "Build path"): walk lines, mark the start of each
// top-level (or member-level) definition as a candidate boundary, greedily
// accumulate lines into a chunk, and when the next line would exceed
// chunkSize snap the cut back to the latest candidate boundary that still
// fits. If a single definition is itself larger than chunkSize there is no
// boundary to snap to, so it is split at line boundaries (the line-chunker
// fallback rule). Chunks are a contiguous, non-overlapping partition, so
// concatenating Text in order reproduces the source byte-for-byte.
package regex

import (
	"bytes"
	"regexp"
	"regexp/syntax"
	"sort"
	"unicode/utf8"

	"github.com/townsendmerino/aikit/chunk"
)

type strategy int

const (
	braceStrategy  strategy = iota // top-level ⇔ brace depth 0 (C-likes)
	indentStrategy                 // top-level ⇔ no leading whitespace (Python)
)

// scannerCfg tells the brace-depth scanner which literal/comment forms to
// skip so braces inside them do not perturb the depth count.
type scannerCfg struct {
	lineComment string // e.g. "//"
	dq          bool   // "double quoted" with \ escapes
	sq          bool   // 'single quoted' with \ escapes (char/rune/JS string)
	backtick    bool   // `raw / template` (Go raw, TS template)
	tripleQuote bool   // Java text blocks: """ ... """
	rustRaw     bool   // Rust raw strings: r"..." / r#"..."#
}

// LanguageRules is the per-language driver for the generic engine.
type LanguageRules struct {
	lang     string
	defs     []*regexp.Regexp // a (left-trimmed) line that starts a definition
	skip     []*regexp.Regexp // … unless it also matches one of these (control-flow lines that look defn-ish)
	attach   []*regexp.Regexp // lines that glue onto the following def (docs, annotations, attributes, decorators)
	strat    strategy
	maxDepth int        // brace strategy: a def is a boundary iff depthBefore ≤ maxDepth
	scan     scannerCfg // brace strategy only

	// Literal prefixes for the three rule sets above, one per regexp, filled by
	// register. An empty entry means "no prescreen available, run the regexp".
	defsPre, skipPre, attachPre [][]byte
}

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

// languageRules is populated by each per-language file's init().
var languageRules = map[string]LanguageRules{}

// Chunker implements chunk.Chunker.
type Chunker struct{ rules map[string]LanguageRules }

// New returns a Chunker over every registered language ruleset.
func New() *Chunker { return &Chunker{rules: languageRules} }

func init() { chunk.Register("regex", New()) }

func (*Chunker) Name() string { return "regex" }

func (c *Chunker) SupportedLanguages() []string {
	out := make([]string, 0, len(c.rules))
	for l := range c.rules {
		out = append(out, l)
	}
	sort.Strings(out)
	return out
}

func (c *Chunker) Chunk(source []byte, language string, chunkSize int) ([]chunk.Chunk, error) {
	if chunkSize <= 0 {
		chunkSize = chunk.DefaultChunkSize
	}
	r, ok := c.rules[language]
	if !ok {
		// Defensive: ChunkFile routes unsupported languages to the line
		// chunker, so this is rare. Degrade to size-only splitting (no
		// defs ⇒ no boundaries) — still a valid byte-exact partition.
		r = LanguageRules{lang: language, strat: braceStrategy, maxDepth: -1}
	}
	return chunkWith(r, source, chunkSize), nil
}

func chunkWith(r LanguageRules, src []byte, chunkSize int) []chunk.Chunk {
	if len(src) == 0 {
		return nil
	}

	// lineStart[i] = byte offset of line i. A trailing '\n' does not start
	// a phantom empty line (matches the Stage 1 line chunker).
	lineStart := []int{0}
	for i := range src {
		if src[i] == '\n' && i+1 < len(src) {
			lineStart = append(lineStart, i+1)
		}
	}
	n := len(lineStart)
	off := func(k int) int {
		if k >= n {
			return len(src)
		}
		return lineStart[k]
	}
	rawLine := func(i int) []byte { return src[off(i):off(i+1)] }

	// Per-line "is this the start of a definition?" with attachment of the
	// preceding doc-comment / annotation / decorator block.
	var depth []int
	if r.strat == braceStrategy {
		depth = scanDepth(src, lineStart, r.scan)
	}
	isBoundary := make([]bool, n)
	for i := range n {
		line := rawLine(i)
		var probe []byte
		switch r.strat {
		case indentStrategy:
			// Top-level only: a definition with no leading whitespace.
			if len(line) > 0 && (line[0] == ' ' || line[0] == '\t') {
				continue
			}
			probe = line
		default: // braceStrategy
			if r.maxDepth < 0 || depth[i] > r.maxDepth {
				continue
			}
			probe = bytes.TrimLeft(line, " \t")
		}
		if !anyMatch(r.defs, r.defsPre, probe) || anyMatch(r.skip, r.skipPre, probe) {
			continue
		}
		// Snap the boundary up over a contiguous attach block (so a
		// function keeps its doc comment / @annotation / #[attr] / @decorator).
		b := i
		// b > 0, not b-1 >= 1: the latter halts at b==1, so an attach block
		// (license header, module doc) that starts at line 0 and runs into the
		// first definition marked isBoundary[1] — a line INSIDE the comment —
		// producing a single-line first chunk holding only line 0 (audit #22).
		for b > 0 && attachMatch(r, rawLine(b-1)) {
			b--
		}
		isBoundary[b] = true
	}

	var boundaries []int // sorted line indices > 0 where a chunk may start
	for i := 1; i < n; i++ {
		if isBoundary[i] {
			boundaries = append(boundaries, i)
		}
	}
	// lastBoundaryLE: greatest boundary b with lo < b ≤ hi, else -1.
	lastBoundaryLE := func(lo, hi int) int {
		j := sort.Search(len(boundaries), func(k int) bool { return boundaries[k] > hi })
		if j == 0 {
			return -1
		}
		if b := boundaries[j-1]; b > lo {
			return b
		}
		return -1
	}

	var out []chunk.Chunk
	emit := func(a, b int) {
		out = append(out, chunk.Chunk{
			StartLine: a + 1,
			EndLine:   b,
			Text:      string(src[off(a):off(b)]),
		})
	}

	start := 0
	for start < n {
		end := start + 1
		for end < n && off(end+1)-off(start) <= chunkSize {
			end++
		}
		if end >= n {
			emit(start, n)
			break
		}
		if b := lastBoundaryLE(start, end); b > start {
			emit(start, b) // snap the cut back to a definition boundary
			start = b
			continue
		}
		emit(start, end) // oversized single unit ⇒ line-split fallback
		start = end
	}
	return out
}

// anyMatch reports whether line matches any of res, skipping the regexp engine
// for patterns whose literal prefix the line does not start with. See register
// for why that is exact rather than a heuristic.
func anyMatch(res []*regexp.Regexp, pre [][]byte, line []byte) bool {
	for i, re := range res {
		if p := pre[i]; len(p) > 0 && !bytes.HasPrefix(line, p) {
			continue
		}
		if re.Match(line) {
			return true
		}
	}
	return false
}

// attachMatch reports whether line glues onto the following definition. A
// blank line never attaches (it separates a def from anything above it).
func attachMatch(r LanguageRules, line []byte) bool {
	probe := line
	if r.strat == braceStrategy {
		probe = bytes.TrimLeft(line, " \t")
	}
	if len(bytes.TrimSpace(probe)) == 0 {
		return false
	}
	return anyMatch(r.attach, r.attachPre, probe)
}

// scanDepth returns the brace depth at the START of each line, ignoring
// braces inside comments and string/char literals. Best-effort: an
// undercount inside an exotic literal only yields a suboptimal boundary,
// never data loss (chunks are always a contiguous byte partition).
func scanDepth(src []byte, lineStart []int, cfg scannerCfg) []int {
	n := len(lineStart)
	depth := make([]int, n)
	cur := 0
	nextLineIdx := 0 // next line whose start we still need to record

	type st int
	const (
		normal st = iota
		lineCmt
		blockCmt
		inDq
		inSq
		inBt
		inTriple // Java text block
		inRawN   // Rust raw string, hashes counted in rawHashes
	)
	state := normal
	rawHashes := 0
	cmtMark := cfg.lineComment

	// nextPos is lineStart[nextLineIdx] hoisted into a scalar, or a sentinel past
	// the end of the input once the line starts are exhausted. It replaces a
	// closure called once per BYTE of every indexed file whose body — a
	// bounds-checked slice load and a compare — was false ~97% of the time
	// (lens doc §3.1: scanDepth.func1 was 8.1% flat of chunkWith). The loop below
	// now touches `depth` only at an actual line start.
	nextPos := len(src) + 1
	if n > 0 {
		nextPos = lineStart[0]
	}
	// recordLineStarts is the `<=`, not `==`, rule, unchanged: a state handler can
	// advance i past a byte (an escape `\<c>` in a string, `*/`, `"""`, …). If the
	// skipped byte is a line start, `== pos` would never match it, nextLineIdx
	// would stall, and every subsequent line's depth would be frozen at 0 for the
	// rest of the file (e.g. a `\` followed by a newline — a legal JS/Rust line
	// continuation). `<=` records any jumped-over line starts at the current depth.
	recordLineStarts := func(pos int) {
		for nextLineIdx < n && lineStart[nextLineIdx] <= pos {
			depth[nextLineIdx] = cur
			nextLineIdx++
		}
		if nextLineIdx < n {
			nextPos = lineStart[nextLineIdx]
		} else {
			nextPos = len(src) + 1
		}
	}

	// cmt0 is cmtMark's first byte, so the hasPrefixAt below can be gated on a
	// byte compare. cmtMark is a string VARIABLE, so `string(src[i:i+len(s)]) == s`
	// cannot be specialized into byte compares — it lowers to a runtime.memequal
	// CALL, made once per byte of source in the normal state (lens doc §3.1:
	// memeqbody 13.7% flat, hasPrefixAt 24.2% cumulative). Gating on the first
	// byte skips the call for the ~99% of bytes that cannot possibly start the
	// mark, and cannot change the outcome: hasPrefixAt(src, i, s) already requires
	// src[i] == s[0].
	var cmt0 byte
	if cmtMark != "" {
		cmt0 = cmtMark[0]
	}

	for i := 0; i < len(src); i++ {
		if i >= nextPos {
			recordLineStarts(i)
		}
		c := src[i]
		switch state {
		case normal:
			switch {
			case c == cmt0 && cmtMark != "" && hasPrefixAt(src, i, cmtMark):
				state = lineCmt
				i += len(cmtMark) - 1
			case c == '/' && hasPrefixAt(src, i, "/*"):
				state = blockCmt
				i++
			case c == '"' && cfg.tripleQuote && hasPrefixAt(src, i, `"""`):
				state = inTriple
				i += 2
			case cfg.rustRaw && c == 'r' && (peek(src, i+1) == '"' || peek(src, i+1) == '#'):
				j := i + 1
				h := 0
				for peek(src, j) == '#' {
					h++
					j++
				}
				if peek(src, j) == '"' {
					rawHashes = h
					state = inRawN
					i = j
				}
			case cfg.dq && c == '"':
				state = inDq
			case cfg.sq && c == '\'':
				state = inSq
			case cfg.backtick && c == '`':
				state = inBt
			case c == '{':
				cur++
			case c == '}':
				if cur > 0 {
					cur--
				}
			}
		case lineCmt:
			if c == '\n' {
				state = normal
			}
		case blockCmt:
			if c == '*' && peek(src, i+1) == '/' {
				state = normal
				i++
			}
		case inDq:
			if c == '\\' {
				i++
			} else if c == '"' {
				state = normal
			}
		case inSq:
			if c == '\\' {
				i++
			} else if c == '\'' {
				state = normal
			}
		case inBt:
			if c == '`' {
				state = normal
			}
		case inTriple:
			if c == '"' && hasPrefixAt(src, i, `"""`) {
				state = normal
				i += 2
			}
		case inRawN:
			if c == '"' {
				j := i + 1
				h := 0
				for h < rawHashes && peek(src, j) == '#' {
					h++
					j++
				}
				if h == rawHashes {
					state = normal
					i = j - 1
				}
			}
		}
	}
	recordLineStarts(len(src))
	return depth
}

func hasPrefixAt(src []byte, i int, s string) bool {
	if i+len(s) > len(src) {
		return false
	}
	return string(src[i:i+len(s)]) == s
}

func peek(src []byte, i int) byte {
	if i < 0 || i >= len(src) {
		return 0
	}
	return src[i]
}

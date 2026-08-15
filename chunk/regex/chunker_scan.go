package regex

// scanDepth is the hand-rolled brace-depth state machine: a single pass over the
// source that tracks nesting depth while skipping braces inside comments and
// string/char literals (so `if s := "{" ...` does not perturb the count). Its
// depth[] output is chunkWith's boundary test for the brace strategy (C-likes).
// Split out of chunker.go — one file per concern in what was an 850+ line type;
// see chunker.go's doc comment.

// scanDepth returns the brace depth at the START of each line, ignoring
// braces inside comments and string/char literals. Best-effort: an
// undercount inside an exotic literal only yields a suboptimal boundary,
// never data loss (chunks are always a contiguous byte partition).
func scanDepth(src []byte, lineStart []int, cfg scannerCfg) []int32 {
	n := len(lineStart)
	// int32, not int: this is compared against maxDepth ∈ {-1, 0, 1} and a brace
	// nesting level that no real source approaches, so 8 B/line was twice what
	// the value needs (lens doc §4.8).
	depth := make([]int32, n)
	cur := int32(0)
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

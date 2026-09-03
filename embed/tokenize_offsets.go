package embed

// tokenize_offsets.go — per-piece character-offset tracking for the WordPiece
// path: EncodeOffsets is the offset_mapping half of HF fast tokenizers, which
// the ids-only Encode deliberately does not carry (tokenize.go).
//
// It reproduces the same pipeline as Encode — added-token carve-out on the RAW
// text, then BertNormalizer → BertPreTokenizer → WordPiece — but every stage
// also propagates WHERE each output rune came from, so every emitted piece gets
// a byte span [start, end) into the ORIGINAL input. These are byte offsets, not
// Python's character offsets; on ASCII they coincide, and Go callers slice with
// them directly (the same convention ner.Entity documents).
//
// Two normalization shortcuts are exact for the BertNormalizer configurations
// aikit targets (BERT-uncased family) but worth stating:
//
//   - NFD runs per rune rather than over the whole string. Canonical
//     reordering only rearranges combining marks within one base's sequence,
//     and strip_accents then DROPS those marks, so the surviving rune sequence
//     — and therefore every offset — is identical either way.
//   - lowercasing uses Go's simple per-rune mapping, which never expands one
//     rune into two (Rust's full mapping can, e.g. İ → i + U+0307). With
//     strip_accents on, HF drops that combining dot before lowercasing and the
//     two agree; a cased tokenizer with strip_accents explicitly off and such
//     an input would offset differently.

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// EncodeOffsets tokenizes text to WordPiece IDs WITHOUT special-token wrapping
// and returns one byte span per piece into the original text:
//
//	ids[k] covers text[offs[k][0]:offs[k][1]]
//
// The span is half-open and always within bounds; a piece whose span is empty
// cannot occur on the WordPiece path. Added-token literals ([CLS], [MASK], …)
// found verbatim in the input report the literal's own raw span, exactly as HF
// fast tokenizers do for added tokens.
//
// Only the WordPiece pipeline (BertNormalizer + BertPreTokenizer) supports
// offsets; the Unigram/BPE backends return an error rather than a silent empty
// span.
func (t *Tokenizer) EncodeOffsets(text string) ([]int32, [][2]int, error) {
	if t.uni != nil || t.bpe != nil {
		return nil, nil, fmt.Errorf("embed: EncodeOffsets supports WordPiece tokenizers only")
	}
	var (
		ids  []int32
		offs [][2]int
	)
	// Added-token carve-out, rune-stepped so offsets stay valid on any input.
	// Same match rule as Encode (raw text, longest-first, no word-boundary
	// check — BERT's specials carry single_word=false).
	start, i := 0, 0
	for i < len(text) {
		matched := ""
		for _, k := range t.addedKeys {
			if strings.HasPrefix(text[i:], k) {
				matched = k
				break
			}
		}
		if matched != "" {
			if i > start {
				t.encodeSegmentOffsets(text[start:i], start, &ids, &offs)
			}
			ids = append(ids, t.addedTokens[matched])
			offs = append(offs, [2]int{i, i + len(matched)})
			i += len(matched)
			start = i
			continue
		}
		_, w := utf8.DecodeRuneInString(text[i:])
		i += w
	}
	if start < len(text) {
		t.encodeSegmentOffsets(text[start:], start, &ids, &offs)
	}
	return ids, offs, nil
}

// encodeSegmentOffsets runs normalize → pre-tokenize → WordPiece over a
// carve-out-free fragment, tracking every piece's origin. base is the fragment's
// byte offset in the caller's full text; all spans are emitted absolute.
func (t *Tokenizer) encodeSegmentOffsets(seg string, base int, ids *[]int32, offs *[][2]int) {
	x := t.normalizeTracked(seg)
	for _, tok := range runePreTokens(x.r) {
		a, b := tok[0], tok[1]
		sByte, eByte := x.o[a], x.o[b-1]+srcRuneLen(seg, x.o[b-1])
		// Defensive added-token check, mirroring encodeSegment: a normalized
		// word that IS an added literal emits that one id.
		if id, ok := t.addedTokens[string(x.r[a:b])]; ok {
			*ids = append(*ids, id)
			*offs = append(*offs, [2]int{base + sByte, base + eByte})
			continue
		}
		pids, ends := t.wordPieceSpans(x.r[a:b])
		prev := a
		for k, id := range pids {
			rs, re := prev, a+ends[k]
			ps, pe := x.o[rs], x.o[re-1]+srcRuneLen(seg, x.o[re-1])
			*ids = append(*ids, id)
			*offs = append(*offs, [2]int{base + ps, base + pe})
			prev = re
		}
	}
}

// srcRuneLen is the byte length of the ORIGINAL rune at byte off in seg — the
// rune a normalized output rune traces back to. Normalization can shrink a rune
// (İ → i after accent strip) but never merges two originals into one output
// rune, so decoding at the origin always yields the right extent.
func srcRuneLen(seg string, off int) int {
	_, w := utf8.DecodeRuneInString(seg[off:])
	return w
}

// offRunes is a rune sequence paired with each rune's origin: o[k] is the byte
// offset in the source fragment of the original rune x.r[k] came from. Runes
// synthesized by normalization (the spaces around CJK) share their base's
// origin; runes deleted by normalization (controls, combining marks) simply
// disappear.
type offRunes struct {
	r []rune
	o []int
}

// normalizeTracked applies the BertNormalizer steps in HF's order — cleanText,
// handleCJK, stripAccents, lowercase — carrying origins through each.
func (t *Tokenizer) normalizeTracked(seg string) offRunes {
	var x offRunes
	for i := 0; i < len(seg); {
		r, w := utf8.DecodeRuneInString(seg[i:])
		x.r = append(x.r, r)
		x.o = append(x.o, i)
		i += w
	}
	if t.cleanText {
		x = x.filter(func(r rune) rune {
			if r == 0 || r == 0xFFFD || isControl(r) {
				return -1
			}
			if isWhitespace(r) {
				return ' '
			}
			return r
		})
	}
	if t.handleCJK {
		x = x.expand(func(r rune) []rune {
			if !isCJK(r) {
				return nil
			}
			return []rune{' ', r, ' '}
		})
	}
	if t.stripAccents {
		x = x.mapRunes(func(r rune) []rune {
			dec := []rune(norm.NFD.String(string(r)))
			out := dec[:0]
			for _, d := range dec {
				if !unicode.Is(unicode.Mn, d) {
					out = append(out, d)
				}
			}
			return out
		})
	}
	if t.lowercase {
		x = x.mapRunes(func(r rune) []rune { return []rune{unicode.ToLower(r)} })
	}
	return x
}

// filter keeps each rune's image (dropping it when keep returns a negative
// rune), preserving origins.
func (x offRunes) filter(keep func(rune) rune) offRunes {
	out := offRunes{x.r[:0:0], x.o[:0:0]}
	for i, r := range x.r {
		m := keep(r)
		if m < 0 {
			continue
		}
		out.r = append(out.r, m)
		out.o = append(out.o, x.o[i])
	}
	return out
}

// expand lets one rune become several (all sharing its origin); returning nil
// keeps the rune unchanged.
func (x offRunes) expand(f func(rune) []rune) offRunes {
	out := offRunes{x.r[:0:0], x.o[:0:0]}
	for i, r := range x.r {
		if extra := f(r); extra != nil {
			for _, e := range extra {
				out.r = append(out.r, e)
				out.o = append(out.o, x.o[i])
			}
			continue
		}
		out.r = append(out.r, r)
		out.o = append(out.o, x.o[i])
	}
	return out
}

// mapRunes maps each rune to zero or more runes (all sharing its origin).
func (x offRunes) mapRunes(f func(rune) []rune) offRunes {
	out := offRunes{x.r[:0:0], x.o[:0:0]}
	for i, r := range x.r {
		for _, m := range f(r) {
			out.r = append(out.r, m)
			out.o = append(out.o, x.o[i])
		}
	}
	return out
}

// runePreTokens splits a rune sequence the way preTokenize splits bytes:
// whitespace runs separate tokens, and within a run each punctuation rune is a
// token of its own. Returns half-open rune index ranges; every range is
// non-empty.
func runePreTokens(r []rune) [][2]int {
	var out [][2]int
	n := len(r)
	i := 0
	for i < n {
		if unicode.IsSpace(r[i]) {
			i++
			continue
		}
		a := i
		for i < n && !unicode.IsSpace(r[i]) {
			if isPunct(r[i]) {
				if i > a {
					out = append(out, [2]int{a, i})
				}
				out = append(out, [2]int{i, i + 1})
				a = i + 1
			}
			i++
		}
		if a < i {
			out = append(out, [2]int{a, i})
		}
	}
	return out
}

// wordPieceSpans is wordPieceCompute with boundaries: alongside each piece id
// it reports the rune END index (exclusive) of that piece within chars, so a
// caller can map pieces back to character positions. Failure semantics are the
// WordPiece ones — a word over maxCharsPerWord, or with no matching prefix at
// any position, becomes a single [UNK] spanning the WHOLE word, exactly the
// offsets HF reports for it.
func (t *Tokenizer) wordPieceSpans(chars []rune) (ids []int32, ends []int) {
	if len(chars) > t.maxCharsPerWord {
		return []int32{t.unkID}, []int{len(chars)}
	}
	if len(chars) == 0 {
		return nil, nil
	}
	var b []byte
	start := 0
	for start < len(chars) {
		end := len(chars)
		matched := false
		for end > start {
			b = b[:0]
			if start > 0 {
				b = append(b, t.continuingPrefix...)
			}
			for _, r := range chars[start:end] {
				b = utf8.AppendRune(b, r)
			}
			if id, ok := t.vocab[string(b)]; ok {
				ids = append(ids, id)
				ends = append(ends, end)
				start = end
				matched = true
				break
			}
			end--
		}
		if !matched {
			return []int32{t.unkID}, []int{len(chars)}
		}
	}
	return ids, ends
}

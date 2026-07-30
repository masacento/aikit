package embed

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// ──────────────────────────────────────────────────────────────────────────────
// Byte-level BPE (GPT-2 / RoBERTa)
// ──────────────────────────────────────────────────────────────────────────────
//
// RoBERTa-family models (granite-embedding-*-english) tokenize with GPT-2's
// byte-level BPE: no normalizer; the text is split by the GPT-2 regex, each
// piece's raw UTF-8 bytes are mapped to a reversible set of printable runes, and
// ranked merges are applied within each piece. Every one of the 256 byte-runes is
// in the vocab, so this model never emits <unk>. Wrapped <s>…</s> by a
// RobertaProcessing post-processor. This reproduces HF `tokenizers` id-for-id.
//
// The one subtlety is the pre-tokenizer. GPT-2's regex ends with
// `…|\s+(?!\S)|\s+`; Go's RE2 has no lookahead, so we drop the `(?!\S)` clause and
// reproduce its effect with a give-back post-pass (see preTokenize): a maximal
// whitespace run that is followed by a non-space gives back its last character —
// a space becomes the leading space of the next token, a tab/newline becomes its
// own piece. A terminal run stays whole.

// gpt2Pattern is the GPT-2 pre-tokenizer regex with the `\s+(?!\S)` lookahead
// clause folded into the trailing `\s+` (the give-back post-pass in preTokenize
// restores the lookahead's behavior). \s is ASCII here, which covers the
// space/tab/newline whitespace these models see.
var gpt2Pattern = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

// bytesToUnicode builds GPT-2's byte→rune table. The 188 "printable" bytes map to
// themselves; the other 68 map, in ascending byte order, to U+0100, U+0101, …
// This is a bijection onto a set of printable runes, so merges never straddle a
// UTF-8 boundary and every byte has a vocab symbol.
func bytesToUnicode() [256]rune {
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	var table [256]rune
	n := 0
	for b := range 256 {
		if printable(b) {
			table[b] = rune(b)
		} else {
			table[b] = rune(256 + n)
			n++
		}
	}
	return table
}

// bpeBackend is the byte-level BPE tokenizer. It plugs into embed.Tokenizer, which
// dispatches to it when tokenizer.json's model.type is "BPE".
type bpeBackend struct {
	byte2rune   [256]rune
	vocab       map[string]int32  // byte-mapped symbol → id
	rank        map[[2]string]int // adjacent-symbol pair → merge priority (lower = earlier)
	addedTokens map[string]int32
	addedKeys   []string // longest-first carve-out scan order
	unkID       int32
	vocabSize   int
	prefixIDs   []int32 // RobertaProcessing: <s> before the sequence
	suffixIDs   []int32 // ... and </s> after
}

// isAllSpaceASCII reports whether s is a non-empty run of ASCII whitespace.
func isAllSpaceASCII(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\n', '\r', '\f', '\v':
		default:
			return false
		}
	}
	return true
}

// preTokenize splits text into GPT-2 pieces, applying the whitespace give-back
// that replaces the dropped `\s+(?!\S)` lookahead: a whitespace-only match that is
// not the final match gives back its last character to the following token (a
// space) or emits it as its own piece (a non-space whitespace char).
func (b *bpeBackend) preTokenize(text string) []string {
	matches := gpt2Pattern.FindAllString(text, -1)
	out := make([]string, 0, len(matches))
	for i := range matches {
		m := matches[i]
		if i < len(matches)-1 && isAllSpaceASCII(m) {
			last := m[len(m)-1]
			head := m[:len(m)-1]
			if last == ' ' {
				if head != "" {
					out = append(out, head)
				}
				matches[i+1] = " " + matches[i+1] // becomes the next token's leading space
				continue
			}
			// tab/newline/etc.: leading run is one piece, the last char its own.
			if head != "" {
				out = append(out, head)
			}
			out = append(out, string(last))
			continue
		}
		out = append(out, m)
	}
	return out
}

// bpe applies the ranked merges to one byte-mapped piece: repeatedly find the
// adjacent symbol pair with the lowest merge rank and merge every occurrence of
// it, until no adjacent pair is a known merge. Matches GPT-2 bpe() / HF Rust BPE.
func (b *bpeBackend) bpe(mapped string) []string {
	symbols := make([]string, 0, len(mapped))
	for _, r := range mapped {
		symbols = append(symbols, string(r))
	}
	if len(symbols) < 2 {
		return symbols
	}
	for {
		bestRank := int(^uint(0) >> 1)
		bestI := -1
		for i := 0; i < len(symbols)-1; i++ {
			if r, ok := b.rank[[2]string{symbols[i], symbols[i+1]}]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		a, c := symbols[bestI], symbols[bestI+1]
		merged := symbols[:0:0] // fresh backing so we don't clobber while scanning
		for i := 0; i < len(symbols); {
			if i < len(symbols)-1 && symbols[i] == a && symbols[i+1] == c {
				merged = append(merged, a+c)
				i += 2
			} else {
				merged = append(merged, symbols[i])
				i++
			}
		}
		symbols = merged
	}
	return symbols
}

// encode runs the added-token carve-out, then byte-level BPE over each segment,
// producing the BARE id sequence (no <s>/</s>).
func (b *bpeBackend) encode(text string) []int32 {
	if len(b.addedKeys) == 0 {
		return b.encodeSegment(text)
	}
	var out []int32
	var seg strings.Builder
	flush := func() {
		if seg.Len() > 0 {
			out = append(out, b.encodeSegment(seg.String())...)
			seg.Reset()
		}
	}
	for i := 0; i < len(text); {
		matched := ""
		for _, k := range b.addedKeys {
			if strings.HasPrefix(text[i:], k) {
				matched = k
				break
			}
		}
		if matched != "" {
			flush()
			out = append(out, b.addedTokens[matched])
			i += len(matched)
			continue
		}
		_, size := utf8.DecodeRuneInString(text[i:])
		if size == 0 {
			size = 1
		}
		seg.WriteString(text[i : i+size])
		i += size
	}
	flush()
	return out
}

func (b *bpeBackend) encodeSegment(text string) []int32 {
	var ids []int32
	for _, piece := range b.preTokenize(text) {
		var sb strings.Builder
		sb.Grow(len(piece))
		for i := 0; i < len(piece); i++ { // map each RAW byte to its GPT-2 rune
			sb.WriteRune(b.byte2rune[piece[i]])
		}
		for _, sym := range b.bpe(sb.String()) {
			id, ok := b.vocab[sym]
			if !ok {
				id = b.unkID // unreachable: every byte-symbol is in the vocab
			}
			ids = append(ids, id)
		}
	}
	return ids
}

// encodeWithSpecials wraps encode with the RobertaProcessing specials
// (<s> ++ body ++ </s>), right-truncating body so the total is at most maxLen.
func (b *bpeBackend) encodeWithSpecials(text string, maxLen int) []int32 {
	fixed := len(b.prefixIDs) + len(b.suffixIDs)
	if maxLen < fixed {
		maxLen = fixed
	}
	body := b.encode(text)
	if len(body) > maxLen-fixed {
		body = body[:maxLen-fixed]
	}
	out := make([]int32, 0, len(body)+fixed)
	out = append(out, b.prefixIDs...)
	out = append(out, body...)
	out = append(out, b.suffixIDs...)
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// tokenizer.json → bpeBackend
// ──────────────────────────────────────────────────────────────────────────────

type bpeJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Model struct {
		Type   string           `json:"type"`
		Vocab  map[string]int32 `json:"vocab"`
		Merges []string         `json:"merges"`
	} `json:"model"`
	PostProcessor struct {
		Type string            `json:"type"`
		Cls  []json.RawMessage `json:"cls"` // ["<s>", 0]
		Sep  []json.RawMessage `json:"sep"` // ["</s>", 2]
	} `json:"post_processor"`
}

// isBPETokenizer reports whether tokenizer.json is a byte-level BPE model.
func isBPETokenizer(model string) bool { return model == "BPE" }

// parseBPETokenizer builds a bpeBackend from tokenizer.json bytes.
func parseBPETokenizer(data []byte) (*bpeBackend, error) {
	var raw bpeJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json (bpe): %w", err)
	}
	if len(raw.Model.Vocab) == 0 {
		return nil, fmt.Errorf("bpe: empty vocab")
	}

	rank := make(map[[2]string]int, len(raw.Model.Merges))
	for i, m := range raw.Model.Merges {
		parts := strings.SplitN(m, " ", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("bpe: merge[%d] %q not a space-joined pair", i, m)
		}
		rank[[2]string{parts[0], parts[1]}] = i
	}

	added := make(map[string]int32, len(raw.AddedTokens))
	for _, at := range raw.AddedTokens {
		added[at.Content] = at.ID
	}
	addedKeys := make([]string, 0, len(added))
	for k := range added {
		addedKeys = append(addedKeys, k)
	}
	sort.Slice(addedKeys, func(i, j int) bool {
		if len(addedKeys[i]) != len(addedKeys[j]) {
			return len(addedKeys[i]) > len(addedKeys[j])
		}
		return addedKeys[i] < addedKeys[j]
	})

	// RobertaProcessing wraps <s> (cls) ++ body ++ </s> (sep).
	prefixIDs, err := postProcID(raw.PostProcessor.Cls)
	if err != nil {
		return nil, fmt.Errorf("bpe: post_processor cls: %w", err)
	}
	suffixIDs, err := postProcID(raw.PostProcessor.Sep)
	if err != nil {
		return nil, fmt.Errorf("bpe: post_processor sep: %w", err)
	}
	// unk id: byte-level BPE never emits it, but expose <unk> if present.
	unkID := int32(0)
	if id, ok := added["<unk>"]; ok {
		unkID = id
	} else if id, ok := raw.Model.Vocab["<unk>"]; ok {
		unkID = id
	}

	return &bpeBackend{
		byte2rune:   bytesToUnicode(),
		vocab:       raw.Model.Vocab,
		rank:        rank,
		addedTokens: added,
		addedKeys:   addedKeys,
		unkID:       unkID,
		vocabSize:   len(raw.Model.Vocab),
		prefixIDs:   prefixIDs,
		suffixIDs:   suffixIDs,
	}, nil
}

// postProcID reads a RobertaProcessing ["<literal>", id] pair into a 1-element id
// slice. An absent (nil) pair yields no id (a bare-LM checkpoint with no wrap).
func postProcID(pair []json.RawMessage) ([]int32, error) {
	if len(pair) == 0 {
		return nil, nil
	}
	if len(pair) != 2 {
		return nil, fmt.Errorf("expected [literal, id], got %d elems", len(pair))
	}
	var id int32
	if err := json.Unmarshal(pair[1], &id); err != nil {
		return nil, fmt.Errorf("id: %w", err)
	}
	return []int32{id}, nil
}

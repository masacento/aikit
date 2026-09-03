package embed

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
	"unsafe"

	"golang.org/x/text/unicode/norm"
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
// The OLMo/ModernBERT exports (Ettin, cross-encoder/ettin-reranker-*) are the same
// model with three newer spellings, all handled below: `merges` as [a,b] pairs
// rather than "a b" strings (`tokenizers` ≥0.20), a TemplateProcessing
// post-processor rather than RobertaProcessing, and an explicit NFC normalizer
// rather than none. The first is a hard parse failure; the other two are silent —
// an unhandled post-processor drops [CLS]/[SEP] from every sequence and an ignored
// normalizer only diverges on non-ASCII — so both are validated, not skipped.
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
// resolveUnkID finds the <unk> token id, preferring the added-tokens table over
// the base vocab (an added entry overrides a same-named vocab entry everywhere
// else in this parser, so it does here too). Byte-level BPE never actually emits
// <unk> — every byte is representable — but the id is exposed when the
// checkpoint defines one, and 0 is the documented "absent" value. The OLMo
// family spells it [UNK]; GPT-2/RoBERTa spell it <unk>. Without the [UNK]
// probe the fallback leaves unkID at 0, which for those checkpoints is a
// real added token rather than a sentinel.
func resolveUnkID(added map[string]int32, vocab map[string]int32) int32 {
	for _, lit := range [...]string{"<unk>", "[UNK]"} {
		if id, ok := added[lit]; ok {
			return id
		}
		if id, ok := vocab[lit]; ok {
			return id
		}
	}
	return 0
}

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
	// addedLstrip / addedRstrip hold only the added tokens whose flag is set (a
	// handful per vocab, usually just [MASK]); absent means false.
	addedLstrip map[string]bool
	addedRstrip map[string]bool
	unkID       int32
	vocabSize   int
	// normNFC applies Unicode NFC before pre-tokenization, for the exports that
	// declare it (OLMo/ModernBERT). GPT-2 and RoBERTa declare no normalizer.
	normNFC   bool
	prefixIDs []int32 // specials before the sequence (<s> / [CLS])
	suffixIDs []int32 // ... and after (</s> / [SEP])
}

// isSpaceByteASCII reports whether c is one of the ASCII whitespace bytes. In valid
// UTF-8 none of these can occur inside a multi-byte rune, so the lstrip/rstrip scans
// in encode may walk bytes rather than runes.
func isSpaceByteASCII(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '\f', '\v':
		return true
	}
	return false
}

// isAllSpaceASCII reports whether s is a non-empty run of ASCII whitespace.
func isAllSpaceASCII(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if !isSpaceByteASCII(s[i]) {
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

// bpeSpan is a symbol as a [lo,hi) byte range of the byte-mapped piece. Every BPE
// symbol is, by induction, a CONTIGUOUS substring of `mapped`: the initial symbols
// are its runes in order, and a merge joins two adjacent ranges. So the whole merge
// loop runs on offsets — the rank-map keys and the final vocab lookups take free
// sub-slices of `mapped`, and a merge is just span{lo1, hi2} (lens §3.3).
type bpeSpan struct{ lo, hi int }

// bpeScratch is the per-segment scratch bpeInto reuses across a segment's pieces:
// mapped holds the current piece's byte-mapped bytes (so no strings.Builder alloc
// per piece), and cur/nxt are the merge loop's double buffer (so no fresh backing
// per merge round).
type bpeScratch struct {
	mapped   []byte
	cur, nxt []bpeSpan
}

// bpeInto applies the ranked merges to one byte-mapped piece — repeatedly find the
// adjacent symbol pair with the lowest merge rank and merge every occurrence, until
// no adjacent pair is a known merge — and appends the resulting symbol ids to ids.
// Matches GPT-2 bpe() / HF Rust BPE. Bit-identical to the prior string-based bpe
// (same greedy lowest-rank merges, same symbols); the only change is that it works
// on spans, so there is no per-rune string(r), no fresh backing per merge, and no
// a+c concat (lens §3.3: 4.78 M allocs → ~0, 2.83×).
func (b *bpeBackend) bpeInto(mapped string, sc *bpeScratch, ids []int32) []int32 {
	cur := sc.cur[:0]
	for i := 0; i < len(mapped); {
		_, size := utf8.DecodeRuneInString(mapped[i:])
		cur = append(cur, bpeSpan{i, i + size})
		i += size
	}
	nxt := sc.nxt[:0]
	for len(cur) >= 2 {
		bestRank := int(^uint(0) >> 1)
		bestI := -1
		for i := 0; i < len(cur)-1; i++ {
			key := [2]string{mapped[cur[i].lo:cur[i].hi], mapped[cur[i+1].lo:cur[i+1].hi]}
			if r, ok := b.rank[key]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		aS := mapped[cur[bestI].lo:cur[bestI].hi]
		cS := mapped[cur[bestI+1].lo:cur[bestI+1].hi]
		nxt = nxt[:0]
		for i := 0; i < len(cur); {
			if i < len(cur)-1 && mapped[cur[i].lo:cur[i].hi] == aS && mapped[cur[i+1].lo:cur[i+1].hi] == cS {
				nxt = append(nxt, bpeSpan{cur[i].lo, cur[i+1].hi})
				i += 2
			} else {
				nxt = append(nxt, cur[i])
				i++
			}
		}
		cur, nxt = nxt, cur // swap; nxt is reset to [:0] at the top of the next round
	}
	for _, s := range cur {
		id, ok := b.vocab[mapped[s.lo:s.hi]]
		if !ok {
			id = b.unkID // unreachable: every byte-symbol is in the vocab
		}
		ids = append(ids, id)
	}
	sc.cur, sc.nxt = cur, nxt // retain both grown backings for the next piece
	return ids
}

// encode runs the added-token carve-out, then byte-level BPE over each segment,
// producing the BARE id sequence (no <s>/</s>).
//
// Segments are contiguous ranges of text, so they are sliced rather than rebuilt.
// A matched added token honours its lstrip / rstrip flags: HF's AddedVocabulary
// compiles those into the match pattern as `\s*`, so the whitespace they name is
// consumed by the added token instead of reaching the BPE. [MASK] is lstrip in
// every BERT-lineage vocab (and <mask> in RoBERTa's), so ignoring the flags leaves
// a stray space token before every mask.
func (b *bpeBackend) encode(text string) []int32 {
	if len(b.addedKeys) == 0 {
		return b.encodeSegment(text)
	}
	var out []int32
	segStart := 0
	flush := func(end int) {
		if end > segStart {
			out = append(out, b.encodeSegment(text[segStart:end])...)
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
		if matched == "" {
			_, size := utf8.DecodeRuneInString(text[i:])
			if size == 0 {
				size = 1
			}
			i += size
			continue
		}
		// lstrip only ever reaches whitespace that was NOT itself carved out as an
		// added token: this scan is left-to-right, so a vocab whitespace run (ids
		// 50254-50276 in the OLMo vocab) has already been emitted at its own
		// position — which is what HF's leftmost-match regex does too.
		end := i
		if b.addedLstrip[matched] {
			for end > segStart && isSpaceByteASCII(text[end-1]) {
				end--
			}
		}
		flush(end)
		out = append(out, b.addedTokens[matched])
		i += len(matched)
		if b.addedRstrip[matched] {
			for i < len(text) && isSpaceByteASCII(text[i]) {
				i++
			}
		}
		segStart = i
	}
	flush(len(text))
	return out
}

func (b *bpeBackend) encodeSegment(text string) []int32 {
	// Normalize HERE, not in encode: HF's AddedVocabulary carves added-token
	// literals out of the RAW text and normalizes only what is left between them.
	if b.normNFC {
		text = norm.NFC.String(text)
	}
	var ids []int32
	var sc bpeScratch // reused across this segment's pieces
	for _, piece := range b.preTokenize(text) {
		m := sc.mapped[:0]
		for i := 0; i < len(piece); i++ { // map each RAW byte to its GPT-2 rune
			m = utf8.AppendRune(m, b.byte2rune[piece[i]])
		}
		sc.mapped = m
		// A string VIEW of the reused buffer: bpeInto only slices it and probes the
		// read-only rank/vocab maps (which never retain a key), and never keeps the
		// string past the call, so reusing m for the next piece is safe.
		mapped := unsafe.String(unsafe.SliceData(m), len(m))
		ids = b.bpeInto(mapped, &sc, ids)
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
		Lstrip  bool   `json:"lstrip"`
		Rstrip  bool   `json:"rstrip"`
	} `json:"added_tokens"`
	Normalizer json.RawMessage `json:"normalizer"`
	Model      struct {
		Type   string            `json:"type"`
		Vocab  map[string]int32  `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
	PostProcessor struct {
		Type string            `json:"type"`
		Cls  []json.RawMessage `json:"cls"` // RobertaProcessing: ["<s>", 0]
		Sep  []json.RawMessage `json:"sep"` // ...                ["</s>", 2]
		// TemplateProcessing: single is the one-sequence template, e.g.
		// [CLS] $A [SEP]; special_tokens maps each literal to its ids.
		Single        []map[string]templateElement `json:"single"`
		SpecialTokens map[string]struct {
			IDs []int32 `json:"ids"`
		} `json:"special_tokens"`
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

	normNFC, err := bpeNormalizer(raw.Normalizer)
	if err != nil {
		return nil, err
	}
	rank, err := parseBPEMerges(raw.Model.Merges, "bpe")
	if err != nil {
		return nil, err
	}

	added := make(map[string]int32, len(raw.AddedTokens))
	var lstrip, rstrip map[string]bool
	for _, at := range raw.AddedTokens {
		added[at.Content] = at.ID
		if at.Lstrip {
			if lstrip == nil {
				lstrip = map[string]bool{}
			}
			lstrip[at.Content] = true
		}
		if at.Rstrip {
			if rstrip == nil {
				rstrip = map[string]bool{}
			}
			rstrip[at.Content] = true
		}
	}
	addedKeys := sortedAddedKeys(added)

	var prefixIDs, suffixIDs []int32
	switch raw.PostProcessor.Type {
	case "", "ByteLevel":
		// A bare LM checkpoint: no wrap.
	case "RobertaProcessing":
		// Wraps <s> (cls) ++ body ++ </s> (sep).
		if prefixIDs, err = postProcID(raw.PostProcessor.Cls); err != nil {
			return nil, fmt.Errorf("bpe: post_processor cls: %w", err)
		}
		if suffixIDs, err = postProcID(raw.PostProcessor.Sep); err != nil {
			return nil, fmt.Errorf("bpe: post_processor sep: %w", err)
		}
	case "TemplateProcessing":
		// Shared with the Unigram and SentencePiece-BPE paths — the template shape
		// is the same regardless of which model type sits under it.
		prefixIDs, suffixIDs = templateSpecials(raw.PostProcessor.Single, raw.PostProcessor.SpecialTokens)
	default:
		// Not skipped: an unhandled post-processor yields no specials at all, and a
		// sequence silently missing its [CLS]/[SEP] mis-scores every downstream head.
		return nil, fmt.Errorf("bpe: unsupported post_processor.type %q", raw.PostProcessor.Type)
	}
	unkID := resolveUnkID(added, raw.Model.Vocab)

	return &bpeBackend{
		byte2rune:   bytesToUnicode(),
		vocab:       raw.Model.Vocab,
		rank:        rank,
		addedTokens: added,
		addedKeys:   addedKeys,
		addedLstrip: lstrip,
		addedRstrip: rstrip,
		unkID:       unkID,
		vocabSize:   len(raw.Model.Vocab),
		normNFC:     normNFC,
		prefixIDs:   prefixIDs,
		suffixIDs:   suffixIDs,
	}, nil
}

// bpeNormalizer reads tokenizer.json's normalizer for a byte-level BPE model and
// reports whether NFC must be applied. GPT-2 / RoBERTa declare none (null);
// OLMo/ModernBERT declare {"type":"NFC"}. Anything else is rejected rather than
// ignored: a normalizer that silently does not run diverges from the reference
// only on the inputs that need it most.
func bpeNormalizer(raw json.RawMessage) (bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return false, nil
	}
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false, fmt.Errorf("bpe: normalizer: %w", err)
	}
	switch probe.Type {
	case "NFC":
		return true, nil
	case "":
		return false, nil
	default:
		return false, fmt.Errorf("bpe: unsupported normalizer.type %q (NFC or none)", probe.Type)
	}
}

// parseBPEMerges ranks a BPE merge table, accepting BOTH spellings HF has shipped:
// the original space-joined string ("Ġ t") and the [a, b] pair array that
// `tokenizers` ≥0.20 writes. Rank is the position in the list, lower merging first.
// what names the caller for error messages.
func parseBPEMerges(merges []json.RawMessage, what string) (map[[2]string]int, error) {
	rank := make(map[[2]string]int, len(merges))
	for i, m := range merges {
		var pair [2]string
		if err := json.Unmarshal(m, &pair); err == nil {
			rank[pair] = i
			continue
		}
		var s string
		if err := json.Unmarshal(m, &s); err != nil {
			return nil, fmt.Errorf("%s: merge[%d]: neither a [a,b] pair nor a string", what, i)
		}
		parts := strings.SplitN(s, " ", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("%s: merge[%d] %q not a space-joined pair", what, i, s)
		}
		rank[[2]string{parts[0], parts[1]}] = i
	}
	return rank, nil
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

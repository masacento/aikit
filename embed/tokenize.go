package embed

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer is a pure-Go WordPiece tokenizer matching the HF tokenizers
// pipeline for the BERT-uncased family. For potion-code-16M specifically:
//
//   - BertNormalizer (clean_text + handle_chinese_chars + strip_accents + lowercase)
//   - BertPreTokenizer (whitespace + punctuation split)
//   - WordPiece (greedy longest-match, "##" continuation)
//
// Only the input → token-IDs path is implemented. No decoding, no offsets,
// no post-processing template wrapping (none is configured for this model).
type Tokenizer struct {
	vocab       map[string]int32 // string → id
	addedTokens map[string]int32 // [PAD], [UNK]
	addedKeys   []string         // addedTokens keys, sorted longest-first, then lex (carve-out scan order)
	// addedFirst[b] is true when some added key begins with byte b, and when
	// exactly one distinct first byte exists addedOne holds it. Both are the
	// carve-out scan's prefilter — see Encode (perf-campaign A2).
	addedFirst       [256]bool
	addedOne         byte
	addedSingle      bool
	unkID            int32
	continuingPrefix string // "##"
	maxCharsPerWord  int    // 100

	// BertNormalizer config (verified against potion-code-16M tokenizer.json)
	cleanText    bool
	handleCJK    bool
	stripAccents bool
	lowercase    bool

	// uni is set for Unigram/SentencePiece tokenizers (XLM-R family); when
	// non-nil, the public methods dispatch to it instead of the WordPiece path.
	// See tokenize_unigram.go.
	uni *unigramBackend

	// bpe is set for byte-level BPE tokenizers (GPT-2 / RoBERTa family); when
	// non-nil, the public methods dispatch to it. See tokenize_bpe.go.
	bpe *bpeBackend

	// spbpe is set for SentencePiece-style BPE tokenizers (Metaspace + byte-fallback;
	// ModernBERT / bekko family); when non-nil, the public methods dispatch to it.
	// See tokenize_sp_bpe.go.
	spbpe *spBPEBackend

	// wp memoizes wordPiece(word) → ids (the WordPiece path only; nil for uni/bpe).
	// Immutable vocab ⇒ pure function ⇒ safe to cache; ~98% of calls repeat.
	wp *wpCache
}

// tokenizer.json shape — we only parse the fields we need.
type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer struct {
		Type               string `json:"type"`
		CleanText          bool   `json:"clean_text"`
		HandleChineseChars bool   `json:"handle_chinese_chars"`
		StripAccents       *bool  `json:"strip_accents"` // nullable
		Lowercase          bool   `json:"lowercase"`
	} `json:"normalizer"`
	PreTokenizer struct {
		Type string `json:"type"`
	} `json:"pre_tokenizer"`
	Model struct {
		Type                    string           `json:"type"`
		UnkToken                string           `json:"unk_token"`
		ContinuingSubwordPrefix string           `json:"continuing_subword_prefix"`
		MaxInputCharsPerWord    int              `json:"max_input_chars_per_word"`
		Vocab                   map[string]int32 `json:"vocab"`
	} `json:"model"`
}

// LoadTokenizer parses an HF tokenizer.json file from disk.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}
	return parseTokenizer(data)
}

// LoadTokenizerFromFS parses an HF tokenizer.json file out of fsys at name.
// Same semantics as LoadTokenizer; takes an fs.FS for embed.FS / fstest.MapFS
// callers.
func LoadTokenizerFromFS(fsys fs.FS, name string) (*Tokenizer, error) {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}
	return parseTokenizer(data)
}

// sortedAddedKeys returns added's keys sorted longest-first, then
// lexicographically — the carve-out scan order an Encode's added-token match
// needs so overlapping literals resolve to the canonical greedy (longest)
// match, with a stable lex tiebreak for deterministic output. Shared by the
// WordPiece (parseTokenizer, below), byte-level BPE (tokenize_bpe.go), and
// Unigram (tokenize_unigram.go) loaders, which each build their own added
// map from the tokenizer.json AddedTokens list but need the identical scan
// order.
func sortedAddedKeys(added map[string]int32) []string {
	keys := make([]string, 0, len(added))
	for k := range added {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if len(keys[i]) != len(keys[j]) {
			return len(keys[i]) > len(keys[j])
		}
		return keys[i] < keys[j]
	})
	return keys
}

// parseTokenizer is the shared tokenizer.json parser used by LoadTokenizer
// and LoadTokenizerFromFS.
func parseTokenizer(data []byte) (*Tokenizer, error) {
	// Probe model.type / normalizer.type first — the Unigram vocab is an array of
	// [piece,score] pairs, which won't unmarshal into the WordPiece map below, so
	// dispatch must happen before the full WordPiece parse.
	var probe struct {
		Model      struct{ Type string } `json:"model"`
		Normalizer struct{ Type string } `json:"normalizer"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}
	// Unigram/SentencePiece tokenizers (XLM-R family) take a separate path; the
	// WordPiece parse below stays byte-identical for the BERT family.
	if isUnigramTokenizer(probe.Model.Type, probe.Normalizer.Type) {
		uni, err := parseUnigramTokenizer(data)
		if err != nil {
			return nil, err
		}
		return &Tokenizer{uni: uni, unkID: uni.unkID}, nil
	}
	// SentencePiece-style BPE (Metaspace + byte-fallback, e.g. ModernBERT / bekko):
	// model.type is "BPE" but the Metaspace pre-tokenizer distinguishes it from the
	// GPT-2 byte-level path below, which would otherwise mis-parse it.
	if probe.Model.Type == "BPE" && isMetaspaceBPETokenizer(data) {
		spbpe, err := parseSPBPETokenizer(data)
		if err != nil {
			return nil, err
		}
		return &Tokenizer{spbpe: spbpe, unkID: spbpe.unkID}, nil
	}
	// Byte-level BPE (GPT-2 / RoBERTa family, e.g. granite-embedding-english).
	if isBPETokenizer(probe.Model.Type) {
		bpe, err := parseBPETokenizer(data)
		if err != nil {
			return nil, err
		}
		return &Tokenizer{bpe: bpe, unkID: bpe.unkID}, nil
	}

	var raw tokenizerJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}
	if raw.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("unsupported model.type %q (expected WordPiece)", raw.Model.Type)
	}
	if raw.Normalizer.Type != "BertNormalizer" {
		return nil, fmt.Errorf("unsupported normalizer.type %q (expected BertNormalizer)", raw.Normalizer.Type)
	}
	if raw.PreTokenizer.Type != "BertPreTokenizer" {
		return nil, fmt.Errorf("unsupported pre_tokenizer.type %q (expected BertPreTokenizer)",
			raw.PreTokenizer.Type)
	}

	// HF BertNormalizer rule: if strip_accents is null and lowercase is true,
	// accents are stripped. Otherwise strip_accents is taken as the explicit value.
	stripAccents := raw.Normalizer.Lowercase
	if raw.Normalizer.StripAccents != nil {
		stripAccents = *raw.Normalizer.StripAccents
	}

	added := make(map[string]int32, len(raw.AddedTokens))
	for _, at := range raw.AddedTokens {
		added[at.Content] = at.ID
	}
	// Pre-sort the added-token literals longest-first so the Encode carve-out
	// (see Encode below) picks the canonical greedy match when literals
	// overlap as prefixes. Stable lex tiebreak keeps output deterministic.
	addedKeys := sortedAddedKeys(added)

	unkID, ok := raw.Model.Vocab[raw.Model.UnkToken]
	if !ok {
		// Fallback to added tokens
		unkID, ok = added[raw.Model.UnkToken]
		if !ok {
			return nil, fmt.Errorf("unk_token %q not found in vocab or added_tokens", raw.Model.UnkToken)
		}
	}

	prefix := raw.Model.ContinuingSubwordPrefix
	if prefix == "" {
		prefix = "##"
	}
	maxChars := raw.Model.MaxInputCharsPerWord
	if maxChars <= 0 {
		// 0 = unset → the WordPiece default; a negative value would make every
		// word exceed the cap and emit [UNK], silently zeroing the tokenizer.
		maxChars = 100
	}

	tk := &Tokenizer{
		vocab:            raw.Model.Vocab,
		addedTokens:      added,
		addedKeys:        addedKeys,
		unkID:            unkID,
		continuingPrefix: prefix,
		maxCharsPerWord:  maxChars,
		cleanText:        raw.Normalizer.CleanText,
		handleCJK:        raw.Normalizer.HandleChineseChars,
		stripAccents:     stripAccents,
		lowercase:        raw.Normalizer.Lowercase,
		wp:               newWPCache(),
	}
	// Prefilter for Encode's carve-out scan: which bytes can begin an added key,
	// and whether they are all the same byte. For every BERT-family tokenizer the
	// keys are [PAD], [UNK], [CLS], [SEP], [MASK] — one distinct first byte, so
	// the scan becomes strings.IndexByte.
	for _, k := range addedKeys {
		if k == "" {
			continue
		}
		tk.addedFirst[k[0]] = true
	}
	distinct := 0
	for b := range 256 {
		if tk.addedFirst[b] {
			distinct++
			tk.addedOne = byte(b)
		}
	}
	tk.addedSingle = distinct == 1
	return tk, nil
}

// Encode tokenizes a string to WordPiece IDs. No CLS/SEP wrapping (not
// configured for potion-code-16M).
//
// Added-token carve-out (HF AddedVocabulary semantics, matching this
// model's `normalized=false, single_word=false` flags): added-token
// literals are matched against the RAW text before normalization, in
// longest-first order; matches emit the added-token id atomically and
// non-matched regions run through normalize → pre-tokenize → wordpiece.
// This is the §3 Risk B rule — `[PAD]`/`[UNK]` appear literally in this
// repo (doc strings about the tokenizer) and the parity harness caught a
// per-word-only check missing it. Skip the carve loop when there are no
// added tokens for the small but real speedup on long inputs.
func (t *Tokenizer) Encode(text string) []int32 {
	if t.uni != nil {
		return t.uni.encode(text)
	}
	if t.bpe != nil {
		return t.bpe.encode(text)
	}
	if t.spbpe != nil {
		return t.spbpe.encode(text)
	}
	if len(t.addedKeys) == 0 {
		return t.encodeSegment(text)
	}
	if !utf8.ValidString(text) {
		return t.encodeAddedRebuild(text)
	}

	// The carve-out used to test every added key at every BYTE of the document
	// and rebuild the whole document through a strings.Builder on the way past.
	// addedKeys holds variable-length strings, so strings.HasPrefix cannot be
	// specialized into byte compares — it lowers to a runtime.memequal CALL, five
	// per byte — and the segments it built were already contiguous ranges of
	// `text`. Measured at 10.2% of an index run before this (perf-campaign A2,
	// docs/internal/perf-amdahl-linux-amd64.md §1).
	//
	// Scanning by BYTE rather than by rune is safe here and is why ValidString
	// gates the path. In valid UTF-8 an ASCII byte never appears inside a
	// multi-byte rune and a lead byte never appears as a continuation, so a byte
	// equal to some key's first byte is necessarily at a rune boundary — the only
	// place the rune-stepping original would have tried a match.
	var out []int32
	start, i := 0, 0
	for i < len(text) {
		// Find the next byte that could begin a key.
		if t.addedSingle {
			j := strings.IndexByte(text[i:], t.addedOne)
			if j < 0 {
				break
			}
			i += j
		} else {
			for i < len(text) && !t.addedFirst[text[i]] {
				i++
			}
			if i >= len(text) {
				break
			}
		}
		matched := ""
		for _, k := range t.addedKeys {
			if strings.HasPrefix(text[i:], k) {
				matched = k
				break
			}
		}
		if matched == "" {
			i++
			continue
		}
		if i > start {
			out = append(out, t.encodeSegment(text[start:i])...)
		}
		out = append(out, t.addedTokens[matched])
		i += len(matched)
		start = i
	}
	if start < len(text) {
		out = append(out, t.encodeSegment(text[start:])...)
	}
	return out
}

// encodeAddedRebuild is the pre-A2 carve-out, kept verbatim for invalid UTF-8.
//
// It is not equivalent to the sliced path there, and the difference is the point:
// DecodeRuneInString yields U+FFFD for a bad byte and WriteRune re-encodes it as
// the three-byte replacement character, where slicing preserves the raw byte.
// In practice encodeSegment → normalize → cleanText drops U+FFFD anyway, so the
// two agree — but cleanText is a config flag, and this way the agreement does not
// depend on it.
func (t *Tokenizer) encodeAddedRebuild(text string) []int32 {
	var (
		out []int32
		seg strings.Builder
	)
	flush := func() {
		if seg.Len() > 0 {
			out = append(out, t.encodeSegment(seg.String())...)
			seg.Reset()
		}
	}
	for i := 0; i < len(text); {
		matched := ""
		for _, k := range t.addedKeys {
			if strings.HasPrefix(text[i:], k) {
				matched = k
				break
			}
		}
		if matched != "" {
			flush()
			out = append(out, t.addedTokens[matched])
			i += len(matched)
			continue
		}
		r, size := utf8.DecodeRuneInString(text[i:])
		seg.WriteRune(r)
		i += size
	}
	flush()
	return out
}

// encodeSegment runs the BertNormalizer → BertPreTokenizer → WordPiece
// pipeline over a text fragment that already had its added-token literals
// carved out. The inner addedTokens check is defensive — after Encode's
// carve a normalized fragment can't contain `[PAD]`/`[UNK]` literally,
// but a future model with whitespace- or case-equivalent added tokens
// could trigger it.
func (t *Tokenizer) encodeSegment(text string) []int32 {
	normalized := t.normalize(text)
	words := t.preTokenize(normalized)
	var ids []int32
	for _, w := range words {
		if id, ok := t.addedTokens[w]; ok {
			ids = append(ids, id)
			continue
		}
		ids = append(ids, t.wordPiece(w)...)
	}
	return ids
}

// VocabSize reports the number of vocabulary entries.
func (t *Tokenizer) VocabSize() int {
	if t.uni != nil {
		return t.uni.vocabSize
	}
	if t.bpe != nil {
		return t.bpe.vocabSize
	}
	if t.spbpe != nil {
		return t.spbpe.vocabSize
	}
	return len(t.vocab)
}

// UnkID is the integer ID of the [UNK] token.
func (t *Tokenizer) UnkID() int32 { return t.unkID }

// SpecialID returns the ID of an added-token literal (e.g. "[CLS]",
// "[SEP]", "[PAD]"). Returns (0, false) if the literal isn't in the
// tokenizer's added_tokens table. Used by EncodeWithSpecials and by
// downstream models that need to wrap inputs with BERT-style specials.
func (t *Tokenizer) SpecialID(literal string) (int32, bool) {
	if t.uni != nil {
		id, ok := t.uni.addedTokens[literal]
		return id, ok
	}
	if t.bpe != nil {
		id, ok := t.bpe.addedTokens[literal]
		return id, ok
	}
	if t.spbpe != nil {
		id, ok := t.spbpe.addedTokens[literal]
		return id, ok
	}
	id, ok := t.addedTokens[literal]
	return id, ok
}

// TemplateSpecials reports the ids this tokenizer wraps a sequence with — what
// EncodeWithSpecials prepends and appends, typically [CLS] … [SEP] or <s> … </s>.
// Either may be empty (potion has no specials at all).
//
// It exists for heads that build their own sequence and so cannot call
// EncodeWithSpecials: GLiNER interleaves label markers with the words, so it needs
// the wrapping ids without the wrapping. SpecialID is not a substitute — these ids
// come from the model's bos/eos or its post-processor template, and for a
// SentencePiece model [CLS] is a CONTROL piece, deliberately absent from the
// added-token table that SpecialID searches (the literal text "[CLS]" must not
// tokenize to id 1).
func (t *Tokenizer) TemplateSpecials() (prefix, suffix []int32) {
	switch {
	case t.uni != nil:
		return t.uni.prefixIDs, t.uni.suffixIDs
	case t.bpe != nil:
		return t.bpe.prefixIDs, t.bpe.suffixIDs
	case t.spbpe != nil:
		return t.spbpe.prefixIDs, t.spbpe.suffixIDs
	}
	if cls, ok := t.addedTokens["[CLS]"]; ok {
		if sep, ok := t.addedTokens["[SEP]"]; ok {
			return []int32{cls}, []int32{sep}
		}
	}
	return nil, nil
}

// EncodeWithSpecials runs Encode and wraps the result as
//
//	[CLS] ++ Encode(text) ++ [SEP]
//
// truncating from the right if necessary so the full sequence is at
// most maxLen tokens (preserving the leading [CLS] and the trailing
// [SEP]). maxLen ≤ 2 yields exactly [CLS], [SEP].
//
// This is the canonical BERT input shape and the one CodeRankEmbed's
// reference (and tokenizer.json's TemplateProcessing post-processor)
// produces. The base Encode is unchanged so potion-code-16M parity
// stays byte-identical — this method is additive.
//
// Returns an error iff [CLS] or [SEP] is missing from the tokenizer's
// added_tokens. For tokenizers without those specials (potion), call
// Encode directly.
func (t *Tokenizer) EncodeWithSpecials(text string, maxLen int) ([]int32, error) {
	if t.uni != nil {
		return t.uni.encodeWithSpecials(text, maxLen), nil
	}
	if t.bpe != nil {
		return t.bpe.encodeWithSpecials(text, maxLen), nil
	}
	if t.spbpe != nil {
		return t.spbpe.encodeWithSpecials(text, maxLen), nil
	}
	cls, ok := t.addedTokens["[CLS]"]
	if !ok {
		return nil, fmt.Errorf("tokenizer: [CLS] missing from added_tokens")
	}
	sep, ok := t.addedTokens["[SEP]"]
	if !ok {
		return nil, fmt.Errorf("tokenizer: [SEP] missing from added_tokens")
	}
	if maxLen < 2 {
		return []int32{cls, sep}, nil
	}
	body := t.Encode(text)
	if len(body) > maxLen-2 {
		body = body[:maxLen-2]
	}
	out := make([]int32, 0, len(body)+2)
	out = append(out, cls)
	out = append(out, body...)
	out = append(out, sep)
	return out, nil
}

// ──────────────────────────────────────────────────────────────────────────────
// BertNormalizer
// ──────────────────────────────────────────────────────────────────────────────

func (t *Tokenizer) normalize(text string) string {
	if t.cleanText {
		text = cleanText(text)
	}
	if t.handleCJK {
		text = handleCJK(text)
	}
	if t.stripAccents {
		text = stripAccents(text)
	}
	if t.lowercase {
		// strings.ToLower applies Unicode-aware lowercasing. Critically: ToLower
		// leaves German ß unchanged (preserves it), matching HF Rust's
		// str::to_lowercase. Do NOT use strings.ToLower(strings.Map(...))
		// patterns that go through casefold — casefold maps ß → "ss".
		text = strings.ToLower(text)
	}
	return text
}

// cleanText drops NUL / U+FFFD / control chars and replaces whitespace
// with a regular space. Mirrors HF's BertNormalizer.clean_text — note
// the **order**: is_control is checked BEFORE is_whitespace because
// White_Space includes VT (\v) and FF (\f), which HF classifies as Cc
// control chars and drops rather than turning into spaces. (\t / \n /
// \r are exempted from is_control so they fall through to the
// whitespace replacement, matching HF.) The parity harness caught the
// previous swapped order on \v/\f inputs from this repo's own docs.
func cleanText(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		if r == 0 || r == 0xFFFD {
			continue
		}
		if isControl(r) {
			continue
		}
		if isWhitespace(r) {
			b.WriteByte(' ')
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// handleCJK wraps each CJK ideograph in spaces so they tokenize as
// individual tokens during pre-tokenization.
func handleCJK(text string) string {
	var b strings.Builder
	b.Grow(len(text) + 8)
	for _, r := range text {
		if isCJK(r) {
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// stripAccents: NFD-decompose, drop combining marks (Unicode category Mn).
// This handles café → cafe but preserves German ß (which has no NFD decomposition).
func stripAccents(text string) string {
	decomposed := norm.NFD.String(text)
	var b strings.Builder
	b.Grow(len(decomposed))
	for _, r := range decomposed {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// isWhitespace mirrors Rust's `char::is_whitespace` (Unicode White_Space
// property) — the predicate HF's BertNormalizer.clean_text uses. The
// earlier Zs-only check missed Zl (U+2028 LINE SEPARATOR) and Zp
// (U+2029 PARAGRAPH SEPARATOR), which the parity harness flagged on a
// real input from CLAUDE.md. White_Space covers \t \n \r and the
// separators in one go.
func isWhitespace(r rune) bool {
	return unicode.Is(unicode.White_Space, r)
}

// isControl per HF BERT: a control char that is NOT a whitespace
// (whitespace was already handled above).
func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.Is(unicode.Cc, r) || unicode.Is(unicode.Cf, r)
}

// isCJK: HF BERT's "Chinese character" predicate. Covers the major CJK
// Unified Ideograph ranges. Hiragana/katakana/hangul are NOT included
// (HF doesn't split those per-char).
func isCJK(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF:
		return true
	case r >= 0x3400 && r <= 0x4DBF:
		return true
	case r >= 0x20000 && r <= 0x2A6DF:
		return true
	case r >= 0x2A700 && r <= 0x2B73F:
		return true
	case r >= 0x2B740 && r <= 0x2B81F:
		return true
	case r >= 0x2B820 && r <= 0x2CEAF:
		return true
	case r >= 0xF900 && r <= 0xFAFF:
		return true
	case r >= 0x2F800 && r <= 0x2FA1F:
		return true
	}
	return false
}

// ──────────────────────────────────────────────────────────────────────────────
// BertPreTokenizer
// ──────────────────────────────────────────────────────────────────────────────

// preTokenize splits text on whitespace, then within each whitespace-bounded
// chunk further splits on punctuation (punctuation chars become their own
// tokens). Matches HF's BertPreTokenizer exactly.
//
// Every token it emits is a contiguous byte range of `text`, so it SLICES rather
// than rebuilding through a strings.Builder (perf-campaign A4). The rebuild cost
// two things: a Builder allocation per whitespace-bounded chunk, and — worse on
// punctuation-dense input like source code — `string(r)` per punctuation
// character, which escape analysis confirms heap-allocates. preTokenize is 11.9%
// of an index run (docs/internal/perf-amdahl-linux-amd64.md).
//
// THE VALIDITY GATE IS NOT PARANOIA. The two paths differ on invalid UTF-8, and
// only there: ranging a string yields U+FFFD for a bad byte, so `WriteRune`
// rebuilt it as the three-byte replacement character, while slicing preserves
// the raw byte. Every caller reaches this through normalize → cleanText, which
// drops U+FFFD outright, so the input is valid in practice — but `cleanText` is
// a config flag, and a tokenizer with it off would take a silently different
// path. One linear scan buys exactness for every input instead of for the
// configurations someone checked.
func (t *Tokenizer) preTokenize(text string) []string {
	if !utf8.ValidString(text) {
		return t.preTokenizeRebuild(text)
	}
	var out []string
	for part := range strings.FieldsSeq(text) {
		start := 0
		for i := 0; i < len(part); {
			r, w := utf8.DecodeRuneInString(part[i:])
			if isPunct(r) {
				if i > start {
					out = append(out, part[start:i])
				}
				out = append(out, part[i:i+w])
				start = i + w
			}
			i += w
		}
		if start < len(part) {
			out = append(out, part[start:])
		}
	}
	return out
}

// preTokenizeRebuild is the pre-A4 implementation, kept verbatim for invalid
// UTF-8 — where its U+FFFD rebuilding is the behaviour to preserve, not a bug to
// fix. TestPreTokenize_slicedMatchesRebuilt gates the two against each other.
func (t *Tokenizer) preTokenizeRebuild(text string) []string {
	var out []string
	for part := range strings.FieldsSeq(text) {
		var cur strings.Builder
		for _, r := range part {
			if isPunct(r) {
				if cur.Len() > 0 {
					out = append(out, cur.String())
					cur.Reset()
				}
				out = append(out, string(r))
			} else {
				cur.WriteRune(r)
			}
		}
		if cur.Len() > 0 {
			out = append(out, cur.String())
		}
	}
	return out
}

func isPunct(r rune) bool {
	if (r >= 33 && r <= 47) ||
		(r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) ||
		(r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// ──────────────────────────────────────────────────────────────────────────────
// WordPiece
// ──────────────────────────────────────────────────────────────────────────────

// wordPiece tokenizes a single pre-tokenized word into vocab IDs using
// greedy longest-match from the left, prefixing non-initial pieces with "##".
// Words longer than maxCharsPerWord (100) emit [UNK] directly.
// Words for which no prefix matches emit [UNK] for the whole word.
func (t *Tokenizer) wordPiece(word string) []int32 {
	if utf8.RuneCountInString(word) > t.maxCharsPerWord {
		return []int32{t.unkID}
	}
	chars := []rune(word)
	if len(chars) == 0 {
		return nil
	}
	// wordPiece is a pure function of (word, vocab), and the vocab is immutable after
	// LoadTokenizer — so identical words recompute identical ids. On real corpora ~98% of
	// wordPiece calls are repeats (and the repeats skew LONG: unique words average ~8.8 runes
	// vs ~2.7 overall, so the expensive quadratic probes are exactly the ones paid once). Cache
	// the result. The returned slice is READ-ONLY — encodeSegment copies it via append(…,…) and
	// no caller retains or mutates it, so sharing one backing array across callers is safe.
	if t.wp != nil {
		if ids, ok := t.wp.get(word); ok {
			return ids
		}
	}
	out := t.wordPieceCompute(chars)
	if t.wp != nil {
		t.wp.put(word, out)
	}
	return out
}

// wordPieceCompute is the uncached greedy longest-match — the exact prior wordPiece body,
// factored out so the memo can wrap it and the differential fuzz can compare the two paths.
func (t *Tokenizer) wordPieceCompute(chars []rune) []int32 {
	// Build each greedy-match candidate into a pooled []byte and look it up via
	// t.vocab[string(buf)] — the compiler elides that conversion for a map lookup,
	// so the probe is allocation-free. The old code allocated two strings per probe
	// (string(chars[start:end]) plus the "##" concat), ~40 short-lived allocations
	// for a 20-rune identifier, on the tokenizer that feeds every encoder entry
	// point (audit #11). Pooled so concurrent Encode stays safe (as bm25 does).
	bufp := wpBufPool.Get().(*[]byte)
	defer wpBufPool.Put(bufp)

	var out []int32
	start := 0
	for start < len(chars) {
		end := len(chars)
		matched := false
		for end > start {
			b := (*bufp)[:0]
			if start > 0 {
				b = append(b, t.continuingPrefix...)
			}
			for _, r := range chars[start:end] {
				b = utf8.AppendRune(b, r)
			}
			*bufp = b // retain the grown backing array for the next probe
			if id, ok := t.vocab[string(b)]; ok {
				out = append(out, id)
				start = end
				matched = true
				break
			}
			end--
		}
		if !matched {
			return []int32{t.unkID}
		}
	}
	return out
}

// wpBufPool holds reusable candidate-building buffers for wordPiece. Encode is
// goroutine-safe, so the scratch must be per-call — a sync.Pool gives that without
// a per-call allocation.
var wpBufPool = sync.Pool{New: func() any { b := make([]byte, 0, 128); return &b }}

// wpCache memoizes wordPiece results. Encode is goroutine-safe and the access pattern is a
// SHARED read-mostly set (all workers hit the same common words), so this is a sharded RWMutex
// map — not sync.Map, which is tuned for disjoint per-goroutine key sets. Sharding keeps read
// contention flat as workers scale; the FNV-1a shard hash is cheap next to the map probe it
// guards. Bounded per shard so adversarial/multilingual input can't grow it without limit — a
// word past the bound simply recomputes (correct, just uncached), which is the miss cost anyway.
const (
	wpShards      = 32   // power of two → shard index is a mask
	wpCapPerShard = 8192 // ~262k words total; natural unique-word sets converge near vocab size
)

type wpCache struct {
	shards [wpShards]struct {
		mu sync.RWMutex
		m  map[string][]int32
	}
}

func newWPCache() *wpCache {
	c := &wpCache{}
	for i := range c.shards {
		c.shards[i].m = make(map[string][]int32)
	}
	return c
}

// wpShard is FNV-1a over the word's bytes → shard index.
func wpShard(word string) uint32 {
	h := uint32(2166136261)
	for i := 0; i < len(word); i++ {
		h = (h ^ uint32(word[i])) * 16777619
	}
	return h & (wpShards - 1)
}

func (c *wpCache) get(word string) ([]int32, bool) {
	s := &c.shards[wpShard(word)]
	s.mu.RLock()
	ids, ok := s.m[word]
	s.mu.RUnlock()
	return ids, ok
}

// put stores ids for word (the caller must not mutate ids afterward). Skips storing past the
// per-shard cap; the missed word recomputes next time, so the cache stays bounded and correct.
func (c *wpCache) put(word string, ids []int32) {
	s := &c.shards[wpShard(word)]
	s.mu.Lock()
	if len(s.m) < wpCapPerShard {
		// Clone the retained key. Since A4, `word` is a VIEW into the normalized
		// text of whichever chunk was being tokenized, so storing it directly
		// would pin that whole string for the cache's lifetime — up to 8192
		// entries per shard each holding a ~1.5 KB chunk alive. The same hazard
		// bm25.Build documents, arriving here the moment preTokenize stopped
		// materializing its tokens. One clone per newly-cached word (9,463 for
		// aikit's own tree) against a cache that exists to avoid recomputation.
		s.m[strings.Clone(word)] = ids
	}
	s.mu.Unlock()
}

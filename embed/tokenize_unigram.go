package embed

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// ──────────────────────────────────────────────────────────────────────────────
// Precompiled normalizer (SentencePiece charsmap)
// ──────────────────────────────────────────────────────────────────────────────
//
// XLM-R / bge-m3 / multilingual-e5 normalize with SentencePiece's "Precompiled"
// normalizer: a `precompiled_charsmap` blob holding a darts-clone double-array
// trie plus a pool of null-terminated normalized strings. Normalization iterates
// grapheme clusters (base + combining marks); each cluster (or, past 6 bytes, each
// char) is looked up via the trie's shortest-prefix match and replaced with its
// pool string (which may be empty — that's how control chars are deleted), or
// passed through unchanged when no rule matches. This mirrors HF spm_precompiled's
// normalize_string / transform exactly (see firstPrefix / normalize below) and
// reproduces its output byte-for-byte (verified in the norm oracle test against a
// per-codepoint sweep over U+0000..U+2FFFF plus combining sequences).
//
// Blob layout (little-endian): [u32 trieByteSize][trie u32 units][normalized pool].

type precompiled struct {
	array []uint32 // darts-clone double-array trie units
	pool  []byte   // normalized-string pool (null-terminated entries)
}

// newPrecompiled decodes a base64 precompiled_charsmap (as stored in
// tokenizer.json) into its trie + pool.
func newPrecompiled(b64 string) (*precompiled, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Errorf("precompiled_charsmap: base64: %w", err)
	}
	return newPrecompiledBytes(raw)
}

// newPrecompiledBytes is newPrecompiled over an already-decoded blob. A raw
// spm.model stores the same payload verbatim in NormalizerSpec.precompiled_charsmap
// with no base64 layer (see tokenize_spm.go), so both entry points share the parse.
func newPrecompiledBytes(raw []byte) (*precompiled, error) {
	if len(raw) < 4 {
		return nil, fmt.Errorf("precompiled_charsmap: too short (%d bytes)", len(raw))
	}
	trieBytes := int(binary.LittleEndian.Uint32(raw[:4]))
	if trieBytes%4 != 0 || 4+trieBytes > len(raw) {
		return nil, fmt.Errorf("precompiled_charsmap: bad trie size %d (total %d)", trieBytes, len(raw))
	}
	units := trieBytes / 4
	array := make([]uint32, units)
	off := 4
	for i := range units {
		array[i] = binary.LittleEndian.Uint32(raw[off : off+4])
		off += 4
	}
	return &precompiled{array: array, pool: raw[4+trieBytes:]}, nil
}

// darts-clone unit accessors (bit layout shared with SentencePiece / HF spm_precompiled).
func dartsHasLeaf(u uint32) bool { return (u>>8)&1 == 1 }
func dartsValue(u uint32) uint32 { return u & 0x7fffffff }
func dartsLabel(u uint32) uint32 { return u & (1<<31 | 0xff) }
func dartsOffset(u uint32) uint32 {
	return (u >> 10) << ((u & (1 << 9)) >> 6)
}

// firstPrefix walks the trie for key and returns the value of the FIRST (shortest)
// key that is a prefix of key. SentencePiece/HF `transform` deliberately takes
// results[0] — the shortest prefix match, not the longest ("Yes, this seems
// broken. No, I don't know why Google did this." — spm_precompiled). We reproduce
// that exactly. ok is false when no trie key prefixes key.
func (p *precompiled) firstPrefix(key []byte) (val uint32, ok bool) {
	if len(p.array) == 0 {
		return 0, false
	}
	nodePos := int(dartsOffset(p.array[0]))
	for i := range key {
		c := key[i]
		if c == 0 {
			break
		}
		nodePos ^= int(c)
		if nodePos < 0 || nodePos >= len(p.array) {
			return 0, false
		}
		unit := p.array[nodePos]
		if dartsLabel(unit) != uint32(c) {
			return 0, false
		}
		nodePos ^= int(dartsOffset(unit))
		if dartsHasLeaf(unit) {
			if nodePos < 0 || nodePos >= len(p.array) {
				return 0, false
			}
			return dartsValue(p.array[nodePos]), true
		}
	}
	return 0, false
}

// transform is HF spm_precompiled's transform: the shortest-prefix trie match's
// normalized replacement (which may be empty), or ("", false) if the chunk has no
// matching prefix. Note it replaces the WHOLE chunk with the (possibly shorter)
// match's value — bytes past the matched prefix are dropped, matching HF.
func (p *precompiled) transform(chunk []byte) (string, bool) {
	idx, ok := p.firstPrefix(chunk)
	if !ok {
		return "", false
	}
	if int(idx) >= len(p.pool) {
		return "", true
	}
	s := p.pool[idx:]
	if end := indexZero(s); end >= 0 {
		return string(s[:end]), true
	}
	return string(s), true
}

func indexZero(b []byte) int {
	for i, c := range b {
		if c == 0 {
			return i
		}
	}
	return -1
}

// isCombining reports whether r is a combining mark (Unicode Mn/Mc/Me). We use it
// to group a base scalar with its following marks into one cluster — the only
// multi-codepoint keys a SentencePiece NFKC charsmap holds are combining
// sequences, so this reproduces HF's UAX-29 grapheme walk for normalization
// without a segmentation dependency (verified byte-exact against the oracle).
func isCombining(r rune) bool {
	return unicode.In(r, unicode.Mn, unicode.Mc, unicode.Me)
}

// normalize reproduces HF spm_precompiled's normalize_string: iterate grapheme
// clusters (base + following combining marks); for a cluster under 6 bytes try to
// transform it whole, else transform each char and pass non-matching chars
// through unchanged.
func (p *precompiled) normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		// Cluster = base rune + trailing combining marks.
		_, sz := utf8.DecodeRuneInString(s[i:])
		if sz == 0 {
			sz = 1
		}
		end := i + sz
		for end < len(s) {
			r, sz2 := utf8.DecodeRuneInString(s[end:])
			if !isCombining(r) {
				break
			}
			end += sz2
		}
		g := s[i:end]
		i = end

		if len(g) < 6 {
			if norm, ok := p.transform([]byte(g)); ok {
				b.WriteString(norm)
				continue
			}
		}
		for j := 0; j < len(g); {
			_, sz3 := utf8.DecodeRuneInString(g[j:])
			if sz3 == 0 {
				sz3 = 1
			}
			part := g[j : j+sz3]
			if norm, ok := p.transform([]byte(part)); ok {
				b.WriteString(norm)
			} else {
				b.WriteString(part)
			}
			j += sz3
		}
	}
	return b.String()
}

// ──────────────────────────────────────────────────────────────────────────────
// Unigram model (Viterbi)
// ──────────────────────────────────────────────────────────────────────────────
//
// Reproduces HF tokenizers' Unigram encode_optimized (which itself mirrors
// SentencePiece unigram_model.cc): a byte-position Viterbi over the vocab where
// each piece contributes its log-prob, unknown single chars cost
// min_score - K_UNK_PENALTY (10.0), and — with fuse_unk — consecutive unknown
// pieces collapse to one <unk>. byte_fallback is not used by XLM-R.

const unigramUnkPenalty = 10.0 // K_UNK_PENALTY in HF/SentencePiece

type unigram struct {
	piece2id map[string]int32
	scores   []float64 // score by id (vocab order)
	unkID    int32
	minScore float64
	maxBytes int // longest vocab piece in bytes — bounds the DP inner scan
	fuseUnk  bool
	// byteFallback decomposes a character absent from the vocab into its UTF-8
	// bytes as "<0xNN>" tokens (cl-nagoya/ruri-v3-*) instead of a single <unk>.
	// byteIDs maps a raw byte to its "<0xNN>" token id (only valid when
	// byteFallback is true).
	byteFallback bool
	byteIDs      [256]int32
}

// viterbiIDs segments sentence (already normalized + metaspace-prefixed) into the
// max-log-prob path of vocab ids, matching HF encode_optimized exactly.
func (u *unigram) viterbiIDs(sentence string) []int32 {
	size := len(sentence)
	if size == 0 {
		return nil
	}
	unkScore := u.minScore - unigramUnkPenalty

	type node struct {
		id      int32
		score   float64
		startAt int
		reached bool
	}
	best := make([]node, size+1)
	best[0] = node{reached: true}

	for startAt := 0; startAt < size; {
		till := best[startAt].score
		reachedHere := best[startAt].reached
		_, mblen := utf8.DecodeRuneInString(sentence[startAt:])
		if mblen == 0 {
			mblen = 1
		}
		hasSingle := false
		if reachedHere {
			maxEnd := min(startAt+u.maxBytes, size)
			for end := startAt + 1; end <= maxEnd; end++ {
				id, ok := u.piece2id[sentence[startAt:end]]
				if !ok {
					continue
				}
				cand := u.scores[id] + till
				tn := &best[end]
				if !tn.reached || cand > tn.score {
					tn.reached = true
					tn.score = cand
					tn.startAt = startAt
					tn.id = id
				}
				if !hasSingle && end-startAt == mblen {
					hasSingle = true
				}
			}
			if !hasSingle {
				if u.byteFallback {
					// The character has no vocab piece of its own: decompose its UTF-8
					// bytes [startAt, startAt+mblen) into a chain of "<0xNN>" byte-token
					// nodes, each scored by its vocab log-prob (SentencePiece's
					// byte_fallback). Continuation-byte positions can't begin a vocab
					// piece (pieces are valid UTF-8), so they're only ever reached through
					// this chain — the outer loop steps by rune and never expands them,
					// which is exactly right.
					var cum float64
					prev := startAt
					for bi := startAt; bi < startAt+mblen; bi++ {
						id := u.byteIDs[sentence[bi]]
						cum += u.scores[id]
						tn := &best[bi+1]
						if cand := cum + till; !tn.reached || cand > tn.score {
							tn.reached = true
							tn.score = cand
							tn.startAt = prev
							tn.id = id
						}
						prev = bi + 1
					}
				} else {
					tn := &best[startAt+mblen]
					cand := unkScore + till
					if !tn.reached || cand > tn.score {
						tn.reached = true
						tn.score = cand
						tn.startAt = startAt
						tn.id = u.unkID
					}
				}
			}
		}
		startAt += mblen
	}

	// Backtrack, then (fuse_unk) collapse runs of <unk> into one.
	var rev []int32
	for endsAt := size; endsAt > 0; {
		n := best[endsAt]
		if !n.reached { // defensive: unreachable tail (shouldn't happen)
			break
		}
		rev = append(rev, n.id)
		endsAt = n.startAt
	}
	ids := make([]int32, 0, len(rev))
	for i := len(rev) - 1; i >= 0; i-- {
		if u.fuseUnk && rev[i] == u.unkID && len(ids) > 0 && ids[len(ids)-1] == u.unkID {
			continue // fuse consecutive <unk>
		}
		ids = append(ids, rev[i])
	}
	return ids
}

// ──────────────────────────────────────────────────────────────────────────────
// Metaspace pre-tokenizer (WhitespaceSplit + ▁)
// ──────────────────────────────────────────────────────────────────────────────

const metaspace = '▁' // ▁ — SentencePiece's space marker

// metaspaceKind selects which Metaspace pre-tokenizer variant the Unigram backend
// reproduces. The three differ only in how the normalized text is mapped into
// ▁-marked chunks before the Viterbi pass; the Unigram model itself is shared.
type metaspaceKind int

const (
	// metaspaceKindPrepend: a BARE Metaspace(add_prefix_space=true / prepend_scheme
	// "always") — replace every ASCII space with ▁ and prepend a leading ▁, then
	// split so each chunk begins with ▁ (bge-m3). The default.
	metaspaceKindPrepend metaspaceKind = iota
	// metaspaceKindWhitespaceSplit: Sequence[WhitespaceSplit, Metaspace(add_prefix_space
	// =true)] — split on Unicode whitespace (dropping it) and prepend ▁ to each
	// piece (XLM-R / e5).
	metaspaceKindWhitespaceSplit
	// metaspaceKindNever: Metaspace(prepend_scheme="never", split=false) — replace
	// every ASCII space with ▁, prepend nothing, and return the whole text as ONE
	// chunk; the Unigram Viterbi does the splitting (cl-nagoya/ruri-v3-*).
	metaspaceKindNever
	// metaspaceKindSPM: sentencepiece's OWN normalization tail, used when the
	// tokenizer was loaded from a raw spm.model rather than a tokenizer.json
	// (mDeBERTa-v3 / GLiNER). It is neither of the two above: like Never it emits ONE
	// chunk (so a vocab piece may span what used to be a space — spm encodes the whole
	// normalized string in a single Viterbi pass), but like Prepend it inserts a
	// leading ▁ (add_dummy_prefix). It additionally applies remove_extra_whitespaces.
	// Both flags are read from the model's NormalizerSpec into spmAddDummyPrefix /
	// spmRemoveExtraWS.
	metaspaceKindSPM
)

// metaspaceSplit reproduces the XLM-R / e5 pre_tokenizer Sequence[WhitespaceSplit,
// Metaspace(add_prefix_space=true)]: split the normalized text on Unicode
// whitespace (dropping it), then prepend ▁ to each resulting piece.
func metaspaceSplit(text string) []string {
	var out []string
	for _, field := range strings.FieldsFunc(text, unicode.IsSpace) {
		out = append(out, string(metaspace)+field)
	}
	return out
}

// metaspaceBare reproduces a BARE Metaspace(add_prefix_space=true) pre_tokenizer
// (bge-m3): replace every ASCII space with ▁, prepend a leading ▁ unless the text
// already begins with one, then split so each piece begins with ▁ (SentencePiece's
// MergedWithNext). Unlike metaspaceSplit it does NOT drop whitespace, so a lone ▁
// survives (e.g. a trailing space) — matching HF exactly (bge-m3 collapses runs of
// spaces in a preceding Replace normalizer, not here). Empty input yields nothing.
func metaspaceBare(text string) []string {
	if text == "" {
		return nil
	}
	var b strings.Builder
	b.Grow(len(text) + 3)
	for _, r := range text {
		if r == ' ' {
			b.WriteRune(metaspace)
		} else {
			b.WriteRune(r)
		}
	}
	s := b.String()
	if !strings.HasPrefix(s, string(metaspace)) {
		s = string(metaspace) + s
	}
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, sz := utf8.DecodeRuneInString(s[i:])
		if r == metaspace && i > start {
			out = append(out, s[start:i])
			start = i
		}
		i += sz
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

// metaspaceNever reproduces a Metaspace(prepend_scheme="never", split=false)
// pre-tokenizer (cl-nagoya/ruri-v3-*): replace every ASCII space with ▁, prepend
// NOTHING (so a leading word keeps no ▁), and return the whole text as a single
// chunk — the Unigram Viterbi segments it, splitting before each ▁. Only U+0020 is
// replaced (a tab etc. passes through), matching HF's Metaspace " " pattern. Empty
// input yields nothing.
func metaspaceNever(text string) []string {
	if text == "" {
		return nil
	}
	return []string{strings.ReplaceAll(text, " ", string(metaspace))}
}

// ──────────────────────────────────────────────────────────────────────────────
// Unigram tokenizer backend
// ──────────────────────────────────────────────────────────────────────────────

// unigramBackend is the SentencePiece/Unigram tokenizer (XLM-R family): the
// Precompiled normalizer, the Metaspace pre-tokenizer, the Unigram Viterbi model,
// added-token carve-out, and the TemplateProcessing specials. It plugs into the
// public embed.Tokenizer, which dispatches to it when tokenizer.json is Unigram.
type unigramBackend struct {
	norm        *precompiled  // nil = identity normalizer (no Precompiled charsmap; cl-nagoya/ruri-v3-*)
	replaces    []replaceRule // Replace normalizers applied after the charsmap (Sequence tail)
	nfc         bool          // NFC normalizer step (gliner2's mDeBERTa tokenizer.json)
	stripRight  bool          // Strip(strip_right=true) normalizer step (gliner2's mDeBERTa tokenizer.json)
	metaspace   metaspaceKind // Metaspace pre-tokenizer variant (see metaspaceKind)
	model       *unigram
	addedTokens map[string]int32
	addedKeys   []string // longest-first carve-out scan order
	unkID       int32
	vocabSize   int
	prefixIDs   []int32 // TemplateProcessing single: specials before the sequence (<s>)
	suffixIDs   []int32 // ... and after (</s>)

	// spmAddDummyPrefix / spmRemoveExtraWS drive metaspaceKindSPM; both come from
	// the spm.model NormalizerSpec and are ignored by the other Metaspace variants.
	spmAddDummyPrefix bool
	spmRemoveExtraWS  bool
}

// replaceRule is a Replace normalizer (regex pattern → literal content), e.g.
// bge-m3's " {2,}" → " " that collapses runs of spaces.
type replaceRule struct {
	re      *regexp.Regexp
	content string
}

// normalizeText runs the normalizer pipeline: the Precompiled charsmap (when
// present), then any Replace rules, in Sequence order. A nil norm is the identity
// normalizer — tokenizer.json normalizer:null (cl-nagoya/ruri-v3-*), which applies
// no normalization at all.
func (u *unigramBackend) normalizeText(text string) string {
	s := text
	if u.norm != nil {
		s = u.norm.normalize(s)
	}
	for _, r := range u.replaces {
		s = r.re.ReplaceAllString(s, r.content)
	}
	if u.nfc {
		s = norm.NFC.String(s)
	}
	if u.stripRight {
		s = strings.TrimRightFunc(s, unicode.IsSpace)
	}
	return s
}

// preTokenize dispatches to the configured Metaspace variant.
func (u *unigramBackend) preTokenize(text string) []string {
	switch u.metaspace {
	case metaspaceKindWhitespaceSplit:
		return metaspaceSplit(text)
	case metaspaceKindNever:
		return metaspaceNever(text)
	case metaspaceKindSPM:
		return u.metaspaceSPM(text)
	default:
		return metaspaceBare(text)
	}
}

// metaspaceSPM reproduces the tail of sentencepiece's Normalizer::Normalize that
// runs after the charsmap: optionally collapse runs of spaces and trim the ends
// (remove_extra_whitespaces), optionally insert a leading space (add_dummy_prefix),
// then escape every space to ▁. The result is returned as a SINGLE chunk because
// that is what spm does — one Viterbi pass over the whole normalized string.
//
// Two things follow, and only the second is observable for mDeBERTa-v3. A single
// chunk permits a vocab piece to span what used to be a space (this vocab happens
// to contain no piece with an internal ▁, so that half is inert here). It also
// means whitespace the charsmap does NOT fold to U+0020 stays in the text instead
// of being dropped — U+000B and U+0085 are the live cases, and they are why the
// whitespace-split variant is not a legal substitute (see the break-it-first gate
// in tokenize_spm_test.go).
//
// Empty input (or input that is all whitespace with remove_extra_whitespaces on)
// yields nothing: spm short-circuits an empty normalized string rather than
// emitting a bare ▁ from the dummy prefix.
func (u *unigramBackend) metaspaceSPM(text string) []string {
	s := text
	if u.spmRemoveExtraWS {
		var b strings.Builder
		b.Grow(len(s))
		prevSpace := true // leading run is dropped, same as an interior collapse
		for _, r := range s {
			if r == ' ' {
				if prevSpace {
					continue
				}
				prevSpace = true
				b.WriteRune(' ')
				continue
			}
			prevSpace = false
			b.WriteRune(r)
		}
		s = strings.TrimSuffix(b.String(), " ")
	}
	if s == "" {
		return nil
	}
	if u.spmAddDummyPrefix {
		s = " " + s
	}
	return []string{strings.ReplaceAll(s, " ", string(metaspace))}
}

// encode runs normalize → metaspace pre-tokenize → Unigram over the added-token
// carve-out, producing the BARE id sequence (no template specials).
func (u *unigramBackend) encode(text string) []int32 {
	if len(u.addedKeys) == 0 {
		return u.encodeSegment(text)
	}
	var out []int32
	var seg strings.Builder
	flush := func() {
		if seg.Len() > 0 {
			out = append(out, u.encodeSegment(seg.String())...)
			seg.Reset()
		}
	}
	for i := 0; i < len(text); {
		matched := ""
		for _, k := range u.addedKeys {
			if strings.HasPrefix(text[i:], k) {
				matched = k
				break
			}
		}
		if matched != "" {
			flush()
			out = append(out, u.addedTokens[matched])
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

func (u *unigramBackend) encodeSegment(text string) []int32 {
	normalized := u.normalizeText(text)
	var ids []int32
	for _, piece := range u.preTokenize(normalized) {
		ids = append(ids, u.model.viterbiIDs(piece)...)
	}
	return ids
}

// encodeWithSpecials wraps encode with the TemplateProcessing specials
// (prefixIDs ++ body ++ suffixIDs), truncating body from the right so the total
// is at most maxLen. For XLM-R this is <s> ++ body ++ </s>.
func (u *unigramBackend) encodeWithSpecials(text string, maxLen int) []int32 {
	fixed := len(u.prefixIDs) + len(u.suffixIDs)
	if maxLen < fixed {
		maxLen = fixed
	}
	body := u.encode(text)
	if len(body) > maxLen-fixed {
		body = body[:maxLen-fixed]
	}
	out := make([]int32, 0, len(body)+fixed)
	out = append(out, u.prefixIDs...)
	out = append(out, body...)
	out = append(out, u.suffixIDs...)
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// tokenizer.json → unigramBackend
// ──────────────────────────────────────────────────────────────────────────────

// unigramJSON is the tokenizer.json shape for a Unigram/SentencePiece tokenizer
// (the fields the Precompiled + Metaspace + Unigram + TemplateProcessing pipeline
// needs). Vocab is a list of [piece, score] pairs; entry index is the token id.
type unigramJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type         string            `json:"type"`
		UnkID        *int32            `json:"unk_id"`
		ByteFallback bool              `json:"byte_fallback"`
		Vocab        []json.RawMessage `json:"vocab"`
	} `json:"model"`
	PostProcessor struct {
		Type          string                       `json:"type"`
		Single        []map[string]templateElement `json:"single"`
		SpecialTokens map[string]struct {
			IDs []int32 `json:"ids"`
		} `json:"special_tokens"`
	} `json:"post_processor"`
}

type templateElement struct {
	ID string `json:"id"`
}

// isUnigramTokenizer peeks at model.type / normalizer.type to decide whether the
// tokenizer.json is a Unigram/SentencePiece one (handled here) vs WordPiece
// (handled by the base parser). XLM-R omits model.type, so the Precompiled
// normalizer is the reliable tell.
func isUnigramTokenizer(model, normalizer string) bool {
	return model == "Unigram" || (model == "" && normalizer == "Precompiled")
}

// decodeUnigramJSON returns the tokenizer.json fields the Unigram backend needs,
// with the vocab already split into pieces and scores (index = token id).
//
// The fast scanner runs first; it handles every tokenizer.json shape seen in the
// wild and is ~4× the speed of a whole-document unmarshal on a 100k-entry vocab
// (see tokenize_unigram_scan.go). Anything it does not recognise falls through to
// the generic decoder below, which stays the source of truth for error messages.
func decodeUnigramJSON(data []byte) (*unigramJSON, []string, []float64, error) {
	if raw, pieces, scores, ok := scanUnigramJSON(data); ok {
		return raw, pieces, scores, nil
	}
	var raw unigramJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, nil, nil, fmt.Errorf("parse tokenizer.json (unigram): %w", err)
	}
	// Vocab: [piece, score] pairs; index is the id.
	pieces := make([]string, len(raw.Model.Vocab))
	scores := make([]float64, len(raw.Model.Vocab))
	for i, entry := range raw.Model.Vocab {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 {
			return nil, nil, nil, fmt.Errorf("unigram: vocab[%d] not a [piece,score] pair", i)
		}
		if err := json.Unmarshal(pair[0], &pieces[i]); err != nil {
			return nil, nil, nil, fmt.Errorf("unigram: vocab[%d] piece: %w", i, err)
		}
		if err := json.Unmarshal(pair[1], &scores[i]); err != nil {
			return nil, nil, nil, fmt.Errorf("unigram: vocab[%d] score: %w", i, err)
		}
	}
	return &raw, pieces, scores, nil
}

// parseUnigramTokenizer builds a unigramBackend from tokenizer.json bytes.
func parseUnigramTokenizer(data []byte) (*unigramBackend, error) {
	raw, pieces, scores, err := decodeUnigramJSON(data)
	if err != nil {
		return nil, err
	}
	if raw.Model.UnkID == nil {
		return nil, fmt.Errorf("unigram: model.unk_id missing")
	}
	charsmap, replaces, nfc, stripRight, err := buildNormalizer(raw.Normalizer)
	if err != nil {
		return nil, err
	}
	// A null/absent normalizer (cl-nagoya/ruri-v3-*) is the identity: there is no
	// Precompiled charsmap, so norm stays nil and normalizeText passes text through
	// unchanged. NFC / Strip steps (gliner2's mDeBERTa tokenizer.json) may be the
	// only steps present, which is equally legal.
	var normPC *precompiled
	if charsmap != "" {
		if normPC, err = newPrecompiled(charsmap); err != nil {
			return nil, err
		}
	}
	metaspace, err := parseMetaspaceKind(raw.PreTokenizer)
	if err != nil {
		return nil, err
	}

	n := len(pieces)
	piece2id := make(map[string]int32, n)
	minScore := 0.0
	maxBytes := 1
	for i, piece := range pieces {
		piece2id[piece] = int32(i)
		if scores[i] < minScore {
			minScore = scores[i]
		}
		if len(piece) > maxBytes {
			maxBytes = len(piece)
		}
	}

	// byte_fallback (cl-nagoya/ruri-v3-*): an unknown character decomposes into its
	// UTF-8 bytes as "<0xNN>" (uppercase hex) tokens. Map each raw byte to its token
	// id up front; the Viterbi fallback path indexes this directly. All 256 byte
	// tokens must be present when the flag is on.
	var byteIDs [256]int32
	if raw.Model.ByteFallback {
		for b := range 256 {
			piece := fmt.Sprintf("<0x%02X>", b)
			id, ok := piece2id[piece]
			if !ok {
				return nil, fmt.Errorf("unigram: byte_fallback set but vocab lacks %q", piece)
			}
			byteIDs[b] = id
		}
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

	prefixIDs, suffixIDs := templateSpecials(raw.PostProcessor.Single, raw.PostProcessor.SpecialTokens)

	return &unigramBackend{
		norm:       normPC,
		replaces:   replaces,
		nfc:        nfc,
		stripRight: stripRight,
		metaspace:  metaspace,
		model: &unigram{
			piece2id:     piece2id,
			scores:       scores,
			unkID:        *raw.Model.UnkID,
			minScore:     minScore,
			maxBytes:     maxBytes,
			fuseUnk:      true,
			byteFallback: raw.Model.ByteFallback,
			byteIDs:      byteIDs,
		},
		addedTokens: added,
		addedKeys:   addedKeys,
		unkID:       *raw.Model.UnkID,
		vocabSize:   n,
		prefixIDs:   prefixIDs,
		suffixIDs:   suffixIDs,
	}, nil
}

// buildNormalizer walks a tokenizer.json normalizer — a bare Precompiled, or a
// Sequence of the SentencePiece-style steps the multilingual embedders use:
// Precompiled, Replace (regex → literal), NFC, and Strip (right/whitespace only,
// as configured by gliner2's mDeBERTa tokenizer.json). It errors on any other type
// so an unsupported normalizer fails loudly (→ best-effort nil in the loader)
// rather than silently mis-normalizing. A null/absent normalizer
// (cl-nagoya/ruri-v3-*) is the identity: it returns an empty charsmap (→ nil
// precompiled) and no rules.
func buildNormalizer(raw json.RawMessage) (charsmap string, replaces []replaceRule, nfc, stripRight bool, err error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil, false, false, nil
	}
	var n struct {
		Type                string            `json:"type"`
		PrecompiledCharsmap string            `json:"precompiled_charsmap"`
		Normalizers         []json.RawMessage `json:"normalizers"`
		Pattern             struct {
			Regex  string `json:"Regex"`
			String string `json:"String"`
		} `json:"pattern"`
		Content    string `json:"content"`
		StripLeft  bool   `json:"strip_left"`
		StripRight bool   `json:"strip_right"`
	}
	if err = json.Unmarshal(raw, &n); err != nil {
		return "", nil, false, false, fmt.Errorf("unigram: normalizer: %w", err)
	}
	switch n.Type {
	case "Precompiled":
		return n.PrecompiledCharsmap, nil, false, false, nil
	case "Replace":
		pat := n.Pattern.Regex
		if pat == "" { // a literal-string Replace: match it verbatim
			pat = regexp.QuoteMeta(n.Pattern.String)
		}
		re, cerr := regexp.Compile(pat)
		if cerr != nil {
			return "", nil, false, false, fmt.Errorf("unigram: Replace pattern %q: %w", pat, cerr)
		}
		return "", []replaceRule{{re: re, content: n.Content}}, false, false, nil
	case "NFC":
		return "", nil, true, false, nil
	case "Strip":
		// Only right-strip is wired through: no checkpoint this package supports
		// strips left, and a left-strip misread would silently change every
		// encoding. Extend rather than ignore if one ever appears.
		if n.StripLeft {
			return "", nil, false, false, fmt.Errorf("unigram: Strip(strip_left=true) unsupported")
		}
		return "", nil, false, n.StripRight, nil
	case "Sequence":
		for _, sub := range n.Normalizers {
			cm, rs, nSub, srSub, serr := buildNormalizer(sub)
			if serr != nil {
				return "", nil, false, false, serr
			}
			if cm != "" {
				if charsmap != "" {
					return "", nil, false, false, fmt.Errorf("unigram: multiple Precompiled normalizers")
				}
				charsmap = cm
			}
			replaces = append(replaces, rs...)
			nfc = nfc || nSub
			stripRight = stripRight || srSub
		}
		return charsmap, replaces, nfc, stripRight, nil
	default:
		return "", nil, false, false, fmt.Errorf("unigram: unsupported normalizer.type %q", n.Type)
	}
}

// parseMetaspaceKind classifies the pre_tokenizer into its Metaspace variant:
// Sequence[WhitespaceSplit, Metaspace] → metaspaceKindWhitespaceSplit (XLM-R/e5); a
// bare Metaspace → metaspaceKindPrepend (bge-m3) or metaspaceKindNever
// (cl-nagoya/ruri-v3-*) per its prepend config. Errors on any other shape.
func parseMetaspaceKind(raw json.RawMessage) (metaspaceKind, error) {
	var p struct {
		Type           string `json:"type"`
		PrependScheme  string `json:"prepend_scheme"`
		AddPrefixSpace *bool  `json:"add_prefix_space"`
		Pretokenizers  []struct {
			Type string `json:"type"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return metaspaceKindPrepend, fmt.Errorf("unigram: pre_tokenizer: %w", err)
	}
	switch p.Type {
	case "Metaspace":
		return bareMetaspaceKind(p.PrependScheme, p.AddPrefixSpace), nil
	case "Sequence":
		hasWS, hasMeta := false, false
		for _, pt := range p.Pretokenizers {
			switch pt.Type {
			case "WhitespaceSplit":
				hasWS = true
			case "Metaspace":
				hasMeta = true
			default:
				return metaspaceKindPrepend, fmt.Errorf("unigram: unsupported pre_tokenizer %q in sequence", pt.Type)
			}
		}
		if !hasMeta {
			return metaspaceKindPrepend, fmt.Errorf("unigram: pre_tokenizer sequence lacks Metaspace")
		}
		if hasWS {
			return metaspaceKindWhitespaceSplit, nil
		}
		// A Metaspace-only sequence behaves like a bare Metaspace (its prepend config
		// lives on the sub-element; no current model uses this path, so default to the
		// historical prepend behaviour rather than mis-reading the Sequence's fields).
		return metaspaceKindPrepend, nil
	default:
		return metaspaceKindPrepend, fmt.Errorf("unigram: unsupported pre_tokenizer.type %q", p.Type)
	}
}

// bareMetaspaceKind maps a bare Metaspace's prepend config to its kind.
// prepend_scheme="never" — or the legacy add_prefix_space=false — prepends no
// leading ▁ (cl-nagoya/ruri-v3-*); anything else ("always"/"first"/absent, or legacy
// add_prefix_space=true) prepends one (bge-m3), the historical default.
func bareMetaspaceKind(prependScheme string, addPrefixSpace *bool) metaspaceKind {
	if prependScheme == "never" {
		return metaspaceKindNever
	}
	if prependScheme == "" && addPrefixSpace != nil && !*addPrefixSpace {
		return metaspaceKindNever
	}
	return metaspaceKindPrepend
}

// templateSpecials reads a TemplateProcessing "single" template into the special
// ids that come before (prefix) and after (suffix) the sequence — e.g. for XLM-R,
// prefix=[<s>], suffix=[</s>]. Unknown/empty templates yield no specials.
func templateSpecials(single []map[string]templateElement, specials map[string]struct {
	IDs []int32 `json:"ids"`
}) (prefix, suffix []int32) {
	seenSeq := false
	for _, el := range single {
		if _, ok := el["Sequence"]; ok {
			seenSeq = true
			continue
		}
		st, ok := el["SpecialToken"]
		if !ok {
			continue
		}
		ids := specials[st.ID].IDs
		if seenSeq {
			suffix = append(suffix, ids...)
		} else {
			prefix = append(prefix, ids...)
		}
	}
	return prefix, suffix
}

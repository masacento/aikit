package embed

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// ──────────────────────────────────────────────────────────────────────────────
// SentencePiece-style BPE (Metaspace + byte-fallback)
// ──────────────────────────────────────────────────────────────────────────────
//
// ModernBERT (hotchpotch/bekko-embedding-v1-*) tokenizes with a SentencePiece-
// trained BPE exported to HF tokenizer.json. Its model.type is "BPE" — like the
// GPT-2/RoBERTa path in tokenize_bpe.go — but on every other axis it matches the
// SentencePiece family instead, so it gets its own backend:
//
//   - normalizer: a Replace (" " → "▁") — no Precompiled charsmap, no lowercase.
//   - pre_tokenizer: a BARE Metaspace (replacement "▁", prepend always, split),
//     reproduced by metaspaceBare (shared with the Unigram backend). The Replace
//     normalizer and Metaspace both map space→▁, so their composition is exactly
//     metaspaceBare applied once.
//   - model: ranked BPE merges over CHARACTERS (not GPT-2 byte-runes), with
//     byte_fallback — a character absent from the vocab decomposes to its UTF-8
//     bytes as "<0xNN>" (uppercase hex) tokens.
//   - post_processor: TemplateProcessing wrapping <bos> … <eos>.
//
// The merge loop is the same greedy lowest-rank, merge-all-occurrences algorithm
// as tokenize_bpe.go's bpeInto, run over explicit initial spans (one per input
// character / byte-fallback token) rather than per-rune.

// spBPEBackend is the SentencePiece-style BPE tokenizer. It plugs into the public
// embed.Tokenizer, which dispatches to it when tokenizer.json is a BPE model with
// a Metaspace pre-tokenizer.
type spBPEBackend struct {
	vocab        map[string]int32  // symbol → id
	rank         map[[2]string]int // adjacent-symbol pair → merge priority (lower = earlier)
	replaces     []replaceRule     // Replace normalizers, applied in Sequence order
	byteFallback bool
	fuseUnk      bool
	unkToken     string
	addedTokens  map[string]int32
	addedKeys    []string // longest-first carve-out scan order
	addedLstrip  map[string]bool
	addedRstrip  map[string]bool
	unkID        int32
	vocabSize    int
	prefixIDs    []int32 // TemplateProcessing single: specials before the sequence (<bos>)
	suffixIDs    []int32 // ... and after (<eos>)
}

// isMetaspaceBPETokenizer reports whether a BPE tokenizer.json is the
// SentencePiece-style variant (a Metaspace pre-tokenizer, bare or inside a
// Sequence) rather than GPT-2/RoBERTa byte-level BPE (a ByteLevel pre-tokenizer).
// The pre_tokenizer is the reliable tell: the model.type is "BPE" for both.
// The pre_tokenizer object is located with jsonProbeRaw and unmarshalled on its
// own, rather than unmarshalling the whole file for one field — see
// tokenize_probe.go for why that matters at 6.7 MB.
func isMetaspaceBPETokenizer(data []byte) bool {
	raw := jsonProbeRaw(data, "pre_tokenizer")
	if raw == nil {
		return false
	}
	var probe struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type string `json:"type"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	switch probe.Type {
	case "Metaspace":
		return true
	case "Sequence":
		for _, pt := range probe.Pretokenizers {
			if pt.Type == "Metaspace" {
				return true
			}
		}
	}
	return false
}

// normalizeText runs the Replace normalizer rules in order. For bekko this is the
// single " " → "▁" rule; metaspaceBare then applies the Metaspace step.
func (b *spBPEBackend) normalizeText(text string) string {
	for _, r := range b.replaces {
		text = r.re.ReplaceAllString(text, r.content)
	}
	return text
}

// bpePiece runs the byte-fallback character decomposition + ranked BPE merges over
// ONE metaspace piece and returns its ids.
func (b *spBPEBackend) bpePiece(piece string) []int32 {
	// Initial symbols: each character maps to itself when it is a vocab symbol;
	// otherwise (byte_fallback) it decomposes to its UTF-8 bytes as "<0xNN>". The
	// symbols are concatenated into `mapped` with one span per symbol, so the merge
	// loop works on offsets and the rank/vocab keys are free sub-slices of mapped.
	var sb strings.Builder
	sb.Grow(len(piece) + 8)
	var cur []bpeSpan
	for _, c := range piece {
		s := string(c)
		if _, ok := b.vocab[s]; ok {
			start := sb.Len()
			sb.WriteString(s)
			cur = append(cur, bpeSpan{start, sb.Len()})
			continue
		}
		if b.byteFallback {
			for _, by := range []byte(s) {
				start := sb.Len()
				fmt.Fprintf(&sb, "<0x%02X>", by)
				cur = append(cur, bpeSpan{start, sb.Len()})
			}
			continue
		}
		start := sb.Len()
		sb.WriteString(b.unkToken)
		cur = append(cur, bpeSpan{start, sb.Len()})
	}
	mapped := sb.String()

	// Greedy BPE: repeatedly merge every occurrence of the lowest-rank adjacent
	// pair until no adjacent pair is a known merge (same algorithm as bpeInto).
	var nxt []bpeSpan
	for len(cur) >= 2 {
		bestRank := math.MaxInt
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
		cur, nxt = nxt, cur
	}

	ids := make([]int32, 0, len(cur))
	prevUnk := false
	for _, sp := range cur {
		id, ok := b.vocab[mapped[sp.lo:sp.hi]]
		if !ok {
			id = b.unkID
		}
		// fuse_unk collapses consecutive <unk> (rare with byte_fallback on).
		if b.fuseUnk && id == b.unkID && prevUnk {
			continue
		}
		ids = append(ids, id)
		prevUnk = id == b.unkID
	}
	return ids
}

// encodeSegment runs normalize → Metaspace pre-tokenize → BPE over one fragment.
func (b *spBPEBackend) encodeSegment(text string) []int32 {
	normalized := b.normalizeText(text)
	var ids []int32
	for _, piece := range metaspaceBare(normalized) {
		ids = append(ids, b.bpePiece(piece)...)
	}
	return ids
}

// encode runs the added-token carve-out, then SentencePiece-BPE over each segment,
// producing the BARE id sequence (no template specials). Mirrors bpeBackend.encode.
func (b *spBPEBackend) encode(text string) []int32 {
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

// encodeWithSpecials wraps encode with the TemplateProcessing specials
// (<bos> ++ body ++ <eos>), right-truncating body so the total is at most maxLen.
func (b *spBPEBackend) encodeWithSpecials(text string, maxLen int) []int32 {
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
// tokenizer.json → spBPEBackend
// ──────────────────────────────────────────────────────────────────────────────

// spBPEJSON is the tokenizer.json shape for a SentencePiece-style BPE tokenizer
// (the fields the Replace normalizer + Metaspace + BPE + byte-fallback +
// TemplateProcessing pipeline needs). Vocab is a symbol→id map; merges is a list
// of "a b" pairs ranked by position.
type spBPEJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Lstrip  bool   `json:"lstrip"`
		Rstrip  bool   `json:"rstrip"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type         string            `json:"type"`
		UnkToken     string            `json:"unk_token"`
		Vocab        map[string]int32  `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		ByteFallback bool              `json:"byte_fallback"`
		FuseUnk      bool              `json:"fuse_unk"`
	} `json:"model"`
	PostProcessor struct {
		Type          string                       `json:"type"`
		Single        []map[string]templateElement `json:"single"`
		SpecialTokens map[string]struct {
			IDs []int32 `json:"ids"`
		} `json:"special_tokens"`
	} `json:"post_processor"`
}

// parseSPBPETokenizer builds an spBPEBackend from tokenizer.json bytes.
func parseSPBPETokenizer(data []byte) (*spBPEBackend, error) {
	var raw spBPEJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json (sp-bpe): %w", err)
	}
	if len(raw.Model.Vocab) == 0 {
		return nil, fmt.Errorf("sp-bpe: empty vocab")
	}
	replaces, err := spBpeNormalizer(raw.Normalizer)
	if err != nil {
		return nil, err
	}
	if err := spBpePreTok(raw.PreTokenizer); err != nil {
		return nil, err
	}
	if raw.PostProcessor.Type != "TemplateProcessing" {
		return nil, fmt.Errorf("sp-bpe: unsupported post_processor.type %q (TemplateProcessing only)", raw.PostProcessor.Type)
	}

	// Merges appear either as ["a","b"] arrays (the SentencePiece/tiktoken export
	// this model uses) or as "a b" space-joined strings (the classic HF BPE form);
	// parseBPEMerges (tokenize_bpe.go) accepts both, for both BPE paths.
	rank, err := parseBPEMerges(raw.Model.Merges, "sp-bpe")
	if err != nil {
		return nil, err
	}

	added := make(map[string]int32, len(raw.AddedTokens))
	var lstrip, rstrip map[string]bool
	for _, at := range raw.AddedTokens {
		added[at.Content] = at.ID
		if at.Lstrip {
			if lstrip == nil {
				lstrip = make(map[string]bool)
			}
			lstrip[at.Content] = true
		}
		if at.Rstrip {
			if rstrip == nil {
				rstrip = make(map[string]bool)
			}
			rstrip[at.Content] = true
		}
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

	unkToken := raw.Model.UnkToken
	if unkToken == "" {
		unkToken = "<unk>"
	}
	unkID := int32(0)
	if id, ok := added[unkToken]; ok {
		unkID = id
	} else if id, ok := raw.Model.Vocab[unkToken]; ok {
		unkID = id
	}

	prefixIDs, suffixIDs := templateSpecials(raw.PostProcessor.Single, raw.PostProcessor.SpecialTokens)

	return &spBPEBackend{
		vocab:        raw.Model.Vocab,
		rank:         rank,
		replaces:     replaces,
		byteFallback: raw.Model.ByteFallback,
		fuseUnk:      raw.Model.FuseUnk,
		unkToken:     unkToken,
		addedTokens:  added,
		addedKeys:    addedKeys,
		addedLstrip:  lstrip,
		addedRstrip:  rstrip,
		unkID:        unkID,
		vocabSize:    len(raw.Model.Vocab),
		prefixIDs:    prefixIDs,
		suffixIDs:    suffixIDs,
	}, nil
}

// spBpeNormalizer parses the normalizer into an ordered list of Replace rules. It
// supports a bare Replace, a Sequence of Replaces, or an absent normalizer
// (identity); anything else (e.g. a Precompiled charsmap) errors so an unsupported
// normalizer fails loudly (→ best-effort nil in the loader) rather than silently
// mis-normalizing.
func spBpeNormalizer(raw json.RawMessage) ([]replaceRule, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var n struct {
		Type        string            `json:"type"`
		Normalizers []json.RawMessage `json:"normalizers"`
		Pattern     struct {
			Regex  string `json:"Regex"`
			String string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &n); err != nil {
		return nil, fmt.Errorf("sp-bpe: normalizer: %w", err)
	}
	switch n.Type {
	case "Replace":
		pat := n.Pattern.Regex
		if pat == "" { // a literal-string Replace: match it verbatim
			pat = regexp.QuoteMeta(n.Pattern.String)
		}
		re, err := regexp.Compile(pat)
		if err != nil {
			return nil, fmt.Errorf("sp-bpe: Replace pattern %q: %w", pat, err)
		}
		return []replaceRule{{re: re, content: n.Content}}, nil
	case "Sequence":
		var out []replaceRule
		for _, sub := range n.Normalizers {
			rs, err := spBpeNormalizer(sub)
			if err != nil {
				return nil, err
			}
			out = append(out, rs...)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("sp-bpe: unsupported normalizer.type %q", n.Type)
	}
}

// spBpePreTok validates that the pre-tokenizer is exactly the bare Metaspace
// configuration metaspaceBare reproduces. A Sequence or different replacement,
// prepend, or split option is rejected rather than silently tokenized with the
// wrong pipeline.
func spBpePreTok(raw json.RawMessage) error {
	var p struct {
		Type          string `json:"type"`
		Replacement   string `json:"replacement"`
		PrependScheme string `json:"prepend_scheme"`
		Split         *bool  `json:"split"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("sp-bpe: pre_tokenizer: %w", err)
	}
	if p.Type != "Metaspace" {
		return fmt.Errorf("sp-bpe: unsupported pre_tokenizer.type %q", p.Type)
	}
	if p.Replacement != string(metaspace) {
		return fmt.Errorf("sp-bpe: Metaspace replacement %q unsupported (%q only)", p.Replacement, string(metaspace))
	}
	if p.PrependScheme != "always" {
		return fmt.Errorf("sp-bpe: Metaspace prepend_scheme %q unsupported (always only)", p.PrependScheme)
	}
	if p.Split == nil || !*p.Split {
		return fmt.Errorf("sp-bpe: Metaspace split must be true")
	}
	return nil
}

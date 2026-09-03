package ner

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
)

// gliner2.go — GLiNER2 "boundary" extractor (fastino/gliner2.5-multi-v1 and
// friends): zero-shot extraction by prompting the SAME DeBERTa-v2 backbone family
// as GLiNER v1 with a schema prompt, but a completely different head.
//
// The word-level input for an entity request {person} over "Apple hired John." is
//
//	( [P] entities ( [E] person ) ) [SEP_TEXT] Apple hired John .
//
// — literal "(" ")" and the group name are ordinary words; [P]/[E]/[SEP_TEXT] are
// NATIVE vocab tokens (ids 250104/250106/250103), not added markers. After the
// backbone, every extractive marker's own hidden state is one query, each text
// word is represented by its FIRST sub-token, and entities are scored by a
// boundary head (gliner2_boundary.go) rather than v1's span grid.
//
// Config layout differs from v1: the head config is the checkpoint's top-level
// config.json (architecture="boundary" with a boundary_head block) and the
// backbone config is encoder_config/config.json — both shipped in-repo, so nothing
// is fetched from the hub. The tokenizer is a self-contained tokenizer.json
// (Unigram embedded; there is no spm.model).
//
// Parity is pinned against the gliner2 package (scripts/oracle/pin_gliner2.py).

// gliner2Config is the subset of the checkpoint's top-level config.json the
// loader and forward need.
type gliner2Config struct {
	Architecture string `json:"architecture"`
	ModelName    string `json:"model_name"`
	MaxLen       int    `json:"max_len"` // WORDS, not sub-tokens (relative positions)
	TokenPooling string `json:"token_pooling"`

	// BoundaryHead carries the head's dimensions; see gliner2_boundary.go.
	BoundaryHead gliner2BoundaryConfig `json:"boundary_head"`
}

// gliner2BoundaryConfig is the config.json boundary_head block.
type gliner2BoundaryConfig struct {
	BoundaryDim                  int     `json:"boundary_dim"`
	BoundaryAttentionLayers      int     `json:"boundary_attention_layers"`
	BoundaryAttentionHeads       int     `json:"boundary_attention_heads"`
	BoundaryAttentionWindow      int     `json:"boundary_attention_window"`
	BoundaryRefinementLayers     int     `json:"boundary_refinement_layers"`
	BoundaryFFNMultiplier        float64 `json:"boundary_ffn_multiplier"`
	PairDim                      int     `json:"pair_dim"`
	ContentDim                   int     `json:"content_dim"`
	CandidatePool                string  `json:"candidate_pool"`
	PoolSize                     int     `json:"pool_size"`
	CandidateBudget              int     `json:"candidate_budget"`
	PoolBoundaryTopK             int     `json:"pool_boundary_top_k"`
	MinPoolPerQuery              int     `json:"min_pool_per_query"`
	CandidateAttentionLayers     int     `json:"candidate_attention_layers"`
	CandidateAttentionHeads      int     `json:"candidate_attention_heads"`
	QueryAttentionLayers         int     `json:"query_attention_layers"`
	EnableSpanContent            bool    `json:"enable_span_content"`
	EnableRotaryEndpoints        bool    `json:"enable_rotary_endpoints"`
	RotaryBase                   float64 `json:"rotary_base"`
	MultiheadPairCompatHeads     int     `json:"multihead_pair_compat_heads"`
	EndpointDifferenceFeatures   bool    `json:"endpoint_difference_features"`
	UseInsideEvidence            bool    `json:"use_inside_evidence"`
	QueryConditionedInsideWeight bool    `json:"query_conditioned_inside_weight"`
	PairTemperature              float64 `json:"pair_temperature"`
	AbstentionThreshold          float64 `json:"abstention_threshold"`
	OverlapPolicy                string  `json:"overlap_policy"`
}

// GLiNER2 is a loaded boundary-extractor model. Immutable after load; Predict is
// read-only-safe for concurrent use.
type GLiNER2 struct {
	cfg      gliner2Config
	backbone *encoder.DeBERTa
	tok      *embed.Tokenizer
	head     *g2Head

	// Native special-token ids, resolved once at load.
	pID, eID, lID          int32
	sepTextID, sepStructID int32
	maxLen                 int // words
}

// LoadGLiNER2 loads a GLiNER2 boundary checkpoint from dir. It expects:
//
//	config.json             the head config (architecture must be "boundary")
//	encoder_config/         the backbone's deberta-v2 config.json
//	model.safetensors       encoder.* + boundary_head.* weights (F32)
//	tokenizer.json          self-contained Unigram tokenizer
func LoadGLiNER2(dir string) (*GLiNER2, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("ner: read config.json: %w", err)
	}
	var c gliner2Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("ner: parse config.json: %w", err)
	}
	switch {
	case c.Architecture != "boundary":
		return nil, fmt.Errorf("ner: architecture=%q unsupported (boundary only)", c.Architecture)
	case c.TokenPooling != "" && c.TokenPooling != "first":
		return nil, fmt.Errorf("ner: token_pooling=%q unsupported (first only)", c.TokenPooling)
	}
	if c.MaxLen <= 0 {
		c.MaxLen = 4096
	}

	backbone, err := encoder.LoadDeBERTaAt(
		filepath.Join(dir, "encoder_config", "config.json"),
		filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("ner: load backbone: %w", err)
	}

	g := &GLiNER2{cfg: c, backbone: backbone, maxLen: c.MaxLen}
	if err := g.loadTokenizer(dir); err != nil {
		return nil, errors.Join(err, backbone.Close())
	}
	if err := g.loadHead(dir); err != nil {
		return nil, errors.Join(err, backbone.Close())
	}
	return g, nil
}

// loadHead copies the boundary head weights out of the shared safetensors file.
// The backbone's mmap already holds the file open; this opens a second read-only
// mapping, reads ~20 MB of head tensors, and releases it.
func (g *GLiNER2) loadHead(dir string) error {
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return fmt.Errorf("ner: open safetensors: %w", err)
	}
	defer func() { _ = st.Close() }()
	g.head, err = loadG2Head(st, g.cfg.BoundaryHead)
	return err
}

func (g *GLiNER2) loadTokenizer(dir string) error {
	tok, err := embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("ner: load tokenizer: %w", err)
	}
	g.tok = tok
	for _, s := range []struct {
		literal string
		dst     *int32
	}{
		{"[P]", &g.pID}, {"[E]", &g.eID}, {"[L]", &g.lID},
		{"[SEP_TEXT]", &g.sepTextID}, {"[SEP_STRUCT]", &g.sepStructID},
	} {
		id, ok := tok.SpecialID(s.literal)
		if !ok {
			return fmt.Errorf("ner: tokenizer has no %q — not the GLiNER2 tokenizer.json?", s.literal)
		}
		*s.dst = id
	}
	return nil
}

// Close releases the backbone's mmap. Idempotent.
func (g *GLiNER2) Close() error { return g.backbone.Close() }

// MaxLen is the word-level input cap (the reference's max_len counts WORDS —
// deberta-v2's relative positions need no sub-token cap).
func (g *GLiNER2) MaxLen() int { return g.maxLen }

// buildPrompt emits the schema prompt for an entity request and returns the token
// ids plus the sequence position of each [E] marker (one per label, in order):
//
//	( [P] entities ( [E] label₁ [E] label₂ … ) )
//
// The parentheses and the group name are ordinary words. The name "entities" is
// the group key the gliner2 processor uses for the entity group; a named group
// ("name: <prompt>") or descriptions/examples would change it, which this port
// does not support yet.
func (g *GLiNER2) buildPrompt(labels []string) (ids []int32, markerPos []int) {
	ids = append(ids, g.tok.Encode("(")...)
	ids = append(ids, g.pID)
	ids = append(ids, g.tok.Encode("entities")...)
	ids = append(ids, g.tok.Encode("(")...)
	for _, l := range labels {
		markerPos = append(markerPos, len(ids))
		ids = append(ids, g.eID)
		// A label is ONE word as far as the pipeline is concerned ("date of
		// birth" stays whole) — the processor appends the label string to the
		// word list and tokenizes words individually.
		ids = append(ids, g.tok.Encode(l)...)
	}
	ids = append(ids, g.tok.Encode(")")...)
	ids = append(ids, g.tok.Encode(")")...)
	return ids, markerPos
}

// encodeInput assembles the full prompt+text sequence:
//
//	<prompt> [SEP_TEXT] <text words>
//
// and reports, in sequence coordinates, where each [E] marker and each text
// word's first sub-token landed. Text words are truncated to max_len WORDS (the
// reference's unit), and a truncated-away word's position is -1. A prompt that
// lost a marker is impossible here — the prompt precedes the text and is never
// truncated — but the marker count is asserted anyway, keeping the same posture
// as the v1 loader.
func (g *GLiNER2) encodeInput(words []Word, labels []string) (ids []int32, markerPos, wordPos []int, err error) {
	if len(words) > g.maxLen {
		words = words[:g.maxLen]
	}
	ids, markerPos = g.buildPrompt(labels)
	if len(markerPos) != len(labels) {
		return nil, nil, nil, fmt.Errorf("ner: internal: %d markers for %d labels", len(markerPos), len(labels))
	}
	ids = append(ids, g.sepTextID)

	wordTexts := make([]string, len(words))
	for i, w := range words {
		wordTexts[i] = w.Text
	}
	bodyIDs, first := g.tok.EncodeWords(wordTexts)
	base := len(ids)
	ids = append(ids, bodyIDs...)

	wordPos = make([]int, len(first))
	for i, f := range first {
		if f < 0 {
			wordPos[i] = -1
			continue
		}
		wordPos[i] = base + int(f)
	}
	return ids, markerPos, wordPos, nil
}

// endsWithSentencePunct mirrors processor.py: text ends with '.', '!' or '?'.
func endsWithSentencePunct(text string) bool {
	r := []rune(text)
	switch r[len(r)-1] {
	case '.', '!', '?':
		return true
	}
	return false
}

// g2Core is the encoded input plus the pooled word/query states the head runs
// on — _encode_core for B=1 (fast-routing branch).
type g2Core struct {
	Words     []Word
	ids       []int32
	markerPos []int
	wordPos   []int

	text    []float32 // [W, H] first-subtoken word states
	tmask   []bool    // [W]
	queries []float32 // [Q, H] [E]-marker states
	Q       int
}

// encodeCore splits/encodes and runs the backbone, then pools words to their
// first sub-token and gathers the marker states (model.py:1286-1324 fast path).
func (g *GLiNER2) encodeCore(text string, labels []string, cjk bool) (*g2Core, error) {
	// The processor appends terminal punctuation when missing
	// (processor.py:522-525: ".", "!", "?" count), so the LAST word of every
	// input is "." unless the text already ends punctuated. The punctuation
	// participates in the forward and shifts every decoded offset after it.
	if text != "" && !endsWithSentencePunct(text) {
		text += "."
	} else if text == "" {
		text = "."
	}
	words := SplitWords2(text)
	if cjk {
		words = SplitWords2CJK(text)
	}
	words = limitWords(words, g.maxLen)
	ids, markerPos, wordPos, err := g.encodeInput(words, labels)
	if err != nil {
		return nil, err
	}
	hidden := g.backbone.HiddenStates(ids)
	H := g.backbone.HiddenDim()
	L := len(hidden) / H

	c := &g2Core{
		Words:     words,
		ids:       ids,
		markerPos: markerPos,
		wordPos:   wordPos,
		text:      make([]float32, len(words)*H),
		tmask:     make([]bool, len(words)),
		Q:         len(markerPos),
	}
	// Word states: FIRST sub-token, zero rows (masked) when a word produced
	// none — the reference gathers with clamped indices and a zeroing mask.
	for i, p := range wordPos {
		if p < 0 || p >= L {
			continue
		}
		c.tmask[i] = true
		copy(c.text[i*H:(i+1)*H], hidden[p*H:(p+1)*H])
	}
	// Query states: the [E] markers' own contextual embeddings.
	c.queries = make([]float32, len(markerPos)*H)
	for i, p := range markerPos {
		if p >= L {
			return nil, fmt.Errorf("ner: [E] marker %d out of range (max_len=%d)", i, g.maxLen)
		}
		copy(c.queries[i*H:(i+1)*H], hidden[p*H:(p+1)*H])
	}
	return c, nil
}

func limitWords(words []Word, maxLen int) []Word {
	if maxLen > 0 && len(words) > maxLen {
		return words[:maxLen]
	}
	return words
}

// forwardCore runs the full boundary head: encode → marginals → shared pool →
// pooled reranker. Returns the raw pair logits ([C,Q]) and per-query null
// (abstention) logits.
func (g *GLiNER2) forwardCore(c *g2Core) (m *g2Marginals, pool g2Pool, scores, nullLogits []float32, err error) {
	H := g.backbone.HiddenDim()
	W := len(c.Words)
	bs := g.head.boundaryForward(c.text, W, H)
	m = g.head.marginalForward(bs, c.text, W, c.tmask, c.queries, c.Q)
	pool = g.head.buildPool(m, bs)
	scores = g.head.scorePool(m, bs, c.text, W, c.queries, c.Q, pool)
	nullLogits = g.head.nullLogit(c.queries, c.Q)
	return m, pool, scores, nullLogits, nil
}

// ClassResult is one classification label with its probability.
type ClassResult struct {
	Label string
	Prob  float64
}

// Classify runs the shared classification head over a classification schema:
// the prompt is `( [P] task ( [L] l₁ [L] l₂ … ) ) [SEP_TEXT] text` and each
// [L] marker state is scored by the shared MLP. Single-label requests softmax
// over the labels; multi-label ones sigmoid (runtime.py:395-416).
func (g *GLiNER2) Classify(text, task string, labels []string, multiLabel bool) ([]ClassResult, error) {
	if len(labels) == 0 {
		return nil, nil
	}
	// Terminal punctuation, same rule as encodeCore.
	if text != "" && !endsWithSentencePunct(text) {
		text += "."
	} else if text == "" {
		text = "."
	}
	words := SplitWords2(text)

	ids, markerPos := g.buildClassifyPrompt(task, labels)
	ids = append(ids, g.sepTextID)
	wordTexts := make([]string, len(words))
	for i, w := range words {
		wordTexts[i] = w.Text
	}
	bodyIDs, _ := g.tok.EncodeWords(wordTexts)
	ids = append(ids, bodyIDs...)

	hidden := g.backbone.HiddenStates(ids)
	H := g.backbone.HiddenDim()
	L := len(hidden) / H
	markerStates := make([]float32, len(markerPos)*H)
	for i, p := range markerPos {
		if p >= L {
			return nil, fmt.Errorf("ner: [L] marker %d out of range", i)
		}
		copy(markerStates[i*H:(i+1)*H], hidden[p*H:(p+1)*H])
	}
	logits := g.head.classify(markerStates, len(markerPos))

	// softmax or sigmoid, then pick like runtime.py.
	probs := make([]float64, len(logits))
	if multiLabel {
		for i, v := range logits {
			probs[i] = float64(sigmoid(v))
		}
	} else {
		max := float32(math.Inf(-1))
		for _, v := range logits {
			if v > max {
				max = v
			}
		}
		var sum float64
		for _, v := range logits {
			sum += math.Exp(float64(v - max))
		}
		for i, v := range logits {
			probs[i] = math.Exp(float64(v-max)) / sum
		}
	}
	out := make([]ClassResult, len(labels))
	for i, l := range labels {
		out[i] = ClassResult{Label: l, Prob: probs[i]}
	}
	return out, nil
}

// buildClassifyPrompt is _transform_schema for a classification group: the
// entity layout with [L] in place of [E] (no "|" separators — that shape
// belongs to the legacy span prompt builder, not the boundary path).
func (g *GLiNER2) buildClassifyPrompt(task string, labels []string) (ids []int32, markerPos []int) {
	ids = append(ids, g.tok.Encode("(")...)
	ids = append(ids, g.pID)
	ids = append(ids, g.tok.Encode(task)...)
	ids = append(ids, g.tok.Encode("(")...)
	for _, l := range labels {
		markerPos = append(markerPos, len(ids))
		ids = append(ids, g.lID)
		ids = append(ids, g.tok.Encode(l)...)
	}
	ids = append(ids, g.tok.Encode(")")...)
	ids = append(ids, g.tok.Encode(")")...)
	return ids, markerPos
}

// Predict extracts entities of the given types from text — the entity branch of
// BoundaryExtractor._extract_from_batch: threshold, per-query flat overlap
// resolution, then token→character offsets via the word boundary maps.
//
// labels are free-form type names, case- and wording-sensitive like any other
// prompt text.
func (g *GLiNER2) Predict(text string, labels []string, opts Opts) ([]Entity, error) {
	if len(labels) == 0 {
		return nil, nil
	}
	threshold := opts.threshold()
	overlap := "flat"
	if opts.Nested {
		overlap = "nested"
	}

	c, err := g.encodeCore(text, labels, opts.CJKSplit)
	if err != nil {
		return nil, err
	}
	m, pool, scores, nullLogits, err := g.forwardCore(c)
	if err != nil {
		return nil, err
	}
	grouped := g.thresholdCandidates(m, pool, scores, threshold, nullLogits)

	// Per-query overlap resolution and word→byte surface location. Labels are
	// emitted in declaration order, each list ranked like the reference
	// (confidence desc, then start/end/origin).
	out := make([]Entity, 0, len(labels))
	for _, candidate := range selectGrouped(grouped, overlap, opts.MultiLabel) {
		q, s := candidate.Query, candidate.Span
		ws, we := s.Start, s.End
		if ws < 0 || we > len(c.Words) || we <= ws {
			continue
		}
		cs, ce := c.Words[ws].Start, c.Words[we-1].End
		surface := trimSurface(text[cs:ce])
		if surface == "" {
			continue
		}
		out = append(out, Entity{
			Text:  surface,
			Label: labels[q],
			Start: cs,
			End:   ce,
			Score: float64(s.Score),
		})
	}
	return out, nil
}

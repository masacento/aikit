package cjk

import (
	"errors"
	"strings"
	"unicode/utf8"
)

// Word is a segmented word with its POS tag.
type Word struct {
	Word string
	Pos  Upos
}

// Segmenter performs word segmentation and optional POS tagging using a
// loaded model. It mirrors litsea's inference paths exactly.
type Segmenter struct {
	language Language
	adaboost *AdaBoost
	pos      *AveragedPerceptron
	// packed and packedPos hold the AdaBoost and Averaged Perceptron weights
	// compiled to packed integer keys for the two hot loops. Both are always
	// non-nil; a table is empty when the corresponding model is absent.
	packed    *packedModel
	packedPos *packedPerceptron
}

// NewSegmenter creates a word-segmentation Segmenter for the given language
// backed by an AdaBoost model.
func NewSegmenter(language Language, adaboost *AdaBoost) *Segmenter {
	// Compile the packed scoring table eagerly so the common
	// load-then-segment path never has to build it mid-stream.
	return &Segmenter{
		language:  language,
		adaboost:  adaboost,
		packed:    buildPackedModel(language, adaboost),
		packedPos: &packedPerceptron{},
	}
}

// NewPosSegmenter creates a joint segmentation + POS-tagging Segmenter backed
// by an Averaged Perceptron model.
func NewPosSegmenter(language Language, pos *AveragedPerceptron) *Segmenter {
	// There is no AdaBoost model, so an empty table is its correct
	// compilation: Segment would return one word per character.
	return &Segmenter{
		language:  language,
		pos:       pos,
		packed:    &packedModel{},
		packedPos: buildPackedPerceptron(language, pos),
	}
}

// sentCtx is the per-position context of one sentence in the numeric form the
// packed feature keys are built from. The first three positions are the
// B3/B2/B1 head sentinels and the last three are the E1/E2/E3 tail sentinels,
// so the sentence's n characters occupy positions 3..n+2.
type sentCtx struct {
	sentence string
	chars    []uint32 // char codes; sentinels are sentinelBase + k
	types    []uint8  // type ids (indices into Language.typeCodes)
	tags     []uint8  // boundary tag ids, grown one per decided position
	// offs holds the byte offset of each of the n characters plus a final
	// len(sentence) terminator, so character k is sentence[offs[k]:offs[k+1]]
	// and a word spanning characters a..b is sentence[offs[a]:offs[b]].
	offs []int32
}

// newContext builds the padded numeric context for a non-empty text.
func (s *Segmenter) newContext(text string) *sentCtx {
	n := utf8.RuneCountInString(text)
	// types and tags have the same element type and lifetime, so they share
	// one allocation; tags is capped at the split point so that growing it
	// can never reach into types.
	block := make([]uint8, 2*(n+6))
	ctx := &sentCtx{
		sentence: text,
		chars:    make([]uint32, n+6),
		types:    block[:n+6],
		// tags[0..3] are the fixed U padding: three for the head sentinels
		// plus the first character, which has no boundary decision before it.
		tags: append(block[n+6:n+6:2*(n+6)], tagU, tagU, tagU, tagU),
		offs: make([]int32, n+1),
	}
	// types is zero-valued, and otherTypeID is 0, so the sentinel positions
	// (and only they) already carry the right type id.
	for k := 0; k < 3; k++ {
		ctx.chars[k] = sentinelBase + uint32(k)
		ctx.chars[n+3+k] = sentinelBase + uint32(3+k)
	}
	k := 0
	for off, r := range text {
		ctx.offs[k] = int32(off)
		ctx.chars[3+k] = uint32(r)
		ctx.types[3+k] = s.language.charTypeID(r)
		k++
	}
	ctx.offs[n] = int32(len(text))
	return ctx
}

// numChars returns the number of real characters in the sentence.
func (c *sentCtx) numChars() int { return len(c.offs) - 1 }

// word returns the substring spanning characters a..b, where a and b are
// character positions (the same indices the scoring loop uses).
func (c *sentCtx) word(a, b int) string {
	return c.sentence[c.offs[a-3]:c.offs[b-3]]
}

// Segment segments a sentence into words.
func (s *Segmenter) Segment(sentence string) []string {
	if sentence == "" {
		return nil
	}
	ctx := s.newContext(sentence)
	templates := templatesFor(s.language)
	packed := s.packed
	// The bias is a sum over all model weights; compute it once per sentence
	// instead of once per character.
	bias := s.adaboost.Bias()

	n := ctx.numChars()
	result := make([]string, 0, n)
	wordStart := 3
	for i := 4; i < 3+n; i++ {
		// Sum the weights by packed key; a miss adds 0.0, so the float64
		// accumulation sequence matches the string-keyed reference bit for
		// bit.
		score := bias
		for k := range templates {
			score += packed.weight(templates[k].pack(i, ctx.tags, ctx.chars, ctx.types))
		}
		if score >= 0 {
			result = append(result, ctx.word(wordStart, i))
			wordStart = i
			ctx.tags = append(ctx.tags, tagB)
		} else {
			ctx.tags = append(ctx.tags, tagO)
		}
	}
	return append(result, ctx.word(wordStart, 3+n))
}

// SegmentWithPos segments a sentence into words with POS tags.
func (s *Segmenter) SegmentWithPos(sentence string) ([]Word, error) {
	if sentence == "" {
		return nil, nil
	}
	if s.pos == nil {
		return nil, errors.New("POS learner is not set")
	}
	ctx := s.newContext(sentence)
	// Score buffer reused across positions so that scoring a sentence
	// allocates nothing per character.
	scores := make([]float64, len(s.pos.classes))

	// The first character always starts the first word; its predicted label
	// is used only to determine the first word's POS.
	currentPos := posOf(s.predictAt(3, ctx, scores))

	n := ctx.numChars()
	result := make([]Word, 0, n)
	wordStart := 3
	for i := 4; i < 3+n; i++ {
		label := s.predictAt(i, ctx, scores)
		if strings.HasPrefix(label, "B-") {
			result = append(result, Word{Word: ctx.word(wordStart, i), Pos: currentPos})
			wordStart = i
			currentPos = posOf(label)
			ctx.tags = append(ctx.tags, tagB)
		} else {
			ctx.tags = append(ctx.tags, tagO)
		}
	}
	return append(result, Word{Word: ctx.word(wordStart, 3+n), Pos: currentPos}), nil
}

// predictAt sums each feature of position i into scores in table order and
// returns the highest-scoring class name. Features are looked up by packed key,
// so a position is scored without rendering a single feature string.
func (s *Segmenter) predictAt(i int, ctx *sentCtx, scores []float64) string {
	for k := range scores {
		scores[k] = 0
	}
	templates := templatesFor(s.language)
	for k := range templates {
		s.packedPos.accumulate(templates[k].pack(i, ctx.tags, ctx.chars, ctx.types), scores)
	}
	return s.pos.best(scores)
}

// posOf extracts the Upos from a segment label ("B-NOUN" -> NOUN). Returns X
// for the non-boundary label "O" or an empty/unknown label.
func posOf(label string) Upos {
	if strings.HasPrefix(label, "B-") {
		return Upos(label[2:])
	}
	return X
}

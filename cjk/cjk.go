// Package cjk segments CJK word runs (Japanese / Chinese / Korean) with the
// embedded litsea word-segmentation models — the optional Opts.CJKSplit path
// behind ner's word splitting, which otherwise leaves 「権藤三峰は武将である」
// as one opaque word.
package cjk

import (
	_ "embed"
	"fmt"
	"sync"
	"unicode"
)

//go:embed models/japanese.model
var japaneseModel []byte

//go:embed models/chinese.model
var chineseModel []byte

//go:embed models/korean.model
var koreanModel []byte

type defaultEntry struct {
	once sync.Once
	seg  *Segmenter
	err  error
}

var defaultSegmenters [3]defaultEntry // indexed by Language

func embeddedModel(l Language) []byte {
	switch l {
	case Japanese:
		return japaneseModel
	case Chinese:
		return chineseModel
	case Korean:
		return koreanModel
	}
	return nil
}

// Default returns a shared Segmenter for the language backed by the embedded
// segmentation model (models/<language>.model). The model is loaded once per
// language; the segmenter is safe for concurrent use.
func Default(l Language) (*Segmenter, error) {
	if l < Japanese || l > Korean {
		return nil, fmt.Errorf("unknown language: %d", int(l))
	}
	e := &defaultSegmenters[l]
	e.once.Do(func() {
		ada, err := LoadAdaBoostFromBytes(embeddedModel(l))
		if err != nil {
			e.err = err
			return
		}
		e.seg = NewSegmenter(l, ada)
	})
	return e.seg, e.err
}

// HasHangul reports whether the text contains any Hangul character.
func HasHangul(text string) bool {
	for _, r := range text {
		if unicode.Is(unicode.Hangul, r) {
			return true
		}
	}
	return false
}

// HasCJK reports whether the text contains any kana, Hangul, or Han
// character, i.e. anything the whitespace-based splitter cannot segment but
// the CJK word-segmentation models can.
func HasCJK(text string) bool {
	for _, r := range text {
		if unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r) || unicode.Is(unicode.Han, r) {
			return true
		}
	}
	return false
}

// DetectCJK picks the language whose script the text contains: Hangul goes to
// Korean, kana to Japanese, and bare Han (which could be either) to Japanese
// as the default. ok is false when no CJK script is present.
func DetectCJK(text string) (l Language, ok bool) {
	hasKana, hasHangul, hasHan := false, false, false
	for _, r := range text {
		switch {
		case unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r):
			hasKana = true
		case unicode.Is(unicode.Hangul, r):
			hasHangul = true
		case unicode.Is(unicode.Han, r):
			hasHan = true
		}
	}
	switch {
	case hasHangul:
		return Korean, true
	case hasKana, hasHan:
		return Japanese, true
	default:
		return 0, false
	}
}

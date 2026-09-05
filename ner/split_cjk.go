package ner

import cjkseg "github.com/masacento/cjk"

// SplitWordsCJK segments text for word-level NER input, choosing the
// splitter by content:
//
//   - Text containing kana, Hangul, or Han characters goes through the litsea
//     word-segmentation model for the detected language (cjkseg.Default), so
//     「アップルは」 becomes アップル / は instead of one opaque \w+ run. Hangul
//     routes to the Korean model, kana or Han to the Japanese model — bare Han
//     could be either, and Japanese is the default. This is a deliberate
//     departure from GLiNER's reference splitter, which does not segment CJK;
//     offsets stay byte-accurate either way.
//   - Everything else uses SplitWords, which is byte-for-byte the reference
//     behaviour.
//
// If a segmentation model cannot be loaded it falls back to SplitWords
// rather than failing.
func SplitWordsCJK(text string) []Word {
	lang, ok := cjkseg.DetectCJK(text)
	if !ok {
		return SplitWords(text)
	}
	seg, err := cjkseg.Default(lang)
	if err != nil || seg == nil {
		return SplitWords(text)
	}
	toks := seg.SplitWords(text)
	out := make([]Word, len(toks))
	for i, t := range toks {
		out[i] = Word{Text: t.Text, Start: t.Start, End: t.End}
	}
	return out
}

// segmentCJKRun segments a CJK word run with the litsea model for the
// detected language, falling back to the run itself when no model applies.
func segmentCJKRun(run string) []string {
	lang, ok := cjkseg.DetectCJK(run)
	if !ok {
		return []string{run}
	}
	seg, err := cjkseg.Default(lang)
	if err != nil || seg == nil {
		return []string{run}
	}
	return seg.Segment(run)
}

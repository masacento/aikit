package cjk

// Token is one segmented word plus its byte range in the original string,
// matching the shape ner.Word uses so predicted spans can be reported as
// offsets into the caller's text.
type Token struct {
	Text  string
	Start int // byte offset of the first byte
	End   int // byte offset one past the last
}

// SplitWords segments a sentence and returns each word with its byte
// offsets. Because the segmenter partitions the input, offsets accumulate
// by the UTF-8 length of each word.
func (s *Segmenter) SplitWords(sentence string) []Token {
	words := s.Segment(sentence)
	if len(words) == 0 {
		return nil
	}
	out := make([]Token, 0, len(words))
	start := 0
	for _, w := range words {
		end := start + len(w)
		out = append(out, Token{Text: w, Start: start, End: end})
		start = end
	}
	return out
}

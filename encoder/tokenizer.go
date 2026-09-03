package encoder

// Tokenizer is the text-to-input-IDs contract used by the encoder families
// that wrap text with model-specific special tokens.
//
// The returned IDs must already contain the model's required special tokens and
// be truncated to at most maxLen tokens.
type Tokenizer interface {
	EncodeWithSpecials(text string, maxLen int) ([]int32, error)
}

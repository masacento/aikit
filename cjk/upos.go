package cjk

// Upos is a Universal POS (UPOS) tag. The 17 part-of-speech categories
// defined by Universal Dependencies. <https://universaldependencies.org/u/pos/>
type Upos string

// The 17 UPOS tags.
const (
	ADJ   Upos = "ADJ"
	ADP   Upos = "ADP"
	ADV   Upos = "ADV"
	AUX   Upos = "AUX"
	CCONJ Upos = "CCONJ"
	DET   Upos = "DET"
	INTJ  Upos = "INTJ"
	NOUN  Upos = "NOUN"
	NUM   Upos = "NUM"
	PART  Upos = "PART"
	PRON  Upos = "PRON"
	PROPN Upos = "PROPN"
	PUNCT Upos = "PUNCT"
	SCONJ Upos = "SCONJ"
	SYM   Upos = "SYM"
	VERB  Upos = "VERB"
	X     Upos = "X"
)

// String returns the UPOS tag string.
func (u Upos) String() string { return string(u) }

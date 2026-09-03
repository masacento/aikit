package ner

import "fmt"

// TokenEntity is one BIO-decoded entity, located by BYTE offset into the text
// passed to Predict, so text[Start:End] slices the original string directly.
// Label is the entity type with its B-/I- prefix stripped. Score is the mean
// entity-role probability mass over the entity's pieces.
type TokenEntity struct {
	Text  string
	Label string
	Start int
	End   int
	Score float64
}

// TokenOpts tunes token-classification inference. The zero value is the
// reference configuration: argmax decoding over the trunk's full position
// table with stride 128 and no entity threshold.
type TokenOpts struct {
	// Tau switches to threshold mode when > 0: entities are kept only when
	// their span confidence is >= Tau. Zero (or negative) keeps every decoded
	// entity — the reference "argmax" mode. The reference CLI default is 0.99.
	Tau float64
	// MaxLength is the window's full sequence capacity including [CLS]/[SEP].
	// Zero means the trunk's position-table size. It is an error to exceed that:
	// positions past the table cannot be represented.
	MaxLength int
	// Stride is the overlap between consecutive windows. Zero means 128.
	Stride int
}

// Piece is one WordPiece position after windowing and first-window-wins dedup:
// the raw material the BIO decode consumes. It is exported so parity gates and
// callers with custom decoding rules can inspect predictions without rerunning
// the model.
type Piece struct {
	Start, End int     // byte offsets into the input text
	Label      int     // argmax label id (config.json id2label)
	PEntity    float64 // entity-role probability mass P(B)+P(I)
}

func (o TokenOpts) window(trunkMax int) (maxLen, stride int, err error) {
	maxLen = o.MaxLength
	if maxLen == 0 {
		maxLen = trunkMax
	}
	stride = o.Stride
	if stride == 0 {
		stride = 128
	}
	switch {
	case maxLen < 3:
		return 0, 0, fmt.Errorf("ner: MaxLength %d leaves no room for text between [CLS]/[SEP]", maxLen)
	case maxLen > trunkMax:
		return 0, 0, fmt.Errorf("ner: MaxLength %d exceeds the trunk's position table (%d)", maxLen, trunkMax)
	case stride < 1:
		return 0, 0, fmt.Errorf("ner: Stride %d < 1", stride)
	}
	return maxLen, stride, nil
}

// decode is the reference's lenient BIO decode: walk position-sorted pieces,
// open an entity on B (or on I with nothing open), extend on a matching I, and
// close on O. A mismatched I starts another entity and counts as an invalid
// transition. Thresholding applies to the whole entity, never a prefix.
func (m *TokenClassifier) decode(text string, pieces []Piece, tau float64) []TokenEntity {
	type ent struct {
		start, end int
		pSum       float64
		n          int
		label      string
	}
	var out []TokenEntity
	closeCur := func(cur *ent) {
		if cur == nil {
			return
		}
		score := cur.pSum / float64(cur.n)
		if tau <= 0 || score >= tau {
			out = append(out, TokenEntity{
				Text:  text[cur.start:cur.end],
				Label: cur.label,
				Start: cur.start, End: cur.end,
				Score: score,
			})
		}
	}
	var cur *ent
	for _, p := range pieces {
		switch m.role[p.Label] {
		case 'B':
			closeCur(cur)
			cur = &ent{start: p.Start, end: p.End, pSum: p.PEntity, n: 1, label: entityType(m.labels[p.Label])}
		case 'I':
			label := entityType(m.labels[p.Label])
			if cur == nil || cur.label != label {
				closeCur(cur)
				cur = &ent{start: p.Start, end: p.End, pSum: p.PEntity, n: 1, label: label}
			} else {
				cur.end = p.End
				cur.pSum += p.PEntity
				cur.n++
			}
		default:
			closeCur(cur)
			cur = nil
		}
	}
	closeCur(cur)
	return out
}

// entityType strips the BIO prefix off a label name ("B-SECRET" → "SECRET").
func entityType(label string) string { return label[2:] }

// countInvalid is the decode's invalid-transition counter: an I with no open
// entity or whose type differs from the open entity. Both cases start a new
// lenient entity and are reported per document.
func countInvalid(pieces []Piece, labels []string, role []byte) int {
	invalid, openType := 0, ""
	for _, p := range pieces {
		switch role[p.Label] {
		case 'B':
			openType = entityType(labels[p.Label])
		case 'I':
			typ := entityType(labels[p.Label])
			if openType != typ {
				invalid++
			}
			openType = typ
		default:
			openType = ""
		}
	}
	return invalid
}

package ner

// tokenclass_distilbert.go — token classification over raw text: a DistilBERT-family
// ForTokenClassification checkpoint (BERT-style WordPiece + BIO labels)
// mapped back to entity spans of the ORIGINAL input.
//
// The reference contract is the HF token-classification pipeline as
// implemented by the model card's span_infer.py (the pinned checkpoint is
// AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs, a DistilBERT secret
// detector, but nothing here is secret-specific — any id2label whose labels
// partition into O and B-*/I- roles works):
//
//   - the fast tokenizer's offset mapping locates every WordPiece in the raw
//     text (EncodeOffsets — nothing is re-tokenized or pre-split);
//   - long inputs run through manual overflow windows of (MaxLength−2) pieces
//     with `Stride` overlap; a character position seen in more than one window
//     keeps its FIRST window's prediction (HF's return_overflowing_tokens
//     chains unreliably past two windows, which is why the reference windows
//     by hand — Go mirrors that by construction);
//   - entities come from lenient BIO decoding (an I with no open entity opens
//     one; invalid transitions are counted, not fatal);
//   - an entity's score is the mean over its pieces of the entity-role
//     probability mass (P(B)+P(I)), so threshold mode (TokenOpts.Tau) filters
//     whole entities by span confidence.
//
// Labels come from the checkpoint's own config.json id2label (0=O,
// 1=B-SECRET, 2=I-SECRET for the pinned model); the code only requires that
// they partition into an O role and B-/I- prefixed entity roles, and reports
// entities with the type name (the label minus its B-/I- prefix).
//
// Entity offsets are BYTE offsets into the Go input string — the same
// convention as Entity, and the same divergence from the Python reference's
// character offsets (identical on ASCII).

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
)

// TokenClassifier is a loaded token-classification model. Immutable after
// load; Predict and Pieces are read-only-safe for concurrent use.
type TokenClassifier struct {
	trunk *encoder.DistilBERT
	tok   *embed.Tokenizer

	clsID, sepID int32

	clsW []float32 // classifier weight [nLabels, hidden], PyTorch [out, in]
	clsB []float32 // classifier bias [nLabels]

	labels []string // id2label values by id
	role   []byte   // role per label id: 'O', 'B', or 'I'
}

// LoadTokenClassifier loads a DistilBertForTokenClassification checkpoint
// from dir. It expects the published layout: config.json (with id2label),
// model.safetensors (DistilBERT trunk + classifier.*), and tokenizer.json.
// The trunk must be a DistilBERT (encoder.LoadDistilBERT owns that contract);
// a plain-BERT ForTokenClassification checkpoint is a different loader.
func LoadTokenClassifier(dir string) (*TokenClassifier, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("ner: read config: %w", err)
	}
	var cfg struct {
		ID2Label map[string]string `json:"id2label"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("ner: parse config: %w", err)
	}
	n := len(cfg.ID2Label)
	if n == 0 {
		return nil, fmt.Errorf("ner: config.json has no id2label")
	}
	labels := make([]string, n)
	role := make([]byte, n)
	var nB, nI bool
	for k, v := range cfg.ID2Label {
		var id int
		if _, err := fmt.Sscanf(k, "%d", &id); err != nil || id < 0 || id >= n {
			return nil, fmt.Errorf("ner: id2label key %q is not a label index", k)
		}
		labels[id] = v
		switch {
		case v == "O":
			role[id] = 'O'
		case len(v) > 2 && v[:2] == "B-":
			role[id] = 'B'
			nB = true
		case len(v) > 2 && v[:2] == "I-":
			role[id] = 'I'
			nI = true
		default:
			return nil, fmt.Errorf("ner: id2label[%s]=%q is neither O nor B-*/I-*", k, v)
		}
	}
	if !nB || !nI {
		return nil, fmt.Errorf("ner: id2label lacks a B-*/I-* label pair: %v", cfg.ID2Label)
	}

	trunk, err := encoder.LoadDistilBERT(dir)
	if err != nil {
		return nil, fmt.Errorf("ner: load trunk: %w", err)
	}

	// classifier.* is the token-classification head — [nLabels, hidden] weight
	// in the PyTorch [out, in] layout, applied to every position.
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, errors.Join(fmt.Errorf("ner: open safetensors: %w", err), trunk.Close())
	}
	defer func() { _ = st.Close() }()
	D := trunk.HiddenDim()
	w, err := st.TensorF32("classifier.weight", n, D)
	if err != nil {
		return nil, errors.Join(fmt.Errorf("ner: load classifier.weight: %w", err), trunk.Close())
	}
	b, err := st.TensorF32("classifier.bias", n)
	if err != nil {
		return nil, errors.Join(fmt.Errorf("ner: load classifier.bias: %w", err), trunk.Close())
	}
	// The head is small and the mapping closes on return, so copy out of the
	// mmap rather than hand the caller views into it.
	clsW := make([]float32, len(w))
	copy(clsW, w)
	clsB := make([]float32, len(b))
	copy(clsB, b)

	tok, err := embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, errors.Join(fmt.Errorf("ner: load tokenizer: %w", err), trunk.Close())
	}
	prefix, suffix := tok.TemplateSpecials()
	if len(prefix) != 1 || len(suffix) != 1 {
		return nil, errors.Join(fmt.Errorf("ner: tokenizer wraps sequences with %d/%d specials, want "+
			"exactly [CLS] … [SEP]", len(prefix), len(suffix)), trunk.Close())
	}
	return &TokenClassifier{
		trunk: trunk, tok: tok,
		clsID: prefix[0], sepID: suffix[0],
		clsW: clsW, clsB: clsB,
		labels: labels, role: role,
	}, nil
}

// Close releases the trunk's mmap. Idempotent.
func (m *TokenClassifier) Close() error { return m.trunk.Close() }

// Labels reports the checkpoint's id2label table in id order.
func (m *TokenClassifier) Labels() []string {
	out := make([]string, len(m.labels))
	copy(out, m.labels)
	return out
}

// Predict extracts BIO-decoded entities from text — the end-to-end equivalent
// of the reference's infer_spans. Results are in byte-offset order (decode
// order).
func (m *TokenClassifier) Predict(text string, opts TokenOpts) ([]TokenEntity, error) {
	pieces, _, err := m.Pieces(text, opts)
	if err != nil {
		return nil, err
	}
	return m.decode(text, pieces, opts.Tau), nil
}

// Pieces runs the model over every window of text and returns the per-piece
// predictions AFTER first-window-wins dedup, sorted by position — the decode
// input. The second return is the count of lenient-BIO invalid transitions
// decode tolerates (an I with no open entity), reported because the reference
// reports it per document.
func (m *TokenClassifier) Pieces(text string, opts TokenOpts) ([]Piece, int, error) {
	maxLen, stride, err := opts.window(m.trunk.MaxSeqLength())
	if err != nil {
		return nil, 0, err
	}
	full, offs, err := m.tok.EncodeOffsets(text)
	if err != nil {
		return nil, 0, fmt.Errorf("ner: tokenize: %w", err)
	}
	n := len(full)
	body := maxLen - 2
	step := body - stride
	if step < 1 {
		step = 1
	}

	var pieces []Piece
	seen := make(map[[2]int]bool)
	for w0 := 0; ; w0 += step {
		w1 := min(w0+body, n)
		p, err := m.window(full[w0:w1], offs[w0:w1], seen)
		if err != nil {
			return nil, 0, err
		}
		pieces = append(pieces, p...)
		if w1 >= n {
			break
		}
	}
	sort.Slice(pieces, func(i, j int) bool {
		a, b := pieces[i], pieces[j]
		return a.Start < b.Start || (a.Start == b.Start && a.End < b.End)
	})
	return pieces, countInvalid(pieces, m.labels, m.role), nil
}

// window runs one [CLS] … [SEP]-wrapped overflow window and returns its
// not-yet-seen pieces. A position's span (s, e) is the dedup key — the exact
// value the reference's `seen` set holds — so a character position covered by
// two windows keeps the earlier window's prediction.
func (m *TokenClassifier) window(ids []int32, offs [][2]int, seen map[[2]int]bool) ([]Piece, error) {
	seq := make([]int32, 0, len(ids)+2)
	seq = append(seq, m.clsID)
	seq = append(seq, ids...)
	seq = append(seq, m.sepID)

	hidden := m.trunk.HiddenStates(seq)
	D := m.trunk.HiddenDim()

	var out []Piece
	for k := range ids {
		s, e := offs[k][0], offs[k][1]
		if s == e || seen[[2]int{s, e}] {
			continue
		}
		seen[[2]int{s, e}] = true
		label, pEntity := m.classify(hidden[(k+1)*D : (k+2)*D])
		out = append(out, Piece{Start: s, End: e, Label: label, PEntity: pEntity})
	}
	return out, nil
}

// classify applies the classifier head to one hidden state and softmaxes:
// the argmax label id and the entity-role probability mass P(B)+P(I).
func (m *TokenClassifier) classify(h []float32) (int, float64) {
	n := len(m.clsB)
	logits := make([]float64, n)
	best, argmax := math.Inf(-1), 0
	for c := 0; c < n; c++ {
		row := m.clsW[c*m.trunk.HiddenDim() : (c+1)*m.trunk.HiddenDim()]
		var z float64
		for j, v := range row {
			z += float64(v) * float64(h[j])
		}
		z += float64(m.clsB[c])
		logits[c] = z
		if z > best {
			best, argmax = z, c
		}
	}
	mx := math.Inf(-1)
	for _, z := range logits {
		if z > mx {
			mx = z
		}
	}
	var sum, pEntity float64
	for c, z := range logits {
		p := math.Exp(z - mx)
		logits[c] = p
		sum += p
		if m.role[c] == 'B' || m.role[c] == 'I' {
			pEntity += p
		}
	}
	return argmax, pEntity / sum
}

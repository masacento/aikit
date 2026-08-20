package encoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/sparse"
)

// splade.go — SPLADE in-process query/document expansion (roadmap §2.3). A SPLADE
// model is a BERT encoder (LoadBERT) plus a masked-LM head: the expansion projects
// each token's hidden state to vocabulary logits, applies log(1+ReLU), and max-pools
// over the sequence into a sparse vector over the vocab. The result drops straight
// into the sparse package's inverted index — closing the loop so learned-sparse
// retrieval runs end-to-end in-process, no Python at query time.

// SPLADE is a loaded SPLADE model: a BERT encoder plus the BertForMaskedLM head.
type SPLADE struct {
	bert       *BERT
	transformW []float32 // cls.predictions.transform.dense.weight [hidden, hidden]
	transformB []float32 // [hidden]
	transLNW   []float32 // cls.predictions.transform.LayerNorm.weight [hidden]
	transLNB   []float32
	decoderW   []float32 // cls.predictions.decoder.weight [vocab, hidden] (tied to word emb)
	decoderB   []float32 // cls.predictions.bias [vocab]
	vocab      int
}

// LoadSPLADE loads a BertForMaskedLM SPLADE model (config.json + model.safetensors)
// from dir: the BERT encoder (via LoadBERT) plus the masked-LM head.
func LoadSPLADE(dir string) (*SPLADE, error) {
	b, err := LoadBERT(dir)
	if err != nil {
		return nil, err
	}
	D, V := b.cfg.Hidden, b.cfg.VocabSize
	s := &SPLADE{bert: b, vocab: V}

	var e error
	get := func(name string, want ...int) []float32 {
		if e != nil {
			return nil
		}
		var v []float32
		v, e = loadF32(b.st, name, want)
		return v
	}
	s.transformW = get("cls.predictions.transform.dense.weight", D, D)
	s.transformB = get("cls.predictions.transform.dense.bias", D)
	s.transLNW = get("cls.predictions.transform.LayerNorm.weight", D)
	s.transLNB = get("cls.predictions.transform.LayerNorm.bias", D)
	s.decoderB = get("cls.predictions.bias", V)
	if e != nil {
		_ = b.st.Close()
		return nil, fmt.Errorf("encoder: SPLADE MLM head: %w", e)
	}
	// The decoder is usually weight-tied to the word embeddings; load the tensor if
	// the checkpoint stores it, otherwise reuse the already-loaded embeddings.
	if dw, derr := loadF32(b.st, "cls.predictions.decoder.weight", []int{V, D}); derr == nil {
		s.decoderW = dw
	} else {
		s.decoderW = b.wordEmb
	}
	return s, nil
}

// Close releases the underlying BERT's mmap-backed weights. Idempotent.
func (s *SPLADE) Close() error {
	if s.bert == nil {
		return nil
	}
	return s.bert.Close()
}

// Expand runs the SPLADE expansion for text and returns the sparse term-weight
// vector over the model vocabulary (only positive weights — the natural SPLADE
// sparsity). Feed it to sparse.New (documents) or sparse.Index.Query (queries).

func (s *SPLADE) Expand(text string) (sparse.SparseVec, error) {
	ids, err := s.bert.tok.EncodeWithSpecials(text, s.bert.maxSeq)
	if err != nil {
		return sparse.SparseVec{}, err
	}
	return s.expandIDs(ids), nil
}

func (s *SPLADE) expandIDs(ids []int32) sparse.SparseVec {
	D, V, L := s.bert.cfg.Hidden, s.vocab, len(ids)
	h := s.bert.hiddenStates(ids, nil) // [L, D]

	// MLM transform head: t = LayerNorm(gelu(h·Wᵀ + b)).
	t := matmulBT(h, s.transformW, L, D, D)
	addBias(t, s.transformB, L, D)
	gelu(t)
	layerNorm(t, s.transLNW, s.transLNB, L, D, s.bert.cfg.LNEps)

	// Vocabulary logits = t · decoderWᵀ + decoderB → [L, V].
	//
	// Fanned across the V columns, not the L rows: L is a query length here (a
	// handful of tokens is normal), so the trunk's row split has nothing to
	// split, while V=30522 always does. Bit-identical to the serial fill — see
	// matmulBTColsInto.
	logits := make([]float32, L*V)
	matmulBTColsInto(t, s.decoderW, logits, L, D, V)
	addBias(logits, s.decoderB, L, V)

	// SPLADE pooling: max over tokens of log(1 + relu(logit)).
	//
	// The log1p is applied ONCE PER VOCAB ENTRY, after the max — not once per
	// positive element of the [L,V] logit matrix. That is exact, not an
	// approximation: float32∘Log1p∘relu is monotone non-decreasing and maps 0→0, so
	// max_i f(x_i) == f(max_i x_i), including the f32 narrowing, the zero
	// initialisation, and the negative-only column (which stays 0 and is excluded by
	// the `w > 0` filter below either way). NaN is skipped by `x > 0` in both forms.
	//
	// At L=512, V=30522 the old shape called math.Log1p once per positive logit —
	// millions of scalar f64 calls per Expand; this makes it V (perf-campaign
	// item 2). The saving scales with positive density, so it is largest on the
	// dense-logit models and smaller on a trained SPLADE, whose logits are mostly
	// negative by construction.
	pooled := make([]float32, V)
	for i := range L {
		row := logits[i*V : (i+1)*V]
		for v, x := range row {
			if x > pooled[v] {
				pooled[v] = x // raw max; log1p applied below
			}
		}
	}
	log1pPoolInto(pooled)
	var out sparse.SparseVec
	for v, w := range pooled {
		if w > 0 {
			out.Terms = append(out.Terms, uint32(v))
			out.Weights = append(out.Weights, w)
		}
	}
	return out
}

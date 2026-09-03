package encoder

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// modernbert_crossencoder.go — a ModernBERT cross-encoder reranker, e.g.
// cross-encoder/ettin-reranker-17m-v1. Same job as crossencoder.go (score a
// (query, document) PAIR jointly rather than embedding each side), but neither
// half of that file is reusable:
//
//   - The TRUNK is ModernBERT, not BERT. There is no token_type embedding, so a
//     pair is framed [CLS] q [SEP] d [SEP] with NO segment ids at all — HF's own
//     post_processor gives every element type_id 0.
//   - The HEAD is not BertForSequenceClassification's pooler+classifier. These
//     checkpoints declare `architectures: [ModernBertModel]` and carry the head as
//     a sentence-transformers MODULE CHAIN in sibling directories:
//
//     1_Pooling      cls
//     2_Dense        [D,D] linear, no bias, then GELU
//     3_LayerNorm    weight + bias, [D]
//     4_Dense        [labels,D] linear + bias, identity activation
//
//     which is ModernBertForSequenceClassification's head.dense → gelu → head.norm
//     → classifier, sharded across four files. Fetching model.safetensors alone
//     gets a trunk with no head at all.
//
//	score = ( LayerNorm( gelu( h[CLS]·W2ᵀ ) ) )·W4ᵀ + b4

// mbceModules is the module chain this loader implements, in order. The chain is
// validated against modules.json rather than assumed from the directory names: a
// checkpoint with an extra or reordered module is a different computation, and
// scoring it with this forward would produce plausible-looking numbers.
var mbceModules = [...]string{"Transformer", "Pooling", "Dense", "LayerNorm", "Dense"}

// ModernBERTCrossEncoder is a loaded ModernBERT cross-encoder reranker.
type ModernBERTCrossEncoder struct {
	mb *ModernBERT
	// tok is the concrete tokenizer, not ModernBERT.tok's narrow interface: pair
	// framing needs Encode (bare ids) and SpecialID on top of EncodeWithSpecials.
	tok *embed.Tokenizer

	denseW  []float32 // 2_Dense linear.weight [hidden, hidden] (no bias)
	normW   []float32 // 3_LayerNorm norm.weight [hidden]
	normB   []float32 // 3_LayerNorm norm.bias [hidden]
	classW  []float32 // 4_Dense linear.weight [labels, hidden]
	classB  []float32 // 4_Dense linear.bias [labels]
	labels  int
	normEps float64
}

// LoadModernBERTCrossEncoder loads a sentence-transformers ModernBERT cross-encoder
// from dir: the ModernBERT trunk (via LoadModernBERT) plus the 2_Dense / 3_LayerNorm
// / 4_Dense head modules. The number of labels is read from the final Dense's shape
// (1 for a relevance reranker).
func LoadModernBERTCrossEncoder(dir string) (*ModernBERTCrossEncoder, error) {
	if err := checkModuleChain(dir); err != nil {
		return nil, err
	}
	mb, err := LoadModernBERT(dir)
	if err != nil {
		return nil, err
	}
	// abort releases the already-mapped trunk and returns the load error, joined
	// with the release error if the munmap itself failed (both are real; neither
	// should mask the other).
	abort := func(err error) (*ModernBERTCrossEncoder, error) {
		return nil, errors.Join(err, mb.Close())
	}
	// Pooling is not a free parameter here: the head reads one row, and reading the
	// mean of the sequence instead of [CLS] silently rescales every score. The trunk
	// loader takes this from 1_Pooling/config.json.
	if mb.pool != poolCLS {
		return abort(fmt.Errorf("encoder: ModernBERT cross-encoder pooling %q unsupported (cls only)", mb.pool))
	}

	ce := &ModernBERTCrossEncoder{mb: mb, normEps: mb.cfg.NormEps}

	// pairIDs frames [CLS] q [SEP] d [SEP]; without those specials the tokenizer
	// silently yields id 0 for them and the pair is malformed. Require a tokenizer
	// that has both, at load rather than per mis-scored pair. Same contract as
	// LoadCrossEncoder.
	// The trunk already loaded this file. Reuse its tokenizer when it is the
	// concrete type pair framing needs — it holds a ~50k-entry vocab and a ~50k
	// merge table, ~6 MiB, and loading a second copy doubles the model's heap for
	// nothing. It is immutable after load, so sharing is safe.
	if shared, ok := mb.tok.(*embed.Tokenizer); ok {
		ce.tok = shared
	} else if ce.tok, err = embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json")); err != nil {
		return abort(fmt.Errorf("encoder: ModernBERT cross-encoder tokenizer: %w", err))
	}
	for _, lit := range [...]string{"[CLS]", "[SEP]"} {
		if _, ok := ce.tok.SpecialID(lit); !ok {
			return abort(fmt.Errorf("encoder: ModernBERT cross-encoder tokenizer has no %s token", lit))
		}
	}

	D := mb.cfg.Hidden
	// The head modules are ~0.3 MB total, so they are read out and the files closed
	// immediately rather than kept mapped for the model's lifetime like the trunk.
	if ce.denseW, err = headTensor(dir, "2_Dense", "linear.weight", D, D); err != nil {
		return abort(err)
	}
	if ce.normW, err = headTensor(dir, "3_LayerNorm", "norm.weight", D); err != nil {
		return abort(err)
	}
	if ce.normB, err = headTensor(dir, "3_LayerNorm", "norm.bias", D); err != nil {
		return abort(err)
	}
	labels, err := headRows(dir, "4_Dense", "linear.weight", D)
	if err != nil {
		return abort(err)
	}
	ce.labels = labels
	if ce.classW, err = headTensor(dir, "4_Dense", "linear.weight", labels, D); err != nil {
		return abort(err)
	}
	if ce.classB, err = headTensor(dir, "4_Dense", "linear.bias", labels); err != nil {
		return abort(err)
	}
	return ce, nil
}

// checkModuleChain validates dir/modules.json against mbceModules and the two Dense
// activations. The module TYPES are what this forward implements; the directory
// names are only where the weights happen to live. An absent modules.json is
// accepted (the directory layout is then the only evidence available), but a present
// one that disagrees is a hard error.
func checkModuleChain(dir string) error {
	raw, err := os.ReadFile(filepath.Join(dir, "modules.json"))
	if err != nil {
		return nil
	}
	var mods []struct {
		Path string `json:"path"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &mods); err != nil {
		return fmt.Errorf("encoder: parse modules.json: %w", err)
	}
	if len(mods) != len(mbceModules) {
		return fmt.Errorf("encoder: ModernBERT cross-encoder has %d modules, want %d (%v)",
			len(mods), len(mbceModules), mbceModules)
	}
	for i, m := range mods {
		// The type is a dotted Python path; only the leaf class matters.
		leaf := m.Type
		if j := lastDot(leaf); j >= 0 {
			leaf = leaf[j+1:]
		}
		if leaf != mbceModules[i] {
			return fmt.Errorf("encoder: ModernBERT cross-encoder module[%d] is %q, want %s", i, leaf, mbceModules[i])
		}
	}
	// The activations are declared per Dense module, not implied by position:
	// 2_Dense is GELU and 4_Dense is the identity. A checkpoint that puts, say, a
	// Tanh on the second would score through this forward without complaint.
	for _, want := range [...]struct {
		path string
		act  string
		bias bool
	}{
		{"2_Dense", "GELU", false},
		{"4_Dense", "Identity", true},
	} {
		raw, err := os.ReadFile(filepath.Join(dir, want.path, "config.json"))
		if err != nil {
			return fmt.Errorf("encoder: read %s/config.json: %w", want.path, err)
		}
		var dc struct {
			Activation string `json:"activation_function"`
			Bias       bool   `json:"bias"`
		}
		if err := json.Unmarshal(raw, &dc); err != nil {
			return fmt.Errorf("encoder: parse %s/config.json: %w", want.path, err)
		}
		leaf := dc.Activation
		if j := lastDot(leaf); j >= 0 {
			leaf = leaf[j+1:]
		}
		if leaf != want.act {
			return fmt.Errorf("encoder: %s activation %q unsupported (%s only)", want.path, leaf, want.act)
		}
		if dc.Bias != want.bias {
			return fmt.Errorf("encoder: %s bias=%t unsupported (want %t)", want.path, dc.Bias, want.bias)
		}
	}
	return nil
}

func lastDot(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '.' {
			return i
		}
	}
	return -1
}

// headTensor reads one shape-checked f32 tensor out of a head module's own
// safetensors file and closes it again.
func headTensor(dir, module, name string, want ...int) ([]float32, error) {
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, module, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open %s: %w", module, err)
	}
	defer st.Close()
	v, err := st.TensorF32(name, want...)
	if err != nil {
		return nil, fmt.Errorf("encoder: %s/%s: %w", module, name, err)
	}
	// The view aliases the mmap this function is about to unmap.
	return cloneFloat32(v), nil
}

// headRows reports a head tensor's row count, so the label count can be read off
// the checkpoint instead of assumed to be 1.
func headRows(dir, module, name string, wantCols int) (int, error) {
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, module, "model.safetensors"))
	if err != nil {
		return 0, fmt.Errorf("encoder: open %s: %w", module, err)
	}
	defer st.Close()
	t, err := st.Tensor(name)
	if err != nil {
		return 0, fmt.Errorf("encoder: %s/%s: %w", module, name, err)
	}
	if len(t.Shape) != 2 || t.Shape[1] != wantCols {
		return 0, fmt.Errorf("encoder: %s/%s shape %v (want [labels,%d])", module, name, t.Shape, wantCols)
	}
	if t.Shape[0] < 1 {
		return 0, fmt.Errorf("encoder: %s/%s must have ≥1 label, got %d", module, name, t.Shape[0])
	}
	return t.Shape[0], nil
}

// Close releases the trunk's mmap-backed weights. Idempotent.
func (ce *ModernBERTCrossEncoder) Close() error {
	if ce.mb == nil {
		return nil
	}
	return ce.mb.Close()
}

// Labels is the number of classification outputs (1 for a relevance reranker).
func (ce *ModernBERTCrossEncoder) Labels() int { return ce.labels }

// Score returns the relevance logit for a (query, document) pair — higher is more
// relevant. Rank a candidate list by descending Score to rerank. (For a model with
// more than one label, this is label 0; use ScoreAll for the rest.)
func (ce *ModernBERTCrossEncoder) Score(query, doc string) (float32, error) {
	all, err := ce.ScoreAll(query, doc)
	if err != nil {
		return 0, err
	}
	return all[0], nil
}

// ScoreAll returns every classification logit for the pair (length = Labels).
func (ce *ModernBERTCrossEncoder) ScoreAll(query, doc string) ([]float32, error) {
	return ce.scoreIDs(ce.pairIDs(query, doc)), nil
}

// ScoreBatch scores one query against many documents, returning label 0's logit per
// document in the caller's order. concurrency <= 0 means NumCPU.
//
// Structurally identical to CrossEncoder.ScoreBatch, and for the reasons documented
// there: document-level parallelism beats the intra-op parallelism a Score loop
// gets, the query is tokenized once, and documents are dispatched LONGEST FIRST
// because cost is roughly linear in pair length and there is no padded batch kernel,
// leaving makespan as the only scheduling question.
func (ce *ModernBERTCrossEncoder) ScoreBatch(query string, docs []string, concurrency int) ([]float32, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	concurrency = min(concurrency, len(docs))

	qIDs := ce.tok.Encode(query)
	pairs := make([][]int32, len(docs))
	for i, d := range docs {
		pairs[i] = ce.pairIDsFrom(qIDs, d)
	}
	order := make([]int, len(pairs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return len(pairs[order[a]]) > len(pairs[order[b]])
	})

	out := make([]float32, len(docs))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			for {
				k := int(next.Add(1)) - 1
				if k >= len(order) {
					return
				}
				i := order[k]
				out[i] = ce.scoreIDs(pairs[i])[0]
			}
		})
	}
	wg.Wait()
	return out, nil
}

// pairIDs builds [CLS] query [SEP] document [SEP], right-truncating to the model's
// max sequence length. No token-type segments: ModernBERT has no token_type
// embedding and this family's post_processor gives every element type_id 0.
func (ce *ModernBERTCrossEncoder) pairIDs(query, doc string) []int32 {
	return ce.pairIDsFrom(ce.tok.Encode(query), doc)
}

// pairIDsFrom is pairIDs with the query already tokenized, so a batch shares one
// encode. It is the single implementation; pairIDs is the one-shot wrapper.
func (ce *ModernBERTCrossEncoder) pairIDsFrom(qTok []int32, doc string) []int32 {
	cls, _ := ce.tok.SpecialID("[CLS]")
	sep, _ := ce.tok.SpecialID("[SEP]")
	// Copy before trimming: every pair in a batch shares qTok, and the trim below
	// reslices it.
	q := append([]int32(nil), qTok...)
	d := ce.tok.Encode(doc)

	avail := max(ce.mb.maxSeq-3, 0) // room for [CLS] + 2×[SEP]
	// longest_first, which is what this checkpoint's tokenizer.json declares
	// (truncation.strategy "LongestFirst"): trim the currently-longer sequence one
	// token at a time until the pair fits, ties trimming the doc. Giving the query
	// the whole budget first would let a query ≥ avail starve the document to zero
	// tokens, making the score document-independent.
	for len(q)+len(d) > avail {
		if len(q) > len(d) {
			q = q[:len(q)-1]
		} else {
			d = d[:len(d)-1]
		}
	}

	ids := make([]int32, 0, len(q)+len(d)+3)
	ids = append(ids, cls)
	ids = append(ids, q...)
	ids = append(ids, sep)
	ids = append(ids, d...)
	ids = append(ids, sep)
	return ids
}

// scoreIDs runs the trunk on a pre-assembled pair and applies the module-chain head
// to the [CLS] row: classifier(LayerNorm(gelu(dense(h[CLS])))).
func (ce *ModernBERTCrossEncoder) scoreIDs(ids []int32) []float32 {
	D := ce.mb.cfg.Hidden
	h := ce.mb.forward(ids)
	cls := h[:D] // 1_Pooling mode "cls" — row 0, checked at load

	x := matmulBT(cls, ce.denseW, 1, D, D) // 2_Dense (no bias)
	// Exact (erf) GELU, matching torch.nn.GELU's default — NOT the tanh
	// approximation. On a [1,D] row the cost is irrelevant either way; the two
	// differ by ~1e-3 at the knee, which is above this head's parity tolerance.
	linalg.GELUInto(x, x)
	layerNorm(x, ce.normW, ce.normB, 1, D, ce.normEps) // 3_LayerNorm (with bias)
	out := matmulBT(x, ce.classW, 1, D, ce.labels)     // 4_Dense
	addBias(out, ce.classB, 1, ce.labels)
	return out
}

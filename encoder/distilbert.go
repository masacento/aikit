package encoder

// distilbert.go — DistilBERT: the 6-layer distillation of BERT (Sanh et al.
// 2019), loaded here as a forward-only token-classification trunk. It reuses
// bert.go's transformer wholesale — same learned absolute positions, GELU FFN,
// post-norm — through three differences the loader normalizes away:
//
//   - config.json keys are DistilBERT's own (dim / hidden_dim / n_layers /
//     n_heads), not BERT's hidden_size / intermediate_size / num_hidden_layers;
//   - tensor names are distilbert.transformer.layer.N.{attention.q_lin,
//     sa_layer_norm, ffn.lin1, …} rather than bert.encoder.layer.N.…;
//   - there is no token_type embedding at all (typeVocab 0 — the BERT forward
//     already treats that as absent) and no pooler (irrelevant: this trunk
//     exposes HiddenStates only).
//
// The token-classification head (classifier.*) is NOT loaded here — it belongs
// to the consumer (ner's SecretMasker), mirroring how GLiNER's head lives in
// ner while its DeBERTa trunk lives here.
//
// Parity is pinned through ner's SecretMasker golden (scripts/
// pin_tokenclassification.py); a downstream span error that smells like the trunk can
// be isolated with TestDistilBERT_smoke + the golden's hidden-state block.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
)

// distilbertConfig is the checkpoint's config.json, in DistilBERT's own key
// names. HF hardcodes eps at 1e-12 (the config carries no layer_norm_eps), so
// the mapped bertConfig pins that value.
type distilbertConfig struct {
	VocabSize  int    `json:"vocab_size"`
	Dim        int    `json:"dim"`
	HiddenDim  int    `json:"hidden_dim"`
	NLayers    int    `json:"n_layers"`
	NHeads     int    `json:"n_heads"`
	MaxPos     int    `json:"max_position_embeddings"`
	Activation string `json:"activation"`
	Sinusoidal bool   `json:"sinusoidal_pos_embds"`
	PadTokenID int    `json:"pad_token_id"`
	ModelType  string `json:"model_type"`
}

// DistilBERT is a loaded DistilBERT trunk. Immutable after load; HiddenStates
// is read-only-safe for concurrent use (the shared bert.go forward guarantees
// it).
type DistilBERT struct {
	trunk *BERT // the shared forward machinery, configured from the mapped config
}

// LoadDistilBERT loads a DistilBERT checkpoint (config.json + model.safetensors
// with DistilBERT tensor names) from dir. It validates the two architecture
// assumptions the shared forward makes: GELU activation and learned (not
// sinusoidal) absolute positions.
func LoadDistilBERT(dir string) (*DistilBERT, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("encoder: read DistilBERT config: %w", err)
	}
	var c distilbertConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("encoder: parse DistilBERT config: %w", err)
	}
	switch {
	case c.ModelType != "" && c.ModelType != "distilbert":
		return nil, fmt.Errorf("encoder: model_type=%q, want distilbert", c.ModelType)
	case c.Activation != "gelu":
		return nil, fmt.Errorf("encoder: activation=%q unsupported (gelu only)", c.Activation)
	case c.Sinusoidal:
		return nil, fmt.Errorf("encoder: sinusoidal_pos_embds=true unsupported (learned positions only)")
	case c.Dim == 0 || c.NHeads == 0 || c.NLayers == 0 || c.HiddenDim == 0:
		return nil, fmt.Errorf("encoder: DistilBERT config missing a required dim")
	case c.Dim%c.NHeads != 0:
		return nil, fmt.Errorf("encoder: dim %d not divisible by n_heads %d", c.Dim, c.NHeads)
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open DistilBERT safetensors: %w", err)
	}

	// Tensor-name prefix: save_pretrained of a *ForTokenClassification head
	// nests the trunk under "distilbert."; a bare DistilBertModel export leaves
	// it off. Same probe shape LoadBERT uses for "bert."/"roberta.".
	prefix := ""
	if _, e := st.Tensor("embeddings.word_embeddings.weight"); e != nil {
		if _, e2 := st.Tensor("distilbert.embeddings.word_embeddings.weight"); e2 != nil {
			_ = st.Close()
			return nil, fmt.Errorf("encoder: no embeddings.word_embeddings.weight under either naming scheme")
		}
		prefix = "distilbert."
	}

	D, I := c.Dim, c.HiddenDim
	b := &BERT{
		cfg: bertConfig{
			VocabSize: c.VocabSize, Hidden: D, Layers: c.NLayers, Heads: c.NHeads,
			Intermediate: I, MaxPos: c.MaxPos, LNEps: 1e-12, Act: c.Activation,
			ModelType: "distilbert", PadTokenID: c.PadTokenID,
		},
		st:     st,
		layers: make([]bertLayer, c.NLayers),
	}

	var ferr error
	get := func(name string, want ...int) []float32 {
		if ferr != nil {
			return nil
		}
		var v []float32
		v, ferr = loadF32(st, name, want)
		return v
	}
	b.wordEmb = get(prefix+"embeddings.word_embeddings.weight", c.VocabSize, D)
	b.posEmb = get(prefix+"embeddings.position_embeddings.weight", c.MaxPos, D)
	b.embLNW = get(prefix+"embeddings.LayerNorm.weight", D)
	b.embLNB = get(prefix+"embeddings.LayerNorm.bias", D)
	for i := range b.layers {
		p := fmt.Sprintf("%stransformer.layer.%d.", prefix, i)
		l := &b.layers[i]
		l.Wq, l.Bq = get(p+"attention.q_lin.weight", D, D), get(p+"attention.q_lin.bias", D)
		l.Wk, l.Bk = get(p+"attention.k_lin.weight", D, D), get(p+"attention.k_lin.bias", D)
		l.Wv, l.Bv = get(p+"attention.v_lin.weight", D, D), get(p+"attention.v_lin.bias", D)
		l.Wo, l.Bo = get(p+"attention.out_lin.weight", D, D), get(p+"attention.out_lin.bias", D)
		l.AttnLNW, l.AttnLNB = get(p+"sa_layer_norm.weight", D), get(p+"sa_layer_norm.bias", D)
		l.Wi, l.Bi = get(p+"ffn.lin1.weight", I, D), get(p+"ffn.lin1.bias", I)
		l.Wd, l.Bd = get(p+"ffn.lin2.weight", D, I), get(p+"ffn.lin2.bias", D)
		l.OutLNW, l.OutLNB = get(p+"output_layer_norm.weight", D), get(p+"output_layer_norm.bias", D)
	}
	if ferr != nil {
		_ = st.Close()
		return nil, fmt.Errorf("encoder: load DistilBERT weights: %w", ferr)
	}

	// The learned-position table is the hard ceiling (bert.go forward gathers
	// posEmb[i] directly); DistilBERT has no sentence_bert_config to lower it.
	b.maxSeq = c.MaxPos
	return &DistilBERT{trunk: b}, nil
}

// Close releases the mmap-backed weights. Idempotent.
func (d *DistilBERT) Close() error { return d.trunk.Close() }

// HiddenDim is the trunk width (768 for distilbert-base).
func (d *DistilBERT) HiddenDim() int { return d.trunk.cfg.Hidden }

// MaxSeqLength is the learned-position capacity (512) — the longest id sequence
// HiddenStates accepts before it truncates.
func (d *DistilBERT) MaxSeqLength() int { return d.trunk.maxSeq }

// HiddenStates runs the transformer on token ids (already wrapped
// [CLS]…[SEP], no segment ids — DistilBERT has none) and returns the last
// hidden state [L, hidden], row-major.
func (d *DistilBERT) HiddenStates(ids []int32) []float32 {
	return d.trunk.hiddenStates(ids, nil)
}

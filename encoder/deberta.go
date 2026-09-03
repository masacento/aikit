package encoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// deberta.go — the DeBERTa-v2/v3 encoder (microsoft/mdeberta-v3-base and the GLiNER
// models built on it). It is a post-norm BERT-family encoder, but its attention is
// DISENTANGLED: instead of adding a position embedding to the input, every layer
// adds two relative-position terms to the attention scores.
//
//	emb    = LayerNorm( word[ids] )                      // NO position, NO token_type
//	score  = (Q·Kᵀ)/√(d·3) + c2p + p2c                   // scale_factor = 1 + |pos_att_type|
//	h      = LayerNorm( h + attn·Woᵀ )                   // post-norm, eps 1e-7
//	h      = LayerNorm( h + Down(gelu(Up(h))) )          // post-norm
//
// with, per head (share_att_key ⇒ the position projections REUSE query_proj/key_proj,
// and rel_embeddings is LayerNorm'd once by encoder.LayerNorm):
//
//	b(i,j) = logBucket(i-j, buckets=256, maxPos=512)
//	c2p    = gather( Q · Kpos(rel)ᵀ , clamp(b(i,j)+256, 0, 511) )    / √(d·3)
//	p2c    = gather( K · Qpos(rel)ᵀ , clamp(-b(j,i)+256, 0, 511) )ᵀ  / √(d·3)
//
// Three of those details are the ones that fail SILENTLY — no crash, no NaN, just a
// quietly worse model — so each is called out at its site below: the p2c gather uses
// its own index built from the transposed offset (not c2p's index transposed), the
// scale is 1/√(3d) rather than 1/√d, and encoder.LayerNorm normalizes the RELATIVE
// embeddings rather than any hidden state.
//
// Parity is pinned per layer against mdeberta-v3-base (TestDeBERTa_parity,
// scripts/pin_deberta.py).

type debertaConfig struct {
	VocabSize            int             `json:"vocab_size"`
	Hidden               int             `json:"hidden_size"`
	Layers               int             `json:"num_hidden_layers"`
	Heads                int             `json:"num_attention_heads"`
	Intermediate         int             `json:"intermediate_size"`
	MaxPos               int             `json:"max_position_embeddings"`
	TypeVocab            int             `json:"type_vocab_size"`
	LNEps                float64         `json:"layer_norm_eps"`
	Act                  string          `json:"hidden_act"`
	ModelType            string          `json:"model_type"`
	RelativeAttention    bool            `json:"relative_attention"`
	PositionBuckets      int             `json:"position_buckets"`
	MaxRelativePositions int             `json:"max_relative_positions"`
	PosAttType           json.RawMessage `json:"pos_att_type"`
	ShareAttKey          bool            `json:"share_att_key"`
	NormRelEbd           string          `json:"norm_rel_ebd"`
	PositionBiasedInput  bool            `json:"position_biased_input"`
}

type debertaLayer struct {
	Wq, WqB          []float32 // query_proj [D,D] + [D]
	Wk, WkB          []float32 // key_proj
	Wv, WvB          []float32 // value_proj
	Wo, WoB          []float32 // attention.output.dense
	AttnLNW, AttnLNB []float32 // attention.output.LayerNorm (post-norm)
	Up, UpB          []float32 // intermediate.dense [I,D]
	Down, DownB      []float32 // output.dense [D,I]
	MLPLNW, MLPLNB   []float32 // output.LayerNorm (post-norm)

	// Kpos/Qpos are key_proj / query_proj applied to the (LayerNorm'd) relative
	// embeddings: [2*buckets, D]. HF recomputes these inside every attention call,
	// but they depend only on weights — no input, no length — so they are folded
	// into the load. Numerically identical, and it removes two [512,768]x[768,768]
	// matmuls per layer per forward.
	Kpos []float32
	Qpos []float32
}

// DeBERTa is a loaded DeBERTa-v2/v3 encoder. Immutable after load; the forward is
// read-only-safe for concurrent use.
type DeBERTa struct {
	// be is the compute backend for f32 matmuls; nil = the pure-Go path (default).
	be Backend

	cfg debertaConfig
	// wordEmb is [vocab, hidden], f32 or per-row int8 depending on the checkpoint
	// (see loadEmbeddingTable). Either way it aliases the mmap and rows are read
	// through WeightMat.Row, so the table is never resident in full.
	wordEmb linalg.WeightMat
	embLNW  []float32
	embLNB  []float32
	layers  []debertaLayer
	tok     *embed.Tokenizer
	maxSeq  int
	st      *embed.SafetensorsFile

	posAtt   posAttTerms
	attSpan  int // position_buckets — half the rel-embedding table
	maxRel   int // max_relative_positions used to extend relBucket for long inputs
	relScale float64

	// relBucket is logBucket(d) for offsets within the configured MaxPos baseline,
	// indexed by d+(maxSeq-1). Longer forwards build an extended per-call table;
	// the bucket still depends only on the offset, so no [L,L] index grid is needed.
	relBucket []int32
}

// posAttTerms records which disentangled terms the checkpoint declares.
type posAttTerms struct {
	c2p bool
	p2c bool
}

// count is |pos_att_type|; scale_factor is 1 + count.
func (p posAttTerms) count() int {
	n := 0
	if p.c2p {
		n++
	}
	if p.p2c {
		n++
	}
	return n
}

// parsePosAttType accepts both shapes HF configs use for pos_att_type: the
// pipe-joined string mDeBERTa ships ("p2c|c2p") and the list form ["p2c","c2p"].
func parsePosAttType(raw json.RawMessage) (posAttTerms, error) {
	var out posAttTerms
	if len(raw) == 0 || string(raw) == "null" {
		return out, nil
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err != nil {
		var s string
		if err2 := json.Unmarshal(raw, &s); err2 != nil {
			return out, fmt.Errorf("encoder: DeBERTa pos_att_type: not a string or list: %s", raw)
		}
		list = splitPipe(s)
	}
	for _, t := range list {
		switch t {
		case "c2p":
			out.c2p = true
		case "p2c":
			out.p2c = true
		case "":
		default:
			return out, fmt.Errorf("encoder: DeBERTa pos_att_type %q unsupported (c2p/p2c only)", t)
		}
	}
	return out, nil
}

func splitPipe(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '|' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return out
}

// debertaPrefixes are the tensor-name prefixes seen in the wild for this
// architecture: a bare DebertaV2Model, the pretraining checkpoint (mdeberta-v3-base
// ships DebertaV2ForPreTraining, hence "deberta."), GLiNER's nesting, and GLiNER2's
// ("encoder." — the BoundaryExtractor saves the backbone under its attribute name).
var debertaPrefixes = []string{
	"deberta.",
	"",
	"token_rep_layer.bert_layer.model.",
	"encoder.",
}

// LoadDeBERTa loads a DeBERTa-v2/v3 encoder (config.json + model.safetensors) from
// dir. It validates the architecture assumptions this forward implements, so an
// unsupported axis fails at load rather than producing a plausible wrong vector.
func LoadDeBERTa(dir string) (*DeBERTa, error) {
	return LoadDeBERTaAt(
		filepath.Join(dir, "config.json"),
		filepath.Join(dir, "model.safetensors"))
}

// LoadDeBERTaAt is LoadDeBERTa with the config and weight paths given explicitly.
// GLiNER2 keeps the backbone config in encoder_config/config.json while the shared
// safetensors file sits at the checkpoint root, so the two do not always share a
// directory.
func LoadDeBERTaAt(configPath, weightsPath string) (*DeBERTa, error) {
	dir := filepath.Dir(weightsPath)
	raw, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("encoder: read DeBERTa config: %w", err)
	}
	var c debertaConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("encoder: parse DeBERTa config: %w", err)
	}
	posAtt, err := parsePosAttType(c.PosAttType)
	if err != nil {
		return nil, err
	}
	switch {
	case c.ModelType != "" && c.ModelType != "deberta-v2":
		return nil, fmt.Errorf("encoder: DeBERTa model_type=%q unsupported (deberta-v2 only)", c.ModelType)
	case c.Act != "" && c.Act != "gelu":
		return nil, fmt.Errorf("encoder: DeBERTa hidden_act=%q unsupported (gelu only)", c.Act)
	case !c.RelativeAttention:
		return nil, fmt.Errorf("encoder: DeBERTa relative_attention=false unsupported")
	case c.PositionBiasedInput:
		return nil, fmt.Errorf("encoder: DeBERTa position_biased_input=true unsupported " +
			"(this forward adds no absolute position embedding)")
	case !c.ShareAttKey:
		return nil, fmt.Errorf("encoder: DeBERTa share_att_key=false unsupported " +
			"(the position projections would need their own weights)")
	case posAtt.count() == 0:
		return nil, fmt.Errorf("encoder: DeBERTa pos_att_type is empty; nothing disentangled to add")
	case c.NormRelEbd != "" && c.NormRelEbd != "layer_norm":
		return nil, fmt.Errorf("encoder: DeBERTa norm_rel_ebd=%q unsupported (layer_norm only)", c.NormRelEbd)
	case c.TypeVocab != 0:
		return nil, fmt.Errorf("encoder: DeBERTa type_vocab_size=%d unsupported "+
			"(this forward adds no token_type embedding)", c.TypeVocab)
	case c.Hidden == 0 || c.Heads == 0 || c.Layers == 0 || c.Intermediate == 0:
		return nil, fmt.Errorf("encoder: DeBERTa config missing a required dim")
	case c.Hidden%c.Heads != 0:
		return nil, fmt.Errorf("encoder: DeBERTa hidden %d not divisible by heads %d", c.Hidden, c.Heads)
	}
	if c.LNEps == 0 {
		c.LNEps = 1e-7
	}
	if c.MaxPos == 0 {
		c.MaxPos = 512
	}
	// HF DebertaV2Encoder: max_relative_positions < 1 falls back to
	// max_position_embeddings. That value is the log-bucket saturation point, so
	// getting it wrong shifts every bucket past ±128.
	maxRel := c.MaxRelativePositions
	if maxRel < 1 {
		maxRel = c.MaxPos
	}
	attSpan := c.PositionBuckets
	if attSpan <= 0 {
		return nil, fmt.Errorf("encoder: DeBERTa position_buckets=%d unsupported "+
			"(the unbucketed relative-position table is not implemented)", c.PositionBuckets)
	}

	st, err := embed.OpenSafetensorsMmap(weightsPath)
	if err != nil {
		return nil, fmt.Errorf("encoder: open DeBERTa safetensors: %w", err)
	}
	prefix, err := debertaPrefix(st)
	if err != nil {
		_ = st.Close()
		return nil, err
	}

	D, I := c.Hidden, c.Intermediate
	d := &DeBERTa{
		cfg: c, st: st, layers: make([]debertaLayer, c.Layers),
		posAtt: posAtt, attSpan: attSpan, maxRel: maxRel,
	}
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = loadF32(st, prefix+name, want)
		return v
	}

	// Vocab comes from the TENSOR, not the config. A fine-tune that adds special
	// tokens resizes the embedding table without touching config.vocab_size —
	// GLiNER is exactly that (250105 rows against mdeberta-v3-base's declared
	// 251000), and it reuses the base model's config verbatim. Hidden size stays a
	// hard assertion: that one is architectural, so a mismatch means the wrong file.
	if t, terr := st.Tensor(prefix + "embeddings.word_embeddings.weight"); terr == nil {
		if len(t.Shape) != 2 || t.Shape[1] != D {
			_ = st.Close()
			return nil, fmt.Errorf("encoder: DeBERTa word embedding shape %v, want [vocab %d]", t.Shape, D)
		}
		c.VocabSize = t.Shape[0]
		d.cfg.VocabSize = t.Shape[0]
	}
	if err == nil {
		d.wordEmb, err = loadEmbeddingTable(st, prefix+"embeddings.word_embeddings.weight", c.VocabSize, D)
	}
	d.embLNW = get("embeddings.LayerNorm.weight", D)
	d.embLNB = get("embeddings.LayerNorm.bias", D)

	// The relative embeddings, normalized ONCE. encoder.LayerNorm is not a
	// hidden-state norm — HF applies it to rel_embeddings in get_rel_embedding(),
	// which has no input dependence, so it is folded into the load here.
	relEmb := get("encoder.rel_embeddings.weight", 2*attSpan, D)
	encLNW := get("encoder.LayerNorm.weight", D)
	encLNB := get("encoder.LayerNorm.bias", D)
	if err == nil {
		norm := make([]float32, len(relEmb))
		copy(norm, relEmb)
		layerNorm(norm, encLNW, encLNB, 2*attSpan, D, c.LNEps)
		relEmb = norm
	}

	for i := range d.layers {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		l := &d.layers[i]
		l.Wq, l.WqB = get(p+"attention.self.query_proj.weight", D, D), get(p+"attention.self.query_proj.bias", D)
		l.Wk, l.WkB = get(p+"attention.self.key_proj.weight", D, D), get(p+"attention.self.key_proj.bias", D)
		l.Wv, l.WvB = get(p+"attention.self.value_proj.weight", D, D), get(p+"attention.self.value_proj.bias", D)
		l.Wo, l.WoB = get(p+"attention.output.dense.weight", D, D), get(p+"attention.output.dense.bias", D)
		l.AttnLNW = get(p+"attention.output.LayerNorm.weight", D)
		l.AttnLNB = get(p+"attention.output.LayerNorm.bias", D)
		l.Up, l.UpB = get(p+"intermediate.dense.weight", I, D), get(p+"intermediate.dense.bias", I)
		l.Down, l.DownB = get(p+"output.dense.weight", D, I), get(p+"output.dense.bias", D)
		l.MLPLNW = get(p+"output.LayerNorm.weight", D)
		l.MLPLNB = get(p+"output.LayerNorm.bias", D)
	}
	if err != nil {
		_ = st.Close()
		return nil, err
	}

	// Project the relative embeddings through each layer's key/query projections
	// once. share_att_key is asserted above, so these ARE key_proj/query_proj —
	// there is no separate pos_key_proj to load.
	//
	// HF applies no bias here: DisentangledSelfAttention calls self.key_proj(...) on
	// rel_embeddings, which does include the bias. It cancels in neither term, so it
	// is applied.
	for i := range d.layers {
		l := &d.layers[i]
		l.Kpos = make([]float32, 2*attSpan*D)
		l.Qpos = make([]float32, 2*attSpan*D)
		matmulBTInto(relEmb, l.Wk, l.Kpos, 2*attSpan, D, D)
		addBias(l.Kpos, l.WkB, 2*attSpan, D)
		matmulBTInto(relEmb, l.Wq, l.Qpos, 2*attSpan, D, D)
		addBias(l.Qpos, l.WqB, 2*attSpan, D)
	}

	d.maxSeq = c.MaxPos
	d.relScale = math.Sqrt(float64(D/c.Heads) * float64(1+posAtt.count()))
	d.relBucket = buildRelBucketTable(d.maxSeq, attSpan, maxRel)

	if tok, terr := embed.LoadTokenizer(filepath.Join(dir, "tokenizer.json")); terr == nil {
		d.tok = tok
	} else if tok, terr := embed.LoadTokenizerSPM(
		filepath.Join(dir, "spm.model"), filepath.Join(dir, "added_tokens.json")); terr == nil {
		d.tok = tok
	} else if tok, terr := embed.LoadTokenizerSPM(filepath.Join(dir, "spm.model"), ""); terr == nil {
		d.tok = tok
	}
	return d, nil
}

// loadEmbeddingTable loads the [vocab, hidden] word-embedding table in whichever
// precision the checkpoint stores it, WITHOUT materializing it as f32.
//
// That last part is the whole function. mDeBERTa-v3's table is 250,105 × 768 —
// 768 MB, two thirds of a GLiNER checkpoint — and it is mmapped precisely so that
// a forward, which gathers at most max_len rows, never faults the rest in. So:
//
//   - F32-class storage: wrapped in place, aliasing the mapping. Identical to the
//     `copy` this replaced, byte for byte.
//   - I8 + a companion "<name>.scale" tensor (per-row symmetric, the format
//     LoadModernBERTQ8 already reads and scripts/quantize_embeddings.py writes):
//     wrapped in place as int8, and each gathered row is widened on demand. The
//     file is 4× smaller and the resident set does not move.
//
// The trap this exists to avoid: st.TensorF32 silently DEQUANTIZES an I8 tensor
// with a companion scale into a fresh allocation. Calling it here would produce a
// correct model, a green test suite, and a 768 MB heap allocation — the exact
// opposite of the reason to quantize. Hence the dtype probe first.
func loadEmbeddingTable(st *embed.SafetensorsFile, name string, vocab, hidden int) (linalg.WeightMat, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return linalg.WeightMat{}, err
	}
	if len(t.Shape) != 2 || t.Shape[0] != vocab || t.Shape[1] != hidden {
		return linalg.WeightMat{}, fmt.Errorf("encoder: %s shape %v, want [%d %d]",
			name, t.Shape, vocab, hidden)
	}
	if t.DType != "I8" {
		w, ferr := st.TensorF32(name, vocab, hidden)
		if ferr != nil {
			return linalg.WeightMat{}, ferr
		}
		return linalg.WrapF32(w, vocab, hidden), nil
	}
	q8, qerr := t.Int8s()
	if qerr != nil {
		return linalg.WeightMat{}, qerr
	}
	scales, serr := st.TensorF32(name + ".scale")
	if serr != nil {
		return linalg.WeightMat{}, fmt.Errorf(
			"encoder: I8 embedding table %q has no companion %q: %w", name, name+".scale", serr)
	}
	switch len(scales) {
	case vocab:
		// Per-row: one scale per token, the only calibration that survives an
		// outlier row (see scripts/quantize_embeddings.py).
	case 1:
		// Per-tensor, broadcast. Accepted because the format allows it, but it is
		// a materially worse table — TestDeBERTa_q8Parity's break-it-first arm is
		// exactly this.
		b := make([]float32, vocab)
		for i := range b {
			b[i] = scales[0]
		}
		scales = b
	default:
		return linalg.WeightMat{}, fmt.Errorf(
			"encoder: %q has %d scales, want %d (per-row) or 1 (per-tensor)",
			name+".scale", len(scales), vocab)
	}
	// w8a8=false: the gather never enters a matmul, so the activation side is
	// irrelevant here — only Row() is ever called on this WeightMat.
	return linalg.WrapInt8(q8, scales, vocab, hidden, false), nil
}

// debertaPrefix finds which tensor-name prefix this checkpoint uses by probing for
// the word embedding under each known nesting.
func debertaPrefix(st *embed.SafetensorsFile) (string, error) {
	for _, p := range debertaPrefixes {
		if _, err := st.Tensor(p + "embeddings.word_embeddings.weight"); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("encoder: no DeBERTa word embedding found under any known prefix %q",
		debertaPrefixes)
}

// buildRelBucketTable precomputes logBucket(d) for d in [-(maxSeq-1), maxSeq-1].
func buildRelBucketTable(maxSeq, bucketSize, maxPosition int) []int32 {
	n := 2*maxSeq - 1
	t := make([]int32, n)
	for i := range n {
		t[i] = int32(logBucketPosition(i-(maxSeq-1), bucketSize, maxPosition))
	}
	return t
}

// logBucketPosition is HF's make_log_bucket_position: offsets inside ±(bucketSize/2)
// stay linear, and beyond that they are compressed logarithmically up to maxPosition.
//
// The ceil and the sign reattachment are the whole function, and an off-by-one in
// either is invisible — it produces a valid bucket, just the wrong one, for every
// pair of tokens more than 128 apart.
func logBucketPosition(relPos, bucketSize, maxPosition int) int {
	mid := bucketSize / 2
	if relPos >= -mid && relPos <= mid {
		// HF: abs_pos is forced to mid-1 inside the open interval, which keeps it
		// <= mid, so the torch.where selects relative_pos unchanged. At exactly ±mid
		// abs_pos == mid, still <= mid, so the linear branch holds there too.
		return relPos
	}
	sign := 1
	if relPos < 0 {
		sign = -1
	}
	abs := relPos
	if abs < 0 {
		abs = -abs
	}
	log := math.Ceil(math.Log(float64(abs)/float64(mid))/
		math.Log(float64(maxPosition-1)/float64(mid))*float64(mid-1)) + float64(mid)
	return sign * int(log)
}

// Close releases the mmap-backed weights. Idempotent.
func (d *DeBERTa) Close() error {
	if d.st == nil {
		return nil
	}
	return d.st.Close()
}

// HiddenDim reports the model's hidden width.
func (d *DeBERTa) HiddenDim() int { return d.cfg.Hidden }

// EmbeddingPrecision reports how the word-embedding table is stored — "f32" or
// "int8". It is observable because it is the one thing that differs between a
// checkpoint and its quantized twin: the forward, the config and the tokenizer are
// identical, so without this a caller cannot tell which artifact it loaded.
func (d *DeBERTa) EmbeddingPrecision() string { return d.wordEmb.Kind() }

// MaxSeqLength reports the checkpoint's configured position baseline. DeBERTa
// uses relative positions, so HiddenStates also accepts longer sequences and
// extends its relative-offset table for them.
func (d *DeBERTa) MaxSeqLength() int { return d.maxSeq }

// Tokenizer exposes the loaded tokenizer (nil when the checkpoint shipped none).
func (d *DeBERTa) Tokenizer() *embed.Tokenizer { return d.tok }

// HiddenStates runs the transformer on token ids (already wrapped [CLS]…[SEP]) and
// returns the last hidden state [L, hidden], row-major.
//
// DeBERTa is exported without a pooling/Encode surface on purpose: mdeberta-v3-base
// is a bare LM, not a sentence embedder, and its consumer here is GLiNER's token-level
// head. Cf. FacebookAI/xlm-roberta-base in docs/embedder-coverage.md, which is listed
// forward-only for the same reason.
func (d *DeBERTa) HiddenStates(ids []int32) []float32 {
	h, _ := d.forward(ids, false)
	return h
}

// AllHiddenStates returns the embedding output followed by every layer's output —
// layers+1 slices of [L, hidden]. This is HF's output_hidden_states=True, and it
// exists for the parity gate: a whole-model-only fixture tells you THAT the forward
// diverged, never WHERE, and for this architecture the plausible causes (bucket
// arithmetic, p2c indexing, scale, rel-embedding norm) all present identically at
// the output.
func (d *DeBERTa) AllHiddenStates(ids []int32) [][]float32 {
	_, all := d.forward(ids, true)
	return all
}

func (d *DeBERTa) forward(ids []int32, collect bool) ([]float32, [][]float32) {
	enterForward()
	defer leaveForward()

	c := d.cfg
	L, D := len(ids), c.Hidden
	headDim := D / c.Heads
	I := c.Intermediate
	eps := c.LNEps
	span := d.attSpan
	scale := float32(1.0 / d.relScale)
	relBucket, relCenter := d.relativeBuckets(L)

	s := getScratch()
	s.be = d.be
	defer putScratch(s)
	s.ensureLayer(L, D, I, c.Heads, headDim, L)

	// Embeddings: the word embedding, normalized. No position term
	// (position_biased_input=false) and no token_type term (type_vocab_size=0) —
	// both asserted at load. HF then multiplies by the attention mask; aikit runs
	// unpadded single sequences, so that mask is all-ones and inert here.
	h := make([]float32, L*D)
	vocab := d.wordEmb.Rows()
	for i, id := range ids {
		// Row is a copy for an f32 table and a widen for an int8 one, so at most
		// L rows of the table are ever touched either way.
		d.wordEmb.Row(clampTokenID(id, vocab), h[i*D:(i+1)*D])
	}
	layerNorm(h, d.embLNW, d.embLNB, L, D, eps)

	var all [][]float32
	snapshot := func() {
		if !collect {
			return
		}
		cp := make([]float32, len(h))
		copy(cp, h)
		all = append(all, cp)
	}
	snapshot()

	// Disentangled-attention scratch. These shapes ([L, 2*span] and [2*span,
	// headDim]) have no counterpart in the shared scratch, which is sized for
	// [L,L] scores and [L,headDim] head extracts, so they are allocated per
	// forward rather than per layer or per head.
	c2pRaw := make([]float32, L*2*span)
	p2cRaw := make([]float32, L*2*span)
	posKH := make([]float32, 2*span*headDim)
	posQH := make([]float32, 2*span*headDim)
	kHScaled := make([]float32, L*headDim)

	for li := range d.layers {
		l := &d.layers[li]

		Q, K, V := s.Q[:L*D], s.K[:L*D], s.V[:L*D]
		s.mm(h, l.Wq, Q, L, D, D)
		addBias(Q, l.WqB, L, D)
		s.mm(h, l.Wk, K, L, D, D)
		addBias(K, l.WkB, L, D)
		s.mm(h, l.Wv, V, L, D, D)
		addBias(V, l.WvB, L, D)

		ctx := s.ctx[:L*D]
		qH, kH := s.qH[:L*headDim], s.kH[:L*headDim]
		vHT := s.vH[:headDim*L]
		ctxHead := s.ctxHead[:L*headDim]
		scores := s.scores[:L*L]

		for head := range c.Heads {
			off := head * headDim
			for i := range L {
				src := i*D + off
				copy(qH[i*headDim:(i+1)*headDim], Q[src:src+headDim])
				copy(kH[i*headDim:(i+1)*headDim], K[src:src+headDim])
				for dd := range headDim {
					vHT[dd*L+i] = V[src+dd]
				}
			}
			// HF divides the KEY by the scale before the bmm rather than scaling the
			// product, so do the same: the two differ in the last bits, and this
			// phase exists to be bit-comparable with the reference.
			for i := range L * headDim {
				kHScaled[i] = kH[i] * scale
			}
			s.mm(qH, kHScaled, scores, L, headDim, L)

			if d.posAtt.c2p || d.posAtt.p2c {
				for i := range 2 * span {
					src := i*D + off
					if d.posAtt.c2p {
						copy(posKH[i*headDim:(i+1)*headDim], l.Kpos[src:src+headDim])
					}
					if d.posAtt.p2c {
						copy(posQH[i*headDim:(i+1)*headDim], l.Qpos[src:src+headDim])
					}
				}
			}

			// content → position: score[i][j] += (Q[i] · Kpos[bucket(i-j)+span]) / scale
			if d.posAtt.c2p {
				s.mm(qH, posKH, c2pRaw, L, headDim, 2*span)
				for i := range L {
					row := c2pRaw[i*2*span:]
					out := scores[i*L : (i+1)*L]
					for j := range L {
						out[j] += row[bucketIndex(relBucket, relCenter, i, j, span)] * scale
					}
				}
			}

			// position → content. The index is NOT the c2p index transposed: HF
			// gathers with clamp(-bucket, ...) over a grid built the other way round
			// and THEN transposes the result, which lands as
			//   score[i][j] += (K[j] · Qpos[-bucket(j-i)+span]) / scale
			// Reading it as a transpose of the c2p gather is the classic DeBERTa
			// port bug: it is symmetric-looking, never crashes, and is wrong for
			// every i≠j.
			if d.posAtt.p2c {
				s.mm(kH, posQH, p2cRaw, L, headDim, 2*span)
				for i := range L {
					out := scores[i*L : (i+1)*L]
					for j := range L {
						out[j] += p2cRaw[j*2*span+bucketIndexNeg(relBucket, relCenter, j, i, span)] * scale
					}
				}
			}

			softmaxRows(scores, L, L)
			s.mm(scores, vHT, ctxHead, L, L, headDim)
			for i := range L {
				copy(ctx[i*D+off:i*D+off+headDim], ctxHead[i*headDim:(i+1)*headDim])
			}
		}

		attnOut := s.out[:L*D]
		s.mm(ctx, l.Wo, attnOut, L, D, D)
		addBias(attnOut, l.WoB, L, D)
		for i := range h {
			h[i] += attnOut[i]
		}
		layerNorm(h, l.AttnLNW, l.AttnLNB, L, D, eps)

		up := s.val[:L*I]
		s.mm(h, l.Up, up, L, D, I)
		addBias(up, l.UpB, L, I)
		gelu(up)
		ffn := s.mid[:L*D]
		s.mm(up, l.Down, ffn, L, I, D)
		addBias(ffn, l.DownB, L, D)
		for i := range h {
			h[i] += ffn[i]
		}
		layerNorm(h, l.MLPLNW, l.MLPLNB, L, D, eps)

		snapshot()
	}
	return h, all
}

// relativeBuckets returns an offset table large enough for seqLen. The
// checkpoint's max_position_embeddings is a relative-distance calibration
// value for DeBERTa, not a hard sequence limit, so longer callers get a local
// extension rather than being silently truncated. The loaded table remains
// immutable, preserving concurrent forward safety.
func (d *DeBERTa) relativeBuckets(seqLen int) ([]int32, int) {
	if seqLen <= d.maxSeq {
		return d.relBucket, d.maxSeq - 1
	}
	return buildRelBucketTable(seqLen, d.attSpan, d.maxRel), seqLen - 1
}

// bucketIndex is the c2p gather column for (query i, key j):
// clamp(bucket(i-j) + span, 0, 2*span-1).
func bucketIndex(relBucket []int32, center, i, j, span int) int {
	b := int(relBucket[i-j+center])
	return clampInt(b+span, 0, 2*span-1)
}

// bucketIndexNeg is the p2c gather column: clamp(-bucket(i-j) + span, 0, 2*span-1).
func bucketIndexNeg(relBucket []int32, center, i, j, span int) int {
	b := int(relBucket[i-j+center])
	return clampInt(-b+span, 0, 2*span-1)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

package encoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
)

// modernbert.go — the ModernBERT encoder (answerdotai/ModernBERT-base; the base
// of hotchpotch/bekko-embedding-v1-*). It is a BERT-family encoder like bert.go
// and gte.go, but on five axes it matches neither:
//
//   - PRE-NORM residual blocks (norm before attention / MLP, residual added to
//     the un-normed hidden) — bert.go and gte.go are post-norm.
//   - BIAS-FREE everything: attention QKV/Wo, both LayerNorms and the MLP carry
//     no bias (config attention_bias / norm_bias / mlp_bias = false).
//   - RoPE positions with NO additive position embedding (position_embedding_type
//     "sans_pos" in bekko / newer exports, "rope" in cl-nagoya/ruri-v3-* — the two
//     spell the same architecture); the RoPE convention is bit-identical to
//     encoder/rope.go.
//   - ALTERNATING local / global attention: every global_attn_every_n_layers-th
//     layer attends globally, the rest use a bidirectional sliding window of
//     half-width local_attention/2 (a query at i sees keys j with |i-j| ≤ S).
//   - A GeGLU MLP that activates the FIRST Wi chunk and gates on the SECOND
//     (gelu(x[:I])·x[I:2I]) — the reverse chunk order of gte.go's gelu(gate)·up.
//
// Layer 0's attention pre-norm is the identity (no params); every layer has an
// MLP pre-norm, and a final LayerNorm follows the last layer.
//
//	emb   = LayerNorm( word[ids] )                          // no position embedding
//	attn  = softmax( (RoPE(Q)·RoPE(K)ᵀ)/√d + window_mask ) · V  // fused qkv, no bias
//	h     = h + attn·Woᵀ                                    // pre-norm residual
//	mlp   = ( gelu(Wi[:I]) ⊙ Wi[I:2I] ) · Woᵀ               // GeGLU, no bias
//	h     = h + mlp                                         // pre-norm residual
//	out   = LayerNorm( h )                                  // final_norm
//
// Weights are PyTorch [out, in], so each linear is h·Wᵀ via matmulBT (A·Bᵀ); the
// shared layerNorm / softmaxRow / poolOne primitives are reused. All weights are
// stored BF16 and widened to f32 by TensorF32 at load.

// mbMasked is the additive attention-mask sentinel for sliding-window positions
// a local layer must not attend to. softmaxRow subtracts the row max first, so a
// large finite negative exponentiates to 0 — equivalent to -inf without risking
// inf arithmetic. Each row always keeps its diagonal (|i-i|=0 ≤ S), so a row can
// never be fully masked.
const mbMasked = float32(-1e30)

type modernBertConfig struct {
	VocabSize       int     `json:"vocab_size"`
	Hidden          int     `json:"hidden_size"`
	Layers          int     `json:"num_hidden_layers"`
	Heads           int     `json:"num_attention_heads"`
	Intermediate    int     `json:"intermediate_size"`
	MaxPos          int     `json:"max_position_embeddings"`
	NormEps         float64 `json:"norm_eps"`
	LNEps           float64 `json:"layer_norm_eps"`
	Act             string  `json:"hidden_activation"`
	PosType         string  `json:"position_embedding_type"`
	ModelType       string  `json:"model_type"`
	GlobalRopeTheta float64 `json:"global_rope_theta"`
	LocalRopeTheta  float64 `json:"local_rope_theta"`
	GlobalEvery     int     `json:"global_attn_every_n_layers"`
	LocalAttention  int     `json:"local_attention"`
	AttnBias        bool    `json:"attention_bias"`
	NormBias        bool    `json:"norm_bias"`
	MLPBias         bool    `json:"mlp_bias"`

	// RopeParameters is the transformers-5.x spelling of the two thetas above:
	// the flat global_rope_theta / local_rope_theta keys are gone and each
	// attention class carries its own block. Exports written by transformers ≥5.7
	// (cross-encoder/ettin-reranker-*) use ONLY this form, and because a missing
	// flat key defaults to 10000 rather than erroring, reading just the flat keys
	// mis-computes RoPE silently. See rope thetas in loadModernBertConfig.
	RopeParameters struct {
		FullAttention    modernBertRope `json:"full_attention"`
		SlidingAttention modernBertRope `json:"sliding_attention"`
	} `json:"rope_parameters"`
}

// modernBertRope is one entry of config.json's rope_parameters map.
type modernBertRope struct {
	RopeTheta float64 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

type modernBertLayer struct {
	Wqkv    []float32 // packed [3*hidden, hidden] (no bias)
	AttnWo  []float32 // attention out_proj [hidden, hidden] (no bias)
	AttnLNW []float32 // attention PRE-norm weight [hidden]; nil for layer 0 (identity)
	MLPWi   []float32 // fused input/gate [2*intermediate, hidden] (no bias)
	MLPWo   []float32 // MLP out_proj [hidden, intermediate] (no bias)
	MLPLNW  []float32 // MLP PRE-norm weight [hidden]
	local   bool      // sliding-window attention (vs global)
}

// ModernBERT is a loaded ModernBERT-class encoder. Immutable after load; the
// forward is read-only-safe for concurrent use (no shared mutable state).
type ModernBERT struct {
	// be is the compute backend for f32 matmuls; nil = the pure-Go path (default).
	be Backend

	cfg      modernBertConfig
	wordEmb  []float32 // [vocab, hidden]
	embLNW   []float32 // embedding LayerNorm weight [hidden]
	layers   []modernBertLayer
	finalLNW []float32 // post-last-layer LayerNorm weight [hidden]
	zeroBias []float32 // [hidden] zeros — bias-free layerNorm reuses the f32 kernel
	tok      Tokenizer
	maxSeq   int
	pool     pooling
	slidingW int     // sliding-window half-width S = local_attention/2
	thetaG   float64 // global-layer RoPE base
	thetaL   float64 // local-layer RoPE base
	st       *embed.SafetensorsFile
	ropeG    ropeCache // global-layer rotary table
	ropeL    ropeCache // local-layer rotary table
}

// loadModernBertConfig reads dir/config.json and validates the architecture
// assumptions the ModernBERT forward implements: RoPE-only positions
// (position_embedding_type "sans_pos" or "rope", both sans additive position
// embedding), GELU GeGLU activation, and bias-free attention / norms / MLP
// (the ModernBERT defaults; every released ModernBERT embedder uses them).
// Zero-valued norm eps and RoPE thetas are defaulted. Shared by the f32 and
// Q8 loaders.
func loadModernBertConfig(dir string) (modernBertConfig, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return modernBertConfig{}, fmt.Errorf("encoder: read ModernBERT config: %w", err)
	}
	var c modernBertConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return modernBertConfig{}, fmt.Errorf("encoder: parse ModernBERT config: %w", err)
	}
	switch {
	case c.ModelType != "modernbert":
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT model_type=%q unsupported (modernbert only)", c.ModelType)
	case c.Act != "gelu":
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT hidden_activation=%q unsupported (gelu only)", c.Act)
	case c.PosType != "sans_pos" && c.PosType != "rope":
		// "sans_pos" (bekko / newer sentence-transformers exports) and "rope"
		// (cl-nagoya/ruri-v3-*, the original ModernBERT spelling) name the same
		// architecture: positions come entirely from RoPE, with NO additive position
		// embedding — the checkpoint carries no position tensor and the forward
		// below never adds one. Anything else (e.g. a learned/absolute embedding)
		// is a different model this forward cannot run.
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT position_embedding_type=%q unsupported (sans_pos/rope only)", c.PosType)
	case c.AttnBias || c.NormBias || c.MLPBias:
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT bias=true unsupported (attention/norm/mlp biases must be false)")
	case c.Hidden == 0 || c.Heads == 0 || c.Layers == 0 || c.Intermediate == 0:
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT config missing a required dim")
	case c.Hidden%c.Heads != 0:
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT hidden %d not divisible by heads %d", c.Hidden, c.Heads)
	case (c.Hidden/c.Heads)%2 != 0:
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT head dim %d must be even for RoPE", c.Hidden/c.Heads)
	case c.GlobalEvery <= 0:
		return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT global_attn_every_n_layers=%d must be positive", c.GlobalEvery)
	}
	eps := c.NormEps
	if eps == 0 {
		eps = c.LNEps
	}
	if eps == 0 {
		eps = 1e-5
	}
	c.NormEps = eps

	// RoPE thetas, resolved per attention class: the flat transformers-4.x key
	// first (bekko, ruri-v3), then the nested transformers-5.x block, then the
	// 10000 default. Only "default" rope_type is implemented — a linear / yarn /
	// llama3 entry scales positions and must fail loudly rather than be read for
	// its theta and otherwise ignored.
	full, sliding := c.RopeParameters.FullAttention, c.RopeParameters.SlidingAttention
	for _, r := range [...]modernBertRope{full, sliding} {
		if r.RopeType != "" && r.RopeType != "default" {
			return modernBertConfig{}, fmt.Errorf("encoder: ModernBERT rope_type=%q unsupported (default only)", r.RopeType)
		}
	}
	if c.GlobalRopeTheta == 0 {
		c.GlobalRopeTheta = full.RopeTheta
	}
	if c.GlobalRopeTheta == 0 {
		c.GlobalRopeTheta = 10000
	}
	if c.LocalRopeTheta == 0 {
		c.LocalRopeTheta = sliding.RopeTheta
	}
	if c.LocalRopeTheta == 0 {
		c.LocalRopeTheta = c.GlobalRopeTheta
	}
	return c, nil
}

// loadModernBertTokenizer resolves dir/tokenizer.json via embed and returns nil
// when it cannot be read (the trunk stays usable via Embed with pre-tokenized
// ids). embed covers the whole ModernBERT family: the SentencePiece-style BPE
// vocabs (bekko, ruri-v3) through its spBPE backend and the byte-level BPE ones
// (the OLMo vocab shared by ModernBERT-base and the Ettin line) through its bpe
// backend. Shared by the f32 and int8 loaders.
func loadModernBertTokenizer(dir string) Tokenizer {
	path := filepath.Join(dir, "tokenizer.json")
	if tok, err := embed.LoadTokenizer(path); err == nil {
		return tok
	}
	return nil
}

// LoadModernBERT loads a ModernBERT encoder (config.json + model.safetensors with
// ModernBERT tensor names) from dir. It validates the architecture assumptions
// this forward implements: RoPE-only positions (position_embedding_type "sans_pos"
// or "rope", both sans additive position embedding), GELU GeGLU activation, and
// bias-free attention / norms / MLP (the ModernBERT defaults; every released
// ModernBERT embedder uses them).
func LoadModernBERT(dir string) (*ModernBERT, error) {
	c, err := loadModernBertConfig(dir)
	if err != nil {
		return nil, err
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open ModernBERT safetensors: %w", err)
	}
	D, I := c.Hidden, c.Intermediate
	m := &ModernBERT{
		cfg:      c,
		st:       st,
		layers:   make([]modernBertLayer, c.Layers),
		zeroBias: make([]float32, D),
		slidingW: c.LocalAttention / 2,
		thetaG:   c.GlobalRopeTheta,
		thetaL:   c.LocalRopeTheta,
	}

	// Encoder tensors are bare in sentence-transformers exports (embeddings.*,
	// layers.N.*) but carry a "model." prefix when saved from a raw ModernBertModel
	// wrapper. Detect which, mirroring bert.go's prefix probe.
	prefix := ""
	if _, e := st.Tensor("embeddings.tok_embeddings.weight"); e != nil {
		if _, e2 := st.Tensor("model.embeddings.tok_embeddings.weight"); e2 == nil {
			prefix = "model."
		}
	}

	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = loadF32(st, name, want)
		return v
	}
	m.wordEmb = get(prefix+"embeddings.tok_embeddings.weight", c.VocabSize, D)
	m.embLNW = get(prefix+"embeddings.norm.weight", D)
	for i := range m.layers {
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		l := &m.layers[i]
		l.Wqkv = get(p+"attn.Wqkv.weight", 3*D, D)
		l.AttnWo = get(p+"attn.Wo.weight", D, D)
		// Layer 0's attention pre-norm is nn.Identity — no params in the checkpoint.
		if i > 0 {
			l.AttnLNW = get(p+"attn_norm.weight", D)
		}
		l.MLPWi = get(p+"mlp.Wi.weight", 2*I, D)
		l.MLPWo = get(p+"mlp.Wo.weight", D, I)
		l.MLPLNW = get(p+"mlp_norm.weight", D)
		l.local = i%c.GlobalEvery != 0
	}
	m.finalLNW = get(prefix+"final_norm.weight", D)
	if err != nil {
		_ = st.Close()
		return nil, err
	}

	// max sequence length: sentence-transformers right-truncates here; the hard
	// ceiling is the RoPE position capacity (max_position_embeddings).
	m.maxSeq = c.MaxPos
	if sb, e := os.ReadFile(filepath.Join(dir, "sentence_bert_config.json")); e == nil {
		var v struct {
			MaxSeqLength int `json:"max_seq_length"`
		}
		if json.Unmarshal(sb, &v) == nil && v.MaxSeqLength > 0 {
			m.maxSeq = min(v.MaxSeqLength, c.MaxPos)
		}
	}
	// Tokenizer is best-effort: embed's spBPE backend handles the
	// SentencePiece-style BPE vocabs (bekko, ruri-v3) and its byte-level BPE
	// backend the OLMo-vocab ModernBERTs (Ettin). A tokenizer that embed cannot
	// parse leaves m.tok nil — the trunk stays usable via Embed with pre-tokenized ids.
	m.tok = loadModernBertTokenizer(dir)
	// Pooling is a declared per-model property (sentence-transformers
	// 1_Pooling/config.json); ModernBERT embedders pool mean. Absent file → mean.
	if m.pool, err = poolingFromConfig(dir, poolMean); err != nil {
		_ = st.Close()
		return nil, err
	}
	return m, nil
}

// Close releases the mmap-backed weights. Call once after the last Encode;
// idempotent. See Model.Close.
func (m *ModernBERT) Close() error {
	if m.st == nil {
		return nil
	}
	return m.st.Close()
}

// HiddenDim is the output embedding dimension (384 for bekko / ModernBERT-base).
func (m *ModernBERT) HiddenDim() int { return m.cfg.Hidden }

// Encode tokenizes text (wrapped <bos>…<eos>, right-truncated to the model's max
// sequence length) and returns the pooled, L2-normalized sentence embedding.
func (m *ModernBERT) Encode(text string) ([]float32, error) {
	if m.tok == nil {
		return nil, fmt.Errorf("encoder: ModernBERT has no usable tokenizer; use Embed with pre-tokenized ids")
	}
	ids, err := m.tok.EncodeWithSpecials(text, m.maxSeq)
	if err != nil {
		return nil, err
	}
	return m.Embed(ids), nil
}

// Embed returns the pooled (per 1_Pooling; mean by default), L2-normalized
// sentence embedding for token ids (already wrapped with <bos>…<eos>).
func (m *ModernBERT) Embed(ids []int32) []float32 {
	D := m.cfg.Hidden
	h := m.forward(ids)
	return l2norm(poolOne(h, len(h)/D, D, m.pool))
}

// mbBandQTile is the query-tile height of the banded sliding-window attention
// path. A tile of QT queries starting at q0 touches keys [q0-S, q0+QT-1+S], so
// the score tile is QT×(QT+2S) instead of QT×L: the wasted fraction is
// (QT+2S)/(2S+1), which is 1.49× at QT=S=64 and 1.98× at QT=128. Smaller tiles
// waste less but shrink the GEMM's M dimension and multiply the per-tile V-band
// packing; 64 is the knee, and it also matches linalg's 32-row mBlock evenly.
const mbBandQTile = 64

// bandedAttnOK reports whether the sliding window is narrow enough that the banded
// attention path is worth taking over the dense L×L one.
//
// A local layer only ever reads keys within ±S of the query, so the dense path
// computes L² scores to keep 2S+1 of them per row. The banded path skips the rest
// outright — QK ᵀ, softmax, the mask writes, and scores·V all shrink by roughly
// L/(QT+2S). At ruri-v3-30m's local_attention=128 (S=64) and L=5002 that is a 26×
// cut on six of the ten layers.
//
// The gate is the break-even: below it the band covers most of the row anyway and
// the tiling overhead (per-tile V packing, smaller GEMMs) is not repaid. A backend
// disables it because the tiles run on several goroutines at once and Backend has
// no concurrency contract — the dense path keeps that offload intact.
//
// Free function, not a method: the f32 and int8 ModernBERT forwards share it (the
// int8 model quantizes only the projections — its attention is the same f32 code).
func bandedAttnOK(be Backend, S, L int) bool {
	return be == nil && mbBandQTile+2*S < L/2
}

// mbAttnScale is the 1/√headDim attention scale, folded into Q ONCE at head
// extraction (splitHead) instead of applied to the L×L score matrix afterward.
//
// The score matrix is the biggest buffer in the forward and the scale pass over it
// was serial: 3.6 s of a 3.9 s-per-call profile's `forward` flat time at L=5002.
// Scaling Q instead costs L·headDim multiplies rather than L², a 78× cut at that
// length, and it is exactly what PyTorch's scaled_dot_product_attention does.
//
// Numerics: for headDim 16/64/256 the scale is a power of two (1/4, 1/8, 1/16), so
// every q·scale is exact and each partial sum of the dot scales exactly with it —
// bit-identical to scaling the sums. Other head dims round each Q element once
// instead of each score once (≤1 ulp either way).
func mbAttnScale(headDim int) float32 { return float32(1.0 / math.Sqrt(float64(headDim))) }

// splitHead extracts head headIdx out of the packed [L, D] Q/K/V into the layout
// attention wants: qH/kH [L, headDim] row-major and vHT [headDim, L] — V
// TRANSPOSED, so the scores·V step runs through the same A·Bᵀ kernel as everything
// else. Q is scaled by 1/√headDim on the way out (see mbAttnScale).
func splitHead(Q, K, V, qH, kH, vHT []float32, headIdx, L, D, headDim int, scale float32) {
	for i := range L {
		src := i*D + headIdx*headDim
		qRow := qH[i*headDim : (i+1)*headDim]
		for d, q := range Q[src : src+headDim] {
			qRow[d] = q * scale
		}
		copy(kH[i*headDim:(i+1)*headDim], K[src:src+headDim])
		for d := range headDim {
			vHT[d*L+i] = V[src+d]
		}
	}
}

// bandedAttnHead computes one head of sliding-window attention (half-width S) into
// ctxHead[L, headDim], reading qH/kH [L, headDim] and vHT [headDim, L] (V
// transposed). qH is expected pre-scaled by 1/√headDim (splitHead).
//
// Queries are processed in tiles of mbBandQTile. Tiles are independent — each
// writes its own ctxHead rows and reads only shared, read-only q/k/v — so they are
// claimed one at a time off parallelChunks' counter, each worker holding a private
// slot of s.band. That is also what keeps this path parallel at all: a per-tile
// QKᵀ is far too small to clear parallelThreshold, so the parallelism has to come
// from the tile loop rather than from inside the matmuls, the way the dense path
// gets it.
//
// Numerics: softmaxRow runs over each row's live [lo,hi] span and the rest is set
// to 0. That is bit-identical to the dense path's "write mbMasked, then softmax
// the whole row" — mbMasked is never the row max and exp(mbMasked-max) is exactly
// 0, so both the max and the f64 sum see the same values. Only the QKᵀ reduction
// differs, and only in tiling, the same way linalg documents M-invariance.
func bandedAttnHead(s *scratch, S int, qH, kH, vHT, ctxHead []float32, L, headDim, maxWorkers int) {
	bandW := mbBandQTile + 2*S
	slot := (mbBandQTile + headDim) * bandW
	tiles := (L + mbBandQTile - 1) / mbBandQTile

	parallelChunks(tiles, 1, min(maxWorkers, tiles), func(w, t0, t1 int) {
		arena := s.band[w*slot : (w+1)*slot]
		scoreBuf, vBuf := arena[:mbBandQTile*bandW], arena[mbBandQTile*bandW:]
		for t := t0; t < t1; t++ {
			q0 := t * mbBandQTile
			q1 := min(q0+mbBandQTile, L)
			k0, k1 := max(q0-S, 0), min(q1-1+S+1, L)
			qt, kw := q1-q0, k1-k0

			scores := scoreBuf[:qt*kw]
			matmulBTBlockedInto(qH[q0*headDim:q1*headDim], kH[k0*headDim:k1*headDim], scores, qt, headDim, kw)
			for i := range qt {
				row := scores[i*kw : (i+1)*kw]
				gi := q0 + i
				lo, hi := max(gi-S, 0)-k0, min(gi+S, L-1)-k0
				softmaxRow(row[lo : hi+1])
				clear(row[:lo])
				clear(row[hi+1:])
			}
			// Pack this tile's V columns [k0,k1) into a [headDim, kw] block so the
			// context matmul keeps the contiguous-K layout matmulBT needs (vHT's rows
			// are strided by L, not kw).
			vb := vBuf[:headDim*kw]
			for d := range headDim {
				copy(vb[d*kw:(d+1)*kw], vHT[d*L+k0:d*L+k1])
			}
			matmulBTBlockedInto(scores, vb, ctxHead[q0*headDim:q1*headDim], qt, kw, headDim)
		}
	})
}

// forward runs the ModernBERT transformer on token ids and returns the final
// (post final_norm) hidden state [L, hidden], row-major.
func (m *ModernBERT) forward(ids []int32) []float32 {
	// In-flight accounting for the intra-op matmul gate — see BERT.hiddenStates
	// (perf-campaign item 6).
	enterForward()
	defer leaveForward()

	c := m.cfg
	if len(ids) > m.maxSeq {
		ids = ids[:m.maxSeq]
	}
	L, D := len(ids), c.Hidden
	headDim := D / c.Heads
	I := c.Intermediate
	eps := c.NormEps
	scale := mbAttnScale(headDim)

	s := getScratch()
	s.be = m.be
	defer putScratch(s)
	s.ensureLayer(L, D, I, c.Heads, headDim, L)
	s.ensureFusedMLP(L, I)
	ropeG := m.ropeG.get(L, headDim, m.thetaG)
	ropeL := m.ropeL.get(L, headDim, m.thetaL)

	// Embeddings: word lookup, then a bias-free LayerNorm. Positions come entirely
	// from RoPE (sans_pos) — there is no additive position embedding.
	h := make([]float32, L*D)
	vocab := len(m.wordEmb) / D
	for i, id := range ids {
		if int(id) < 0 || int(id) >= vocab {
			id = 0
		}
		copy(h[i*D:(i+1)*D], m.wordEmb[int(id)*D:int(id)*D+D])
	}
	layerNorm(h, m.embLNW, m.zeroBias, L, D, eps)

	qkv := s.qkv[:L*3*D]
	wi := s.upGate[:L*2*I] // Wi output [L,2I] (fused input/gate); reuses the GTE buffer
	// Banded attention fans its query tiles across cores, one private s.band slot
	// each. Under EncodeBatch every core already runs a sibling forward, so it takes
	// one slot and stays serial — the same contract the matmul gates use, and it
	// keeps B pooled scratches from each carrying NumCPU slots they'd never use.
	bandWorkers := 1
	if inflightForwards.Load() <= 1 {
		bandWorkers = numCPU
	}
	for li := range m.layers {
		l := &m.layers[li]

		// ── Attention (pre-norm residual) ──────────────────────────────
		// normed is the attention input: LayerNorm(h) for layers > 0, h itself for
		// layer 0 (identity pre-norm). It borrows s.out, which is dead until the
		// attention output projection below — the QKV matmul consumes normed first,
		// so reusing s.out for attnOut afterward is safe.
		var normed []float32
		if l.AttnLNW != nil {
			normed = s.out[:L*D]
			copy(normed, h)
			layerNorm(normed, l.AttnLNW, m.zeroBias, L, D, eps)
		} else {
			normed = h
		}

		s.mm(normed, l.Wqkv, qkv, L, D, 3*D)
		Q, K, V := s.Q[:L*D], s.K[:L*D], s.V[:L*D]
		for i := range L {
			base := i * 3 * D
			copy(Q[i*D:(i+1)*D], qkv[base:base+D])
			copy(K[i*D:(i+1)*D], qkv[base+D:base+2*D])
			copy(V[i*D:(i+1)*D], qkv[base+2*D:base+3*D])
		}
		// RoPE on Q and K (rotate_half, full head-dim) at absolute positions; local
		// and global layers differ only in theta (equal for bekko). V is untouched.
		if l.local {
			ropeL.apply(Q, c.Heads)
			ropeL.apply(K, c.Heads)
		} else {
			ropeG.apply(Q, c.Heads)
			ropeG.apply(K, c.Heads)
		}

		ctx := s.ctx[:L*D]
		qH, kH := s.qH[:L*headDim], s.kH[:L*headDim]
		vHT := s.vH[:headDim*L]
		ctxHead := s.ctxHead[:L*headDim]
		scores := s.scores[:L*L]
		banded := l.local && bandedAttnOK(m.be, m.slidingW, L)
		if banded {
			s.ensureBand(bandWorkers, mbBandQTile, mbBandQTile+2*m.slidingW, headDim)
		}
		for headIdx := 0; headIdx < c.Heads; headIdx++ {
			splitHead(Q, K, V, qH, kH, vHT, headIdx, L, D, headDim, scale)
			if banded {
				bandedAttnHead(s, m.slidingW, qH, kH, vHT, ctxHead, L, headDim, bandWorkers)
				for i := range L {
					copy(ctx[i*D+headIdx*headDim:i*D+headIdx*headDim+headDim], ctxHead[i*headDim:(i+1)*headDim])
				}
				continue
			}
			s.mm(qH, kH, scores, L, headDim, L)
			// Local layers restrict each query i to keys j with |i-j| ≤ S
			// (bidirectional sliding window); global layers attend everywhere.
			if l.local {
				S := m.slidingW
				for i := range L {
					row := scores[i*L : (i+1)*L]
					lo := i - S
					if lo < 0 {
						lo = 0
					}
					hi := i + S
					if hi > L-1 {
						hi = L - 1
					}
					for j := 0; j < lo; j++ {
						row[j] = mbMasked
					}
					for j := hi + 1; j < L; j++ {
						row[j] = mbMasked
					}
				}
			}
			softmaxRows(scores, L, L)
			s.mm(scores, vHT, ctxHead, L, L, headDim)
			for i := range L {
				copy(ctx[i*D+headIdx*headDim:i*D+headIdx*headDim+headDim], ctxHead[i*headDim:(i+1)*headDim])
			}
		}
		attnOut := s.out[:L*D]
		s.mm(ctx, l.AttnWo, attnOut, L, D, D)
		for i := range h {
			h[i] += attnOut[i] // residual (pre-norm: added to the un-normed h)
		}

		// ── MLP (pre-norm residual, GeGLU) ─────────────────────────────
		// normed2 borrows s.mid, dead until the MLP output projection: the Wi matmul
		// consumes it first, so reusing s.mid for ffn afterward is safe.
		normed2 := s.mid[:L*D]
		copy(normed2, h)
		layerNorm(normed2, l.MLPLNW, m.zeroBias, L, D, eps)

		s.mm(normed2, l.MLPWi, wi, L, D, 2*I)
		mid := s.val[:L*I]
		// GeGLU: input = wi[:I], gate = wi[I:2I]; mid = gelu(input) ⊙ gate. (The
		// reverse chunk order of gte.go, which does gelu(gate)·up.) Split by token
		// row — elementwise activation, numerically inert — so the per-element
		// math.Erf parallelizes (the largest serial stage after the softmax split).
		parallelRows(L, L*I, func(start, end int) {
			for i := start; i < end; i++ {
				input := wi[i*2*I : i*2*I+I]
				gate := wi[i*2*I+I : i*2*I+2*I]
				mr := mid[i*I : (i+1)*I]
				for j := range I {
					mr[j] = geluScalar(input[j]) * gate[j]
				}
			}
		})
		ffn := s.mid[:L*D]
		s.mm(mid, l.MLPWo, ffn, L, I, D)
		for i := range h {
			h[i] += ffn[i] // residual
		}
	}
	// final_norm after the last layer.
	layerNorm(h, m.finalLNW, m.zeroBias, L, D, eps)
	return h
}

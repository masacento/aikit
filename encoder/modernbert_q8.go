package encoder

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// modernbert_q8.go — int8-quantized ModernBERT forward. Mirrors modernbert.go
// exactly (pre-norm residual, bias-free, GeGLU, alternating local/global
// sliding-window attention, dual RoPE) but stores the four big per-layer
// projections (Wqkv, AttnWo, MLPWi, MLPWo) as per-row int8 + f32 scales via
// linalg.WeightMat. LayerNorm weights and the embedding table stay f32 (small,
// parity-sensitive — same rationale as WeightsQ8).
//
// The loader accepts both an f32-class checkpoint (quantized at load) and a
// pre-quantized one (I8 tensors + per-tensor companion scales, wrapped
// zero-copy off the mmap). Memory: ruri-v3-30m drops from ~140 MB (f32 mmap)
// to ~35 MB of int8 projection data, plus the f32 embedding/LN tail.

type modernBertLayerQ8 struct {
	Wqkv    linalg.WeightMat // [3*D, D]
	AttnWo  linalg.WeightMat // [D, D]
	MLPWi   linalg.WeightMat // [2*I, D]
	MLPWo   linalg.WeightMat // [D, I]
	AttnLNW []float32        // attention PRE-norm weight [D]; nil for layer 0 (identity)
	MLPLNW  []float32        // MLP PRE-norm weight [D]
	local   bool             // sliding-window attention (vs global)
}

// ModernBERTQ8 is a loaded ModernBERT encoder with int8-quantized projections.
// Immutable after load; the forward is read-only-safe for concurrent use.
type ModernBERTQ8 struct {
	be Backend

	cfg      modernBertConfig
	wordEmb  []float32 // [vocab, hidden] — stays f32 (embedding lookup, not matmul)
	embLNW   []float32 // embedding LayerNorm weight [hidden]
	layers   []modernBertLayerQ8
	finalLNW []float32 // post-last-layer LayerNorm weight [hidden]
	zeroBias []float32 // [hidden] zeros — bias-free layerNorm reuses the f32 kernel
	tok      Tokenizer
	maxSeq   int
	pool     pooling
	slidingW int     // sliding-window half-width S = local_attention/2
	thetaG   float64 // global-layer RoPE base
	thetaL   float64 // local-layer RoPE base
	// st is non-nil when the checkpoint was pre-quantized: the int8 weights
	// alias its mmap, so it must stay open until Close. Nil when the model was
	// quantized at load (all weights heap-owned, mmap already released).
	st    *embed.SafetensorsFile
	ropeG ropeCache
	ropeL ropeCache
}

// LoadModernBERTQ8 loads a ModernBERT encoder with int8 projections. Two
// checkpoint formats are accepted, told apart by the projection dtypes:
//
//   - PRE-QUANTIZED: the projections are stored I8 with symmetric scales in a
//     companion "<name>.scale" tensor (per-row [rows] preferred, per-tensor
//     [1] accepted). The int8 data is wrapped in place — zero-copy off the
//     mmap, no dequant→requantize round-trip, so the load pays no quantization
//     cost — and the mmap stays open to back it (Close releases it).
//   - F32-CLASS (F32/BF16/F16): the projections are loaded as f32 and
//     quantized to per-row int8 at load; the small f32 weights are cloned to
//     the heap and the mmap is released, leaving a self-contained model.
//
// Either way the four big per-layer projections end up per-row int8 + f32
// scales; LayerNorm weights and the embedding table stay f32 (small,
// parity-sensitive — same rationale as WeightsQ8).
func LoadModernBERTQ8(dir string) (*ModernBERTQ8, error) {
	c, err := loadModernBertConfig(dir)
	if err != nil {
		return nil, err
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("encoder: open ModernBERT safetensors: %w", err)
	}
	D, I := c.Hidden, c.Intermediate

	// Bare vs "model."-prefixed tensor names — the same probe LoadModernBERT does.
	prefix := ""
	if _, e := st.Tensor("embeddings.tok_embeddings.weight"); e != nil {
		if _, e2 := st.Tensor("model.embeddings.tok_embeddings.weight"); e2 == nil {
			prefix = "model."
		}
	}

	// Storage probe: I8 projections ⇒ a pre-quantized checkpoint (per-tensor
	// symmetric int8 + companion scales), wrapped in place below; anything
	// else ⇒ f32-class storage, quantized at load.
	preQ := false
	if t, e := st.Tensor(prefix + "layers.0.attn.Wqkv.weight"); e == nil && t.DType == "I8" {
		preQ = true
	}

	q := &ModernBERTQ8{
		cfg:      c,
		zeroBias: make([]float32, D),
		slidingW: c.LocalAttention / 2,
		thetaG:   c.GlobalRopeTheta,
		thetaL:   c.LocalRopeTheta,
		maxSeq:   c.MaxPos,
		layers:   make([]modernBertLayerQ8, c.Layers),
	}
	aliased := false // any int8 weight aliasing the mmap ⇒ st must stay open

	// getF32 loads a small weight as f32. TensorF32 widens BF16/F16 and
	// dequantizes I8-with-scale into fresh allocations; only an F32 result is
	// a zero-copy view, and it is cloned when the mmap will be released.
	getF32 := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32(name, want...)
		if err != nil {
			return nil
		}
		if !preQ {
			if t, e := st.Tensor(name); e == nil && t.DType == "F32" {
				v = cloneFloat32(v)
			}
		}
		return v
	}

	// getW loads one [rows, cols] projection as per-row int8. A pre-quantized
	// I8 tensor is wrapped in place and aliases the mmap; its companion scale
	// is used directly when per-row ([rows]) or broadcast when per-tensor
	// ([1]). Anything else is quantized from f32 at load.
	getW := func(name string, rows, cols int) linalg.WeightMat {
		if err != nil {
			return linalg.WeightMat{}
		}
		t, e := st.Tensor(name)
		if e != nil {
			err = e
			return linalg.WeightMat{}
		}
		if t.DType != "I8" {
			var w []float32
			w, err = st.TensorF32(name, rows, cols)
			if err != nil {
				return linalg.WeightMat{}
			}
			return linalg.QuantizeInt8(w, rows, cols, false)
		}
		if len(t.Shape) != 2 || t.Shape[0] != rows || t.Shape[1] != cols {
			err = fmt.Errorf("encoder: ModernBERT tensor %q shape %v != want [%d %d]", name, t.Shape, rows, cols)
			return linalg.WeightMat{}
		}
		q8, e := t.Int8s()
		if e != nil {
			err = e
			return linalg.WeightMat{}
		}
		sv, e := st.TensorF32(name + ".scale")
		if e != nil {
			err = fmt.Errorf("encoder: ModernBERT I8 tensor %q missing companion scale: %w", name, e)
			return linalg.WeightMat{}
		}
		var scales []float32
		switch len(sv) {
		case rows:
			// Per-row scales: an F32 view aliasing the mmap — kept open below.
			scales = sv
		case 1:
			// Per-tensor scale: broadcast to the per-row form WeightMat/mmq8
			// expect.
			scales = make([]float32, rows)
			for i := range scales {
				scales[i] = sv[0]
			}
		default:
			err = fmt.Errorf("encoder: ModernBERT I8 tensor %q scale has %d elements, want 1 (per-tensor) or %d (per-row)", name, len(sv), rows)
			return linalg.WeightMat{}
		}
		aliased = true
		return linalg.WrapInt8(q8, scales, rows, cols, false)
	}

	q.wordEmb = getF32(prefix+"embeddings.tok_embeddings.weight", c.VocabSize, D)
	q.embLNW = getF32(prefix+"embeddings.norm.weight", D)
	for i := range q.layers {
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		l := &q.layers[i]
		l.Wqkv = getW(p+"attn.Wqkv.weight", 3*D, D)
		l.AttnWo = getW(p+"attn.Wo.weight", D, D)
		// Layer 0's attention pre-norm is nn.Identity — no params in the checkpoint.
		if i > 0 {
			l.AttnLNW = getF32(p+"attn_norm.weight", D)
		}
		l.MLPWi = getW(p+"mlp.Wi.weight", 2*I, D)
		l.MLPWo = getW(p+"mlp.Wo.weight", D, I)
		l.MLPLNW = getF32(p+"mlp_norm.weight", D)
		l.local = i%c.GlobalEvery != 0
	}
	q.finalLNW = getF32(prefix+"final_norm.weight", D)
	if err != nil {
		_ = st.Close()
		return nil, err
	}

	// max sequence length: sentence-transformers right-truncates here; the hard
	// ceiling is the RoPE position capacity (max_position_embeddings).
	if sb, e := os.ReadFile(filepath.Join(dir, "sentence_bert_config.json")); e == nil {
		var v struct {
			MaxSeqLength int `json:"max_seq_length"`
		}
		if json.Unmarshal(sb, &v) == nil && v.MaxSeqLength > 0 {
			q.maxSeq = min(v.MaxSeqLength, c.MaxPos)
		}
	}
	// Tokenizer is best-effort (see LoadModernBERT).
	q.tok = loadModernBertTokenizer(dir)
	if q.pool, err = poolingFromConfig(dir, poolMean); err != nil {
		_ = st.Close()
		return nil, err
	}

	if preQ || aliased {
		// int8 weights alias the mapping; it backs the model until Close.
		q.st = st
	} else {
		// f32-class checkpoint: the weights are now int8 + heap copies; release
		// the mmap.
		_ = st.Close()
	}
	return q, nil
}

// Close releases the mmap backing a pre-quantized checkpoint's int8 weights.
// For a checkpoint quantized at load the mmap was already released and Close
// is a no-op. Idempotent; call once after the last Encode.
func (m *ModernBERTQ8) Close() error {
	if m.st == nil {
		return nil
	}
	return m.st.Close()
}

// HiddenDim is the output embedding dimension.
func (m *ModernBERTQ8) HiddenDim() int { return m.cfg.Hidden }

// Encode tokenizes text and returns the pooled, L2-normalized sentence embedding.
func (m *ModernBERTQ8) Encode(text string) ([]float32, error) {
	if m.tok == nil {
		return nil, fmt.Errorf("encoder: ModernBERTQ8 has no usable tokenizer; use Embed with pre-tokenized ids")
	}
	ids, err := m.tok.EncodeWithSpecials(text, m.maxSeq)
	if err != nil {
		return nil, err
	}
	return m.Embed(ids), nil
}

// Embed returns the pooled, L2-normalized sentence embedding for token ids.
func (m *ModernBERTQ8) Embed(ids []int32) []float32 {
	D := m.cfg.Hidden
	h := m.forward(ids)
	return l2norm(poolOne(h, len(h)/D, D, m.pool))
}

// forward runs the int8 ModernBERT transformer. It mirrors ModernBERT.forward
// exactly — same pre-norm residual, same sliding-window mask, same dual RoPE,
// same GeGLU — except the four big linears route through mmq8 (int8 weights
// dequantized once into scratch, then SIMD f32 matmul).
func (m *ModernBERTQ8) forward(ids []int32) []float32 {
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
	// MLPWi [2I, D] is the largest weight; ensureDeqW sizes to
	// max(intermediate, 3D)*D, so pass 2I to cover it.
	s.ensureDeqW(D, 2*I)
	ropeG := m.ropeG.get(L, headDim, m.thetaG)
	ropeL := m.ropeL.get(L, headDim, m.thetaL)

	// Embeddings: word lookup + bias-free LayerNorm (no position embedding).
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
	wi := s.upGate[:L*2*I]
	// Banded attention fans its query tiles across cores; under EncodeBatch a sibling
	// forward already owns every core, so it stays serial — see modernbert.go.
	bandWorkers := 1
	if inflightForwards.Load() <= 1 {
		bandWorkers = numCPU
	}
	for li := range m.layers {
		l := &m.layers[li]

		// ── Attention (pre-norm residual) ──────────────────────────────
		var normed []float32
		if l.AttnLNW != nil {
			normed = s.out[:L*D]
			copy(normed, h)
			layerNorm(normed, l.AttnLNW, m.zeroBias, L, D, eps)
		} else {
			normed = h
		}

		WqkvQ, WqkvScales, _, _ := l.Wqkv.Int8()
		s.mmq8(qkv, normed, WqkvQ, WqkvScales, L, D, 3*D)
		Q, K, V := s.Q[:L*D], s.K[:L*D], s.V[:L*D]
		for i := range L {
			base := i * 3 * D
			copy(Q[i*D:(i+1)*D], qkv[base:base+D])
			copy(K[i*D:(i+1)*D], qkv[base+D:base+2*D])
			copy(V[i*D:(i+1)*D], qkv[base+2*D:base+3*D])
		}
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
		// Quantization stops at the projections: attention is the same f32 code the
		// f32 forward runs, so it takes the same banded sliding-window path (and the
		// same L²-shaped saving) on local layers.
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
		WoQ, WoScales, _, _ := l.AttnWo.Int8()
		s.mmq8(attnOut, ctx, WoQ, WoScales, L, D, D)
		for i := range h {
			h[i] += attnOut[i]
		}

		// ── MLP (pre-norm residual, GeGLU) ─────────────────────────────
		normed2 := s.mid[:L*D]
		copy(normed2, h)
		layerNorm(normed2, l.MLPLNW, m.zeroBias, L, D, eps)

		WiQ, WiScales, _, _ := l.MLPWi.Int8()
		s.mmq8(wi, normed2, WiQ, WiScales, L, D, 2*I)
		mid := s.val[:L*I]
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
		MWoQ, MWoScales, _, _ := l.MLPWo.Int8()
		s.mmq8(ffn, mid, MWoQ, MWoScales, L, I, D)
		for i := range h {
			h[i] += ffn[i]
		}
	}
	layerNorm(h, m.finalLNW, m.zeroBias, L, D, eps)
	return h
}

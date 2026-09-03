package ner

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/embed"
)

// gliner2_boundary.go — the GLiNER2 boundary head: endpoint states at L+1
// boundary positions and per-query start/end/inside marginals, plus the shared
// classification and abstention projections. The candidate pool and reranker
// live in gliner2_pool.go; decoding in gliner2_decode.go.
//
// Reference: fastino-ai/GLiNER2 gliner2/models/boundary/{encoding,heads}.py and
// the shared-path branch of BoundaryHead.forward (model.py). Two config facts
// shrink that reference for this checkpoint and are asserted at load:
//
//   - candidate_pool="shared": inference runs DocumentCandidatePool +
//     SharedPoolScorer. The SparseBoundaryProposer/SparseBoundaryPairScorer path
//     (the only user of rotary endpoint features) runs only as a training-time
//     diagnostic, so rotary.py is NOT ported.
//   - candidate_attention_layers=0 and query_attention_layers=0: the scorer's
//     OverlapBiasedCandidateAttention / EvidenceConditionedQueryAttention stacks
//     are empty, so candidates never attend to each other.
//
// All affines use nn.Linear semantics: y = x·Wᵀ + b with W stored [out,in] — the
// same `linear` helper the v1 head uses. Everything is single-sample (B=1),
// which is all the Go API needs: batch calls loop.

const g2MaskLogit = -1.0e4 // gliner2/models/boundary/constants.py MASK_LOGIT

// g2LN is a LayerNorm's affine part (torch nn.LayerNorm; eps 1e-5, the training
// default — NOT deberta's 1e-7).
type g2LN struct {
	W, B []float32
}

func (n g2LN) apply(x []float32, rows, d int) {
	layerNorm32(x, rows, d, n.W, n.B, 1e-5)
}

// layerNorm32 normalizes each row of x ([rows,d], row-major) in place.
func layerNorm32(x []float32, rows, d int, w, b []float32, eps float64) {
	for i := range rows {
		row := x[i*d : (i+1)*d]
		var mean, varr float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(d)
		for _, v := range row {
			dv := float64(v) - mean
			varr += dv * dv
		}
		varr /= float64(d)
		inv := float32(1 / math.Sqrt(varr+eps))
		for j, v := range row {
			row[j] = float32((float64(v)-mean)*float64(inv)*float64(w[j]) + float64(b[j]))
		}
	}
}

// g2AttnBlock is one BoundaryAttentionBlock: pre-norm, fused QKV [3d,d],
// non-causal local self-attention over boundary positions, residual.
type g2AttnBlock struct {
	norm   g2LN
	qkv    linear // [3d, d]
	out    linear // [d, d]
	heads  int
	window int
}

// forward runs the block over states ([n,d], B=1) and returns the updated
// states. Keys are masked by the boundary mask and the diagonal is OR'd in so
// padded query rows keep one legal key (encoding.py:129-131); with every
// boundary valid, the only live restriction is the window.
func (b *g2AttnBlock) forward(states []float32, n, d int, mask []bool) []float32 {
	normed := append([]float32(nil), states...)
	b.norm.apply(normed, n, d)
	qkv := b.qkv.apply(normed, n) // per row: q at [0,d), k at [d,2d), v at [2d,3d)
	hd := d / b.heads
	scale := 1 / math.Sqrt(float64(hd))

	out := make([]float32, n*d)
	scores := make([]float64, n)
	for q := range n {
		if !mask[q] {
			continue // padded rows are re-zeroed by the encoder's final masking
		}
		qBase := q * 3 * d
		for h := range b.heads {
			qh := qBase + h*hd
			maxScore := math.Inf(-1)
			for k := range n {
				if !mask[k] || (k != q && b.window > 0 && absInt(k-q) > b.window) {
					scores[k] = math.Inf(-1)
					continue
				}
				kh := k*3*d + d + h*hd
				var s float64
				for t := range hd {
					s += float64(qkv[qh+t]) * float64(qkv[kh+t])
				}
				s *= scale
				scores[k] = s
				if s > maxScore {
					maxScore = s
				}
			}
			var sum float64
			for k := range n {
				if math.IsInf(scores[k], -1) {
					scores[k] = 0
					continue
				}
				e := math.Exp(scores[k] - maxScore)
				scores[k] = e
				sum += e
			}
			inv := 1 / sum
			dst := out[q*d+h*hd : q*d+(h+1)*hd]
			for k := range n {
				if scores[k] == 0 {
					continue
				}
				w := scores[k] * inv
				vh := k*3*d + 2*d + h*hd
				for t := range hd {
					dst[t] += float32(w) * qkv[vh+t]
				}
			}
		}
	}
	update := b.out.apply(out, n)
	res := append([]float32(nil), states...)
	for i := range res {
		res[i] += update[i]
	}
	return res
}

// maskAll is an all-valid mask.
func maskAll(n int) []bool {
	m := make([]bool, n)
	for i := range m {
		m[i] = true
	}
	return m
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// g2SwiGLU is one ResidualSwiGLU refinement block: pre-norm, the input
// projection's output split into value/gate halves, value·silu(gate), residual.
type g2SwiGLU struct {
	norm   g2LN
	input  linear // [2*hidden, d]
	output linear // [d, hidden]
}

func (s *g2SwiGLU) forward(states []float32, n, d int) []float32 {
	normed := append([]float32(nil), states...)
	s.norm.apply(normed, n, d)
	h := s.input.apply(normed, n) // [n, 2*hidden]
	hidden := len(h) / (n * 2)
	gated := make([]float32, n*hidden)
	for i := range n {
		row := h[i*2*hidden : (i+1)*2*hidden]
		for j := range hidden {
			v, g := row[j], row[hidden+j]
			gated[i*hidden+j] = v * g / (1 + float32(math.Exp(-float64(g))))
		}
	}
	update := s.output.apply(gated, n)
	res := append([]float32(nil), states...)
	for i := range res {
		res[i] += update[i]
	}
	return res
}

// gelu is torch nn.GELU's default (exact erf) activation.
func gelu(v float32) float32 {
	return 0.5 * v * (1 + float32(math.Erf(float64(v)/math.Sqrt2)))
}

// g2Head holds every head weight the boundary forward needs. Tensors the shared
// path never runs (the sparse proposer, pair_scorer, record/relation
// namespaces) are simply not loaded.
type g2Head struct {
	cfg gliner2BoundaryConfig

	// encoder
	leftProj, rightProj linear // 768 → d
	outProj             linear // 2d → d
	encLN               g2LN
	attn                []g2AttnBlock
	refin               []g2SwiGLU
	bos, eos            []float32

	// marginals (BoundaryQueryHead)
	startBProj, startQProj   linear // d→d, q→d
	endBProj, endQProj       linear
	insideTProj, insideQProj linear // 768→d

	// shared pool builder (DocumentCandidatePool)
	poolStartProj, poolEndProj linear // d→d

	// shared pool scorer (SharedPoolScorer)
	scStartProj, scEndProj linear // d→pair
	lengthProj             linear // 3→pair
	priorProj              linear // 1→pair
	contentValProj         linear // 768→content
	contentLN              g2LN
	contentProj            linear // content→pair
	candNorm               g2LN
	queryProj              linear // q→pair
	film                   linear // pair→2*pair
	filmOut0               linear // pair→64
	filmOut3               linear // 64→1

	// classification + abstention
	classifier0, classifier3 linear // 768→1536, 1536→1
	nullProj                 linear // q→1
}

// loadG2Head copies the head weights out of the checkpoint's safetensors file.
// The mapping is released on return, so every tensor is copied out (same
// posture as the v1 loadHead); the head is ~20 MB, noise next to the backbone.
func loadG2Head(st *embed.SafetensorsFile, cfg gliner2BoundaryConfig) (*g2Head, error) {
	switch {
	case cfg.BoundaryDim <= 0 || cfg.PairDim <= 0 || cfg.ContentDim <= 0:
		return nil, fmt.Errorf("ner: boundary_head config missing a required dim")
	case cfg.CandidatePool != "shared":
		return nil, fmt.Errorf("ner: candidate_pool=%q unsupported (shared only)", cfg.CandidatePool)
	case cfg.CandidateBudget != cfg.PoolSize:
		return nil, fmt.Errorf("ner: candidate_budget %d != pool_size %d; the shared-pool "+
			"path assumes they agree", cfg.CandidateBudget, cfg.PoolSize)
	case cfg.CandidateAttentionLayers != 0 || cfg.QueryAttentionLayers != 0:
		return nil, fmt.Errorf("ner: candidate/query attention layers (%d/%d) unsupported — "+
			"only the attention-free shared scorer this file implements",
			cfg.CandidateAttentionLayers, cfg.QueryAttentionLayers)
	}
	var ferr error
	get := func(name string, want ...int) []float32 {
		if ferr != nil {
			return nil
		}
		v, err := st.TensorF32(name, want...)
		if err != nil {
			ferr = err
			return nil
		}
		out := make([]float32, len(v))
		copy(out, v)
		return out
	}
	const p = "boundary_head."
	// lin/lnc take names RELATIVE to the boundary_head. prefix; the classifier
	// and null_projection live at the top level and use get directly.
	lin := func(name string, in, out int) linear {
		return linear{W: get(p+name+".weight", out, in), B: get(p+name+".bias", out), in: in, out: out}
	}
	lnc := func(name string, d int) g2LN {
		return g2LN{W: get(p+name+".weight", d), B: get(p+name+".bias", d)}
	}

	d := cfg.BoundaryDim
	h := &g2Head{cfg: cfg}
	H := 768 // backbone hidden; asserted against the loaded encoder by the caller
	h.leftProj = lin("boundary_encoder.left_projection", H, d)
	h.rightProj = lin("boundary_encoder.right_projection", H, d)
	h.outProj = lin("boundary_encoder.output_projection", 2*d, d)
	h.encLN = lnc("boundary_encoder.layer_norm", d)
	h.bos = get(p+"boundary_encoder.bos_state", H)
	h.eos = get(p+"boundary_encoder.eos_state", H)
	for i := range cfg.BoundaryAttentionLayers {
		h.attn = append(h.attn, g2AttnBlock{
			norm:   lnc(fmt.Sprintf("boundary_encoder.attention_blocks.%d.norm", i), d),
			qkv:    lin(fmt.Sprintf("boundary_encoder.attention_blocks.%d.qkv_projection", i), d, 3*d),
			out:    lin(fmt.Sprintf("boundary_encoder.attention_blocks.%d.output_projection", i), d, d),
			heads:  cfg.BoundaryAttentionHeads,
			window: cfg.BoundaryAttentionWindow,
		})
	}
	for i := range cfg.BoundaryRefinementLayers {
		// ResidualSwiGLU: hidden_dim = int(dim * boundary_ffn_multiplier); the
		// input projection widens to 2*hidden_dim and is chunked into halves.
		hiddenDim := d * int(cfg.BoundaryFFNMultiplier)
		h.refin = append(h.refin, g2SwiGLU{
			norm:   lnc(fmt.Sprintf("boundary_encoder.refinement_blocks.%d.norm", i), d),
			input:  lin(fmt.Sprintf("boundary_encoder.refinement_blocks.%d.input_projection", i), d, 2*hiddenDim),
			output: lin(fmt.Sprintf("boundary_encoder.refinement_blocks.%d.output_projection", i), hiddenDim, d),
		})
	}

	h.startBProj = lin("boundary_query_head.start_boundary_projection", d, d)
	h.startQProj = lin("boundary_query_head.start_query_projection", H, d)
	h.endBProj = lin("boundary_query_head.end_boundary_projection", d, d)
	h.endQProj = lin("boundary_query_head.end_query_projection", H, d)
	h.insideTProj = lin("boundary_query_head.inside_text_projection", H, d)
	h.insideQProj = lin("boundary_query_head.inside_query_projection", H, d)

	h.poolStartProj = lin("shared_pool_builder.start_projection", d, d)
	h.poolEndProj = lin("shared_pool_builder.end_projection", d, d)

	pd := cfg.PairDim
	h.scStartProj = lin("shared_pool_scorer.start_projection", d, pd)
	h.scEndProj = lin("shared_pool_scorer.end_projection", d, pd)
	h.lengthProj = lin("shared_pool_scorer.length_projection", 3, pd)
	h.priorProj = lin("shared_pool_scorer.prior_projection", 1, pd)
	h.contentValProj = lin("shared_pool_scorer.content_pooler.value_projection", H, cfg.ContentDim)
	h.contentLN = lnc("shared_pool_scorer.content_pooler.layer_norm", cfg.ContentDim)
	h.contentProj = lin("shared_pool_scorer.content_projection", cfg.ContentDim, pd)
	h.candNorm = lnc("shared_pool_scorer.candidate_norm", pd)
	h.queryProj = lin("shared_pool_scorer.query_projection", H, pd)
	h.film = lin("shared_pool_scorer.film", pd, 2*pd)
	h.filmOut0 = lin("shared_pool_scorer.film_output.0", pd, 64)
	h.filmOut3 = lin("shared_pool_scorer.film_output.3", 64, 1)

	h.classifier0 = linear{W: get("classifier.0.weight", 1536, H), B: get("classifier.0.bias", 1536), in: H, out: 1536}
	h.classifier3 = linear{W: get("classifier.3.weight", 1, 1536), B: get("classifier.3.bias", 1), in: 1536, out: 1}
	h.nullProj = linear{W: get(p+"null_projection.weight", 1, H), B: get(p+"null_projection.bias", 1), in: H, out: 1}

	if ferr != nil {
		return nil, fmt.Errorf("ner: load boundary head: %w", ferr)
	}
	return h, nil
}

// g2BoundaryStates is the boundary encoder output for one sample.
type g2BoundaryStates struct {
	States []float32 // [N, d], N = L+1
	Mask   []bool    // [N]
}

// boundaryForward is BoundaryEncoder.forward for B=1: build the L+1 boundary
// positions from shifted text states (BOS/EOS at the ends), project, then refine
// with the attention and SwiGLU stacks. text_lengths == L, so every boundary
// 0..L is valid; the final mask multiply is still applied for shape parity.
func (h *g2Head) boundaryForward(text []float32, L, H int) g2BoundaryStates {
	d := h.cfg.BoundaryDim
	N := L + 1
	left := make([]float32, N*H)
	right := make([]float32, N*H)
	copy(left[H:], text[:L*H]) // left(i) = token i-1
	copy(left[0:H], h.bos)     // left(0) = BOS
	copy(right, text[:L*H])    // right(i) = token i
	copy(right[L*H:], h.eos)   // right(L) = EOS

	leftP := h.leftProj.apply(left, N)
	rightP := h.rightProj.apply(right, N)
	cat := make([]float32, N*2*d)
	for i := range N {
		copy(cat[i*2*d:i*2*d+d], leftP[i*d:(i+1)*d])
		copy(cat[i*2*d+d:(i+1)*2*d], rightP[i*d:(i+1)*d])
	}
	states := h.outProj.apply(cat, N)
	h.encLN.apply(states, N, d)

	mask := maskAll(N)
	for i := range h.attn {
		states = h.attn[i].forward(states, N, d, mask)
	}
	for i := range h.refin {
		states = h.refin[i].forward(states, N, d)
	}
	for i := range mask { // zero padding boundaries (none for B=1, kept for parity)
		if !mask[i] {
			for j := i * d; j < (i+1)*d; j++ {
				states[j] = 0
			}
		}
	}
	return g2BoundaryStates{States: states, Mask: mask}
}

// g2Marginals is BoundaryMarginals for one sample.
type g2Marginals struct {
	Q, N, L int       // queries, boundaries (L+1), text words
	Start   []float32 // [Q,N]
	End     []float32 // [Q,N]
	Inside  []float32 // [Q,L]
	Prefix  []float32 // [Q,N] fp32 centered cumsum, leading zero
	Mean    []float32 // [Q] per-query inside mean
}

// marginalForward is BoundaryQueryHead.forward: scaled dot-product marginals of
// projected boundary/text states against projected query states, plus the
// mean-centered fp32 inside prefix (heads.py:79-97).
func (h *g2Head) marginalForward(bs g2BoundaryStates, text []float32, L int, tmask []bool, queries []float32, Q int) *g2Marginals {
	d := h.cfg.BoundaryDim
	N := L + 1
	scale := 1 / math.Sqrt(float64(d))

	m := &g2Marginals{Q: Q, N: N, L: L,
		Start: make([]float32, Q*N), End: make([]float32, Q*N),
		Inside: make([]float32, Q*L), Prefix: make([]float32, Q*N),
		Mean: make([]float32, Q)}

	startB := h.startBProj.apply(bs.States, N)
	startQ := h.startQProj.apply(queries, Q)
	endB := h.endBProj.apply(bs.States, N)
	endQ := h.endQProj.apply(queries, Q)
	insideT := h.insideTProj.apply(text, L)
	insideQ := h.insideQProj.apply(queries, Q)

	dot := func(a, b []float32) float32 {
		var s float32
		for i := range a {
			s += a[i] * b[i]
		}
		return s
	}
	for q := range Q {
		for i := range N {
			v := float32(float64(dot(startB[i*d:(i+1)*d], startQ[q*d:(q+1)*d])) * scale)
			if !bs.Mask[i] {
				v = g2MaskLogit
			}
			m.Start[q*N+i] = v
			v = float32(float64(dot(endB[i*d:(i+1)*d], endQ[q*d:(q+1)*d])) * scale)
			if !bs.Mask[i] {
				v = g2MaskLogit
			}
			m.End[q*N+i] = v
		}
		var sum float32
		valid := 0
		for l := range L {
			v := float32(float64(dot(insideT[l*d:(l+1)*d], insideQ[q*d:(q+1)*d])) * scale)
			if !tmask[l] {
				v = g2MaskLogit
			} else {
				valid++
			}
			m.Inside[q*L+l] = v
			if !tmask[l] {
				continue // masked positions are zeroed before the prefix sum
			}
			sum += v
		}
		if valid == 0 {
			valid = 1
		}
		mean := sum / float32(valid)
		m.Mean[q] = mean
		m.Prefix[q*N] = 0
		for l := range L {
			c := float32(0)
			if tmask[l] {
				c = m.Inside[q*L+l] - mean
			}
			m.Prefix[q*N+l+1] = m.Prefix[q*N+l] + c
		}
	}
	return m
}

// classify is the shared classification MLP applied per [L]-marker state:
// Linear(768→1536) → GELU → Linear(1536→1) (classifier.{0,3}).
func (h *g2Head) classify(markerStates []float32, rows int) []float32 {
	hid := h.classifier0.apply(markerStates, rows)
	for i := range hid {
		hid[i] = gelu(hid[i])
	}
	return h.classifier3.apply(hid, rows)
}

// nullLogit is the abstention head: one scalar per query state.
func (h *g2Head) nullLogit(queryStates []float32, Q int) []float32 {
	return h.nullProj.apply(queryStates, Q)
}

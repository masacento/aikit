package embed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// ──────────────────────────────────────────────────────────────────────────────
// spm.model (raw SentencePiece ModelProto) → unigramBackend
// ──────────────────────────────────────────────────────────────────────────────
//
// Some checkpoints ship only a raw `spm.model` and no tokenizer.json — notably
// microsoft/mdeberta-v3-base and the GLiNER models built on it. The pipeline they
// need is the one tokenize_unigram.go already implements (Precompiled charsmap →
// Metaspace → Unigram Viterbi → byte_fallback), so this file is a reader, not a
// second tokenizer: it decodes the ModelProto into the same unigramBackend.
//
// The decode is a hand-rolled wire walk rather than a protobuf dependency. The
// fields we need are three top-level ones and two nested messages, all varint or
// length-delimited, and google.golang.org/protobuf would pull a code generator into
// a package that has none:
//
//	ModelProto {
//	  1 repeated SentencePiece pieces { 1 string piece; 2 float score; 3 enum type }
//	  2 TrainerSpec    { 35 bool byte_fallback; 40 unk_id; 41 bos_id; 42 eos_id; 43 pad_id }
//	  3 NormalizerSpec { 2 bytes precompiled_charsmap; 3 bool add_dummy_prefix;
//	                     4 bool remove_extra_whitespaces }
//	}
//
// Unknown fields are skipped by wire type, so a newer sentencepiece writing extra
// fields still loads.

// SentencePiece piece types (ModelProto.SentencePiece.Type). The distinction is
// load-bearing, not bookkeeping: sentencepiece's own Model ctor puts only NORMAL,
// USER_DEFINED and UNUSED into the searchable piece table, and routes UNKNOWN,
// CONTROL and BYTE into a reserved map the Viterbi never matches against. Treating
// them uniformly would let the literal text "[CLS]" tokenize to the control id, and
// "<0x41>" to a byte id — neither of which sentencepiece ever emits from text.
const (
	spmTypeNormal      = 1
	spmTypeUnknown     = 2
	spmTypeControl     = 3
	spmTypeUserDefined = 4
	spmTypeUnused      = 5
	spmTypeByte        = 6
)

// spmModel is the decoded subset of a sentencepiece ModelProto.
type spmModel struct {
	pieces []string
	scores []float64
	types  []int32

	charsmap       []byte
	addDummyPrefix bool
	removeExtraWS  bool

	byteFallback bool
	unkID        int32
	bosID        int32
	eosID        int32
}

// protoScanner is a bounds-checked protobuf wire reader. Every accessor returns an
// error rather than panicking so a truncated or hostile spm.model fails the load
// instead of the process (same posture as the safetensors/GGUF readers).
type protoScanner struct {
	b   []byte
	pos int
}

func (p *protoScanner) done() bool { return p.pos >= len(p.b) }

func (p *protoScanner) varint() (uint64, error) {
	var v uint64
	var shift uint
	for {
		if p.pos >= len(p.b) {
			return 0, fmt.Errorf("truncated varint")
		}
		if shift > 63 {
			return 0, fmt.Errorf("varint overflows 64 bits")
		}
		c := p.b[p.pos]
		p.pos++
		v |= uint64(c&0x7f) << shift
		if c&0x80 == 0 {
			return v, nil
		}
		shift += 7
	}
}

// tag reads a field key and returns its field number and wire type.
func (p *protoScanner) tag() (field int, wire int, err error) {
	key, err := p.varint()
	if err != nil {
		return 0, 0, err
	}
	return int(key >> 3), int(key & 7), nil
}

// bytes reads a length-delimited field's payload as a sub-slice of the backing
// buffer (no copy — callers that retain it must copy or accept the aliasing).
func (p *protoScanner) bytes() ([]byte, error) {
	n, err := p.varint()
	if err != nil {
		return nil, err
	}
	if n > uint64(len(p.b)-p.pos) {
		return nil, fmt.Errorf("length-delimited field overruns buffer (%d > %d)", n, len(p.b)-p.pos)
	}
	out := p.b[p.pos : p.pos+int(n)]
	p.pos += int(n)
	return out, nil
}

func (p *protoScanner) fixed32() (uint32, error) {
	if len(p.b)-p.pos < 4 {
		return 0, fmt.Errorf("truncated fixed32")
	}
	v := binary.LittleEndian.Uint32(p.b[p.pos:])
	p.pos += 4
	return v, nil
}

// skip advances past a field of the given wire type.
func (p *protoScanner) skip(wire int) error {
	switch wire {
	case 0:
		_, err := p.varint()
		return err
	case 1:
		if len(p.b)-p.pos < 8 {
			return fmt.Errorf("truncated fixed64")
		}
		p.pos += 8
		return nil
	case 2:
		_, err := p.bytes()
		return err
	case 5:
		_, err := p.fixed32()
		return err
	default:
		return fmt.Errorf("unsupported wire type %d", wire)
	}
}

// parseSPM decodes a raw spm.model.
func parseSPM(data []byte) (*spmModel, error) {
	m := &spmModel{unkID: -1, bosID: -1, eosID: -1}
	// sentencepiece's own defaults, applied when NormalizerSpec omits the field.
	m.addDummyPrefix = true
	m.removeExtraWS = true

	s := &protoScanner{b: data}
	for !s.done() {
		field, wire, err := s.tag()
		if err != nil {
			return nil, fmt.Errorf("spm.model: %w", err)
		}
		if wire != 2 {
			if err := s.skip(wire); err != nil {
				return nil, fmt.Errorf("spm.model: %w", err)
			}
			continue
		}
		payload, err := s.bytes()
		if err != nil {
			return nil, fmt.Errorf("spm.model: %w", err)
		}
		switch field {
		case 1: // repeated SentencePiece
			piece, score, typ, perr := parseSPMPiece(payload)
			if perr != nil {
				return nil, fmt.Errorf("spm.model: piece %d: %w", len(m.pieces), perr)
			}
			m.pieces = append(m.pieces, piece)
			m.scores = append(m.scores, score)
			m.types = append(m.types, typ)
		case 2: // TrainerSpec
			if err := m.parseTrainerSpec(payload); err != nil {
				return nil, fmt.Errorf("spm.model: trainer_spec: %w", err)
			}
		case 3: // NormalizerSpec
			if err := m.parseNormalizerSpec(payload); err != nil {
				return nil, fmt.Errorf("spm.model: normalizer_spec: %w", err)
			}
		}
	}
	if len(m.pieces) == 0 {
		return nil, fmt.Errorf("spm.model: no pieces")
	}
	return m, nil
}

// parseSPMPiece decodes one SentencePiece message. A piece with no explicit type
// is NORMAL (proto2 default).
func parseSPMPiece(b []byte) (piece string, score float64, typ int32, err error) {
	typ = spmTypeNormal
	s := &protoScanner{b: b}
	for !s.done() {
		field, wire, terr := s.tag()
		if terr != nil {
			return "", 0, 0, terr
		}
		switch {
		case field == 1 && wire == 2:
			v, berr := s.bytes()
			if berr != nil {
				return "", 0, 0, berr
			}
			piece = string(v)
		case field == 2 && wire == 5:
			v, ferr := s.fixed32()
			if ferr != nil {
				return "", 0, 0, ferr
			}
			score = float64(math.Float32frombits(v))
		case field == 3 && wire == 0:
			v, verr := s.varint()
			if verr != nil {
				return "", 0, 0, verr
			}
			typ = int32(v)
		default:
			if serr := s.skip(wire); serr != nil {
				return "", 0, 0, serr
			}
		}
	}
	return piece, score, typ, nil
}

func (m *spmModel) parseTrainerSpec(b []byte) error {
	s := &protoScanner{b: b}
	for !s.done() {
		field, wire, err := s.tag()
		if err != nil {
			return err
		}
		if wire != 0 {
			if err := s.skip(wire); err != nil {
				return err
			}
			continue
		}
		v, err := s.varint()
		if err != nil {
			return err
		}
		switch field {
		case 35:
			m.byteFallback = v != 0
		case 40:
			m.unkID = int32(v)
		case 41:
			m.bosID = int32(v)
		case 42:
			m.eosID = int32(v)
		}
	}
	return nil
}

func (m *spmModel) parseNormalizerSpec(b []byte) error {
	s := &protoScanner{b: b}
	for !s.done() {
		field, wire, err := s.tag()
		if err != nil {
			return err
		}
		switch {
		case field == 2 && wire == 2:
			v, berr := s.bytes()
			if berr != nil {
				return berr
			}
			m.charsmap = v
		case field == 3 && wire == 0:
			v, verr := s.varint()
			if verr != nil {
				return verr
			}
			m.addDummyPrefix = v != 0
		case field == 4 && wire == 0:
			v, verr := s.varint()
			if verr != nil {
				return verr
			}
			m.removeExtraWS = v != 0
		default:
			if serr := s.skip(wire); serr != nil {
				return serr
			}
		}
	}
	return nil
}

// ──────────────────────────────────────────────────────────────────────────────
// Public loader
// ──────────────────────────────────────────────────────────────────────────────

// LoadTokenizerSPM builds a Tokenizer from a raw sentencepiece model, for
// checkpoints that ship spm.model instead of tokenizer.json (mDeBERTa-v3, GLiNER).
//
// addedTokensPath is an optional HF added_tokens.json ({"<<ENT>>": 250103, ...});
// pass "" when there is none. Those ids sit above the spm vocab and are carved out
// of the text before the Viterbi runs, exactly as HF's slow tokenizer does.
//
// The template specials come from the model's own bos_id/eos_id — for mDeBERTa-v3
// that is [CLS] … [SEP], which is the wrapping DeBERTa-v2 expects.
func LoadTokenizerSPM(spmPath, addedTokensPath string) (*Tokenizer, error) {
	data, err := os.ReadFile(spmPath)
	if err != nil {
		return nil, fmt.Errorf("read spm.model: %w", err)
	}
	var added map[string]int32
	if addedTokensPath != "" {
		raw, rerr := os.ReadFile(addedTokensPath)
		if rerr != nil {
			return nil, fmt.Errorf("read added_tokens.json: %w", rerr)
		}
		if jerr := json.Unmarshal(raw, &added); jerr != nil {
			return nil, fmt.Errorf("parse added_tokens.json: %w", jerr)
		}
	}
	uni, err := buildSPMBackend(data, added)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{uni: uni, unkID: uni.unkID}, nil
}

// buildSPMBackend converts a decoded spm.model (plus any added tokens) into the
// shared unigramBackend.
func buildSPMBackend(data []byte, added map[string]int32) (*unigramBackend, error) {
	m, err := parseSPM(data)
	if err != nil {
		return nil, err
	}
	if m.unkID < 0 || int(m.unkID) >= len(m.pieces) {
		return nil, fmt.Errorf("spm.model: unk_id %d out of range (%d pieces)", m.unkID, len(m.pieces))
	}

	var norm *precompiled
	if len(m.charsmap) > 0 {
		if norm, err = newPrecompiledBytes(m.charsmap); err != nil {
			return nil, err
		}
	}

	// Searchable pieces only (see the type constants above). minScore tracks NORMAL
	// pieces alone, matching sentencepiece's Model ctor — the unknown penalty is
	// derived from it, so including a control piece's 0.0 score here would silently
	// change every <unk> cost.
	piece2id := make(map[string]int32, len(m.pieces))
	minScore := 0.0
	maxBytes := 1
	for i, piece := range m.pieces {
		if piece == "" {
			continue
		}
		switch m.types[i] {
		case spmTypeNormal, spmTypeUserDefined, spmTypeUnused:
			piece2id[piece] = int32(i)
			if len(piece) > maxBytes {
				maxBytes = len(piece)
			}
			if m.types[i] == spmTypeNormal && m.scores[i] < minScore {
				minScore = m.scores[i]
			}
		}
	}

	// byte_fallback: map each raw byte to its "<0xNN>" piece. Those pieces are type
	// BYTE and therefore absent from piece2id, so they are reachable only through
	// this table — which is the point.
	var byteIDs [256]int32
	if m.byteFallback {
		byPiece := make(map[string]int32, 256)
		for i, piece := range m.pieces {
			if m.types[i] == spmTypeByte {
				byPiece[piece] = int32(i)
			}
		}
		for b := range 256 {
			piece := fmt.Sprintf("<0x%02X>", b)
			id, ok := byPiece[piece]
			if !ok {
				return nil, fmt.Errorf("spm.model: byte_fallback set but no BYTE piece %q", piece)
			}
			byteIDs[b] = id
		}
	}

	vocabSize := len(m.pieces)
	addedKeys := make([]string, 0, len(added))
	for k, id := range added {
		addedKeys = append(addedKeys, k)
		if int(id) >= vocabSize {
			vocabSize = int(id) + 1
		}
	}
	// Longest-first so a literal that prefixes another can't shadow it.
	sort.Slice(addedKeys, func(i, j int) bool {
		if len(addedKeys[i]) != len(addedKeys[j]) {
			return len(addedKeys[i]) > len(addedKeys[j])
		}
		return addedKeys[i] < addedKeys[j]
	})

	var prefixIDs, suffixIDs []int32
	if m.bosID >= 0 {
		prefixIDs = []int32{m.bosID}
	}
	if m.eosID >= 0 {
		suffixIDs = []int32{m.eosID}
	}

	return &unigramBackend{
		norm:      norm,
		metaspace: metaspaceKindSPM,
		model: &unigram{
			piece2id:     piece2id,
			scores:       m.scores,
			unkID:        m.unkID,
			minScore:     minScore,
			maxBytes:     maxBytes,
			fuseUnk:      true,
			byteFallback: m.byteFallback,
			byteIDs:      byteIDs,
		},
		addedTokens:       added,
		addedKeys:         addedKeys,
		unkID:             m.unkID,
		vocabSize:         vocabSize,
		prefixIDs:         prefixIDs,
		suffixIDs:         suffixIDs,
		spmAddDummyPrefix: m.addDummyPrefix,
		spmRemoveExtraWS:  m.removeExtraWS,
	}, nil
}

// EncodeWords tokenizes each word independently and concatenates the result,
// reporting where each word's sub-tokens begin. This is HF's
// `is_split_into_words=True`, and it is what a word-level head needs: GLiNER pools
// each word to its FIRST sub-token (`subtoken_pooling: "first"`), so it needs the
// word → token-index map that Encode cannot provide (this package implements no
// offsets — see the Tokenizer doc comment).
//
// No template specials are added; the caller wraps the sequence itself, because a
// prompt-based head interleaves its own markers with the words.
//
// firstSubtok[i] is the index into ids of word i's first sub-token, or -1 when the
// word produced none (an empty or all-whitespace word normalizes away). Callers
// must handle -1 rather than indexing blindly — that would silently pool the NEXT
// word's representation into this one.
//
// Backend-agnostic: it drives the public Encode, so it works for every tokenizer
// this package supports, not just the SentencePiece path it was introduced for.
func (t *Tokenizer) EncodeWords(words []string) (ids []int32, firstSubtok []int32) {
	firstSubtok = make([]int32, len(words))
	for i, w := range words {
		sub := t.Encode(w)
		if len(sub) == 0 {
			firstSubtok[i] = -1
			continue
		}
		firstSubtok[i] = int32(len(ids))
		ids = append(ids, sub...)
	}
	return ids, firstSubtok
}

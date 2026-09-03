package cjk

import "unicode/utf8"

// Packed integer feature keys for the segmentation hot loop.
//
// Rendering a feature string ("UW4:こ") and hashing it costs an allocation and
// a variable-length hash per template per character. Instead, every feature the
// template table can generate is encoded in a uint64, and a trained model's
// string keys are converted to those packed keys once at load time
// (buildPackedModel). Segment then scores a position with one map probe per
// template and no allocation at all. This mirrors litsea's packed_model.rs.
//
// The table order in templates.go is load-bearing: Segment sums float64 weights
// in emission order and float addition is not associative, so the packed path
// must visit templates in exactly the order the string writer emitted them to
// stay bit-for-bit identical.

// Boundary-tag ids. TAG_U is the padding/unknown tag, TAG_B marks a word start
// and TAG_O a word continuation.
const (
	tagU uint8 = 0
	tagB uint8 = 1
	tagO uint8 = 2
)

// tagStrs is the tag strings indexed by tag id.
var tagStrs = [3]string{"U", "B", "O"}

// sentinels are the padding strings in context order: B3/B2/B1 precede the
// text, E1/E2/E3 follow it. Index k maps to char code sentinelBase + k.
var sentinels = [6]string{"B3", "B2", "B1", "E1", "E2", "E3"}

// sentinelBase is the first sentinel char code, directly above the Unicode
// scalar range (the largest code point is U+10FFFF), so sentinel codes can
// never collide with a real character's code point.
const sentinelBase uint32 = 0x11_0000

// pack packs this template's feature at position i into a uint64 key.
//
// Layout: the template id occupies the top byte (bits 56..64); the slot values
// are shift-accumulated below it in slot order — 8 bits per tag/type slot,
// 24 bits per char slot. The widest template (BW*, two char slots) uses 48 slot
// bits, so the payload never reaches the id byte, and every slot value is
// strictly below its field width (tag ids < 3, type ids < 256, char codes
// <= 0x110005 < 2^24). The encoding is therefore injective over
// (template, slot values).
func (t *template) pack(i int, tags []uint8, chars []uint32, types []uint8) uint64 {
	var acc uint64
	for _, sl := range t.slots {
		idx := i - 3 + int(sl.delta)
		switch sl.kind {
		case slotTag:
			acc = acc<<8 | uint64(tags[idx])
		case slotTyp:
			acc = acc<<8 | uint64(types[idx])
		case slotChr:
			acc = acc<<24 | uint64(chars[idx])
		}
	}
	return uint64(t.id)<<56 | acc
}

// parseFeatureKeys parses a model feature string against language's templates
// and appends the packed key of every slot-value tuple that renders exactly
// this string. Strings no template can generate (unknown prefix, foreign type
// codes, leftover input) append nothing — such features are unreachable from
// the attribute writer for this language, so omitting them from the packed map
// cannot change any score.
//
// With the current grammar every parse is unique (tags are single characters,
// type-code sets are prefix-free, and sentinel-vs-char choices are forced by
// full consumption); the exhaustive collection is kept for robustness against
// future template changes and is pinned by tests.
func parseFeatureKeys(language Language, feature string, keys []uint64) []uint64 {
	tmpls := templatesFor(language)
	for i := range tmpls {
		tmpl := &tmpls[i]
		rest, ok := cutPrefix(feature, tmpl.prefix)
		if !ok {
			continue
		}
		if rest, ok = cutPrefix(rest, ":"); !ok {
			continue
		}
		keys = parseSlots(language, tmpl.slots, rest, uint64(tmpl.id)<<56, 0, keys)
	}
	return keys
}

// parseSlots is the recursive helper for parseFeatureKeys: it tries every way
// the next slot could have rendered the head of rest, and appends base|acc as a
// complete key when all slots and all input are consumed. base carries the
// fixed template-id byte (bits 56..64) and is never shifted; acc accumulates
// the slot values exactly as pack does.
func parseSlots(language Language, slots []slot, rest string, base, acc uint64, keys []uint64) []uint64 {
	if len(slots) == 0 {
		if rest == "" {
			keys = append(keys, base|acc)
		}
		return keys
	}
	sl, tail := slots[0], slots[1:]
	switch sl.kind {
	case slotTag:
		for id, tag := range tagStrs {
			if r, ok := cutPrefix(rest, tag); ok {
				keys = parseSlots(language, tail, r, base, acc<<8|uint64(id), keys)
			}
		}
	case slotTyp:
		for id, code := range language.typeCodes() {
			if r, ok := cutPrefix(rest, code); ok {
				keys = parseSlots(language, tail, r, base, acc<<8|uint64(id), keys)
			}
		}
	case slotChr:
		for k, sentinel := range sentinels {
			if r, ok := cutPrefix(rest, sentinel); ok {
				code := sentinelBase + uint32(k)
				keys = parseSlots(language, tail, r, base, acc<<24|uint64(code), keys)
			}
		}
		if c, size := utf8.DecodeRuneInString(rest); size > 0 {
			keys = parseSlots(language, tail, rest[size:], base, acc<<24|uint64(c), keys)
		}
	}
	return keys
}

// cutPrefix reports whether s starts with prefix and returns the remainder.
func cutPrefix(s, prefix string) (string, bool) {
	if len(s) < len(prefix) || s[:len(prefix)] != prefix {
		return s, false
	}
	return s[len(prefix):], true
}

// packedModel is a trained AdaBoost model compiled to packed integer feature
// keys for allocation-free scoring in Segment's hot loop.
//
// The lookup table is a flat open-addressed map rather than a Go map: the hot
// loop does one probe per template per character and nothing else, so the
// generic map's function-call and bucket-walk overhead dominates. Entries are
// key/value pairs in a single array so that a probe touches one cache line.
type packedModel struct {
	entries []packedEntry
	// mask wraps a probe index; shift turns a hash into the initial index.
	mask  uint64
	shift uint
}

// packedEntry is one slot of the open-addressed table. key holds the packed
// feature key biased by one so that a zero key marks an empty slot (the packed
// key 0 itself is reachable: template UP1 with tag "U").
type packedEntry struct {
	key    uint64
	weight float64
}

// hashMultiplier is the odd 64-bit constant of Fibonacci hashing; multiplying
// by it and taking the high bits mixes the sparse packed keys well enough that
// linear probing stays short.
const hashMultiplier uint64 = 0x9E37_79B9_7F4A_7C15

// tableSize returns the power-of-two capacity that holds n entries at a load
// factor of at most 1/2 — short probe sequences, and always at least one empty
// slot to terminate them — together with the shift that turns a hash into an
// index in range.
func tableSize(n int) (capacity uint64, shift uint) {
	capacity, shift = 8, 61
	for capacity < 2*uint64(n)+2 {
		capacity *= 2
		shift--
	}
	return capacity, shift
}

// weight returns the model weight of a packed feature key, or 0.0 if the key
// is not in the model. A miss adding 0.0 keeps the float64 accumulation
// sequence identical to the string-keyed path.
func (p *packedModel) weight(key uint64) float64 {
	if p.entries == nil {
		return 0
	}
	k := key + 1
	i := (k * hashMultiplier) >> p.shift
	for {
		e := &p.entries[i]
		if e.key == k {
			return e.weight
		}
		if e.key == 0 {
			return 0
		}
		i = (i + 1) & p.mask
	}
}

// buildPackedModel compiles a learner's string-keyed weights into packed keys
// for language. Called once per model load, never on the hot path. The result
// mirrors every feature of the model that the attribute writer can generate
// for language.
func buildPackedModel(language Language, a *AdaBoost) *packedModel {
	packed := make(map[uint64]float64, len(a.weights))
	keys := make([]uint64, 0, 4)
	for feature, weight := range a.weights {
		keys = parseFeatureKeys(language, feature, keys[:0])
		for _, key := range keys {
			packed[key] = weight
		}
	}

	capacity, shift := tableSize(len(packed))
	m := &packedModel{
		entries: make([]packedEntry, capacity),
		mask:    capacity - 1,
		shift:   shift,
	}
	for key, weight := range packed {
		k := key + 1
		i := (k * hashMultiplier) >> shift
		for m.entries[i].key != 0 {
			i = (i + 1) & m.mask
		}
		m.entries[i] = packedEntry{key: k, weight: weight}
	}
	return m
}

// packedPerceptron is an Averaged Perceptron compiled to the same packed
// feature keys as packedModel, so SegmentWithPos can score a position without
// rendering any feature strings. Each entry points at the feature's per-class
// weight vector inside one flat backing slice.
type packedPerceptron struct {
	entries []posEntry
	// weights holds every feature's class-weight vector end to end; entry e
	// covers weights[e.offset : e.offset+numClasses].
	weights    []float64
	numClasses int
	mask       uint64
	shift      uint
}

// posEntry is one slot of the open-addressed table. key is the packed feature
// key biased by one, so a zero key marks an empty slot.
type posEntry struct {
	key    uint64
	offset int32
}

// accumulate adds the per-class weights of a packed feature key into scores.
// An unknown key contributes nothing, exactly as a missing map entry did.
func (p *packedPerceptron) accumulate(key uint64, scores []float64) {
	if p.entries == nil {
		return
	}
	k := key + 1
	i := (k * hashMultiplier) >> p.shift
	for {
		e := &p.entries[i]
		if e.key == k {
			for j, w := range p.weights[e.offset : int(e.offset)+p.numClasses] {
				scores[j] += w
			}
			return
		}
		if e.key == 0 {
			return
		}
		i = (i + 1) & p.mask
	}
}

// buildPackedPerceptron compiles a POS model's string-keyed class weights into
// packed keys for language. Called once per model load, never on the hot path.
// Features the attribute writer cannot generate for language are dropped, the
// same way buildPackedModel drops them: they are unreachable, so omitting them
// cannot change any score.
func buildPackedPerceptron(language Language, p *AveragedPerceptron) *packedPerceptron {
	numClasses := len(p.classes)
	offsets := make(map[uint64]int32, len(p.weights))
	flat := make([]float64, 0, len(p.weights)*numClasses)
	keys := make([]uint64, 0, 4)
	for feature, slot := range p.weights {
		keys = parseFeatureKeys(language, feature, keys[:0])
		for _, key := range keys {
			if _, seen := offsets[key]; seen {
				// Distinct feature strings never pack to the same key with
				// the current grammar; keep the first if that ever changes.
				continue
			}
			offsets[key] = int32(len(flat))
			flat = append(flat, slot...)
		}
	}

	capacity, shift := tableSize(len(offsets))
	m := &packedPerceptron{
		entries:    make([]posEntry, capacity),
		weights:    flat,
		numClasses: numClasses,
		mask:       capacity - 1,
		shift:      shift,
	}
	for key, offset := range offsets {
		k := key + 1
		i := (k * hashMultiplier) >> shift
		for m.entries[i].key != 0 {
			i = (i + 1) & m.mask
		}
		m.entries[i] = posEntry{key: k, offset: offset}
	}
	return m
}

package cjk

// slotKind distinguishes the three slot types used by a feature template.
type slotKind int

const (
	slotTag slotKind = iota // boundary tag ("U"/"B"/"O")
	slotChr                 // a character (or sentinel)
	slotTyp                 // a character type code
)

// slot is one slot of a feature template. delta is the position offset:
// context index = i - 3 + delta for character position i.
type slot struct {
	kind  slotKind
	delta uint8
}

// template is one feature template: a distinct prefix plus an ordered slot
// list. A feature string renders as "{prefix}:{slot values concatenated}".
type template struct {
	id     uint8
	prefix string
	slots  []slot
}

func t(id uint8, prefix string, slots ...slot) template {
	return template{id: id, prefix: prefix, slots: slots}
}

func tagSlot(d uint8) slot { return slot{kind: slotTag, delta: d} }
func chrSlot(d uint8) slot { return slot{kind: slotChr, delta: d} }
func typSlot(d uint8) slot { return slot{kind: slotTyp, delta: d} }

// templates is the full feature template set in historical emission order.
// Slot deltas map the classic template variables to context positions:
// w1..w6 = Chr(0..5), c1..c6 = Typ(0..5), p1..p3 = Tag(0..2).
var templates = []template{
	t(0, "UP1", tagSlot(0)),
	t(1, "UP2", tagSlot(1)),
	t(2, "UP3", tagSlot(2)),
	t(3, "BP1", tagSlot(0), tagSlot(1)),
	t(4, "BP2", tagSlot(1), tagSlot(2)),
	t(5, "UW1", chrSlot(0)),
	t(6, "UW2", chrSlot(1)),
	t(7, "UW3", chrSlot(2)),
	t(8, "UW4", chrSlot(3)),
	t(9, "UW5", chrSlot(4)),
	t(10, "UW6", chrSlot(5)),
	t(11, "BW1", chrSlot(1), chrSlot(2)),
	t(12, "BW2", chrSlot(2), chrSlot(3)),
	t(13, "BW3", chrSlot(3), chrSlot(4)),
	t(14, "UC1", typSlot(0)),
	t(15, "UC2", typSlot(1)),
	t(16, "UC3", typSlot(2)),
	t(17, "UC4", typSlot(3)),
	t(18, "UC5", typSlot(4)),
	t(19, "UC6", typSlot(5)),
	t(20, "BC1", typSlot(1), typSlot(2)),
	t(21, "BC2", typSlot(2), typSlot(3)),
	t(22, "BC3", typSlot(3), typSlot(4)),
	t(23, "TC1", typSlot(0), typSlot(1), typSlot(2)),
	t(24, "TC2", typSlot(1), typSlot(2), typSlot(3)),
	t(25, "TC3", typSlot(2), typSlot(3), typSlot(4)),
	t(26, "TC4", typSlot(3), typSlot(4), typSlot(5)),
	t(27, "UQ1", tagSlot(0), typSlot(0)),
	t(28, "UQ2", tagSlot(1), typSlot(1)),
	t(29, "UQ3", tagSlot(2), typSlot(2)),
	t(30, "BQ1", tagSlot(1), typSlot(1), typSlot(2)),
	t(31, "BQ2", tagSlot(1), typSlot(2), typSlot(3)),
	t(32, "BQ3", tagSlot(2), typSlot(1), typSlot(2)),
	t(33, "BQ4", tagSlot(2), typSlot(2), typSlot(3)),
	t(34, "TQ1", tagSlot(1), typSlot(0), typSlot(1), typSlot(2)),
	t(35, "TQ2", tagSlot(1), typSlot(1), typSlot(2), typSlot(3)),
	t(36, "TQ3", tagSlot(2), typSlot(0), typSlot(1), typSlot(2)),
	t(37, "TQ4", tagSlot(2), typSlot(1), typSlot(2), typSlot(3)),
	// Language-specific char + char-type mixed features (Japanese/Chinese
	// only); kept last so templatesFor can slice.
	t(38, "WC1", chrSlot(2), typSlot(3)),
	t(39, "WC2", typSlot(2), chrSlot(3)),
	t(40, "WC3", chrSlot(2), typSlot(2)),
	t(41, "WC4", chrSlot(3), typSlot(3)),
}

// baseTemplateCount is the number of templates shared by all languages.
const baseTemplateCount = 38

// templatesFor returns the templates applicable to language.
// Japanese and Chinese use all 42; other languages use the 38 base templates.
func templatesFor(l Language) []template {
	if l == Japanese || l == Chinese {
		return templates
	}
	return templates[:baseTemplateCount]
}

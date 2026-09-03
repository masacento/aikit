package cjk

import "fmt"

// Language is the target language for segmentation.
type Language int

const (
	// Japanese
	Japanese Language = iota
	// Chinese
	Chinese
	// Korean
	Korean
)

// ParseLanguage parses a language name string into a Language.
// Accepted values: "japanese", "chinese", "korean".
func ParseLanguage(s string) (Language, error) {
	switch s {
	case "japanese":
		return Japanese, nil
	case "chinese":
		return Chinese, nil
	case "korean":
		return Korean, nil
	default:
		return 0, fmt.Errorf("unknown language: %q", s)
	}
}

func (l Language) String() string {
	switch l {
	case Japanese:
		return "Japanese"
	case Chinese:
		return "Chinese"
	case Korean:
		return "Korean"
	default:
		return "Unknown"
	}
}

// Type ids of the codes shared by all languages; they occupy fixed indices in
// every language's code table.
const (
	otherTypeID uint8 = 0 // "O"
	punctTypeID uint8 = 1 // "P"
	latinTypeID uint8 = 2 // "A"
	digitTypeID uint8 = 3 // "N"
)

// Per-language ordered tables of type codes. The index of a code in a table is
// its type id. Shared codes occupy fixed indices across all languages
// ("O"=0, "P"=1, "A"=2, "N"=3); language-specific codes follow from index 4.
// These are package-level so that lookups are a plain index, not an allocation.
var (
	japaneseTypeCodes = []string{"O", "P", "A", "N", "M", "H", "I", "K"}
	chineseTypeCodes  = []string{"O", "P", "A", "N", "F", "C", "X", "R", "B"}
	koreanTypeCodes   = []string{"O", "P", "A", "N", "E", "SN", "SF", "J", "G", "H"}
)

// typeCodes returns the ordered table of type codes this language can produce.
// The returned slice must not be modified.
func (l Language) typeCodes() []string {
	switch l {
	case Japanese:
		return japaneseTypeCodes
	case Chinese:
		return chineseTypeCodes
	case Korean:
		return koreanTypeCodes
	default:
		return nil
	}
}

// charTypeID classifies a rune into a language-specific type id (an index into
// typeCodes). Classification is a direct switch on character ranges, so it is
// allocation-free and O(1).
func (l Language) charTypeID(c rune) uint8 {
	switch l {
	case Japanese:
		return japaneseCharTypeID(c)
	case Chinese:
		return chineseCharTypeID(c)
	case Korean:
		return koreanCharTypeID(c)
	default:
		return otherTypeID
	}
}

// charType classifies a rune into a language-specific type code string.
// Implemented as a table lookup over charTypeID, so the string codes and the
// numeric ids are consistent by construction.
func (l Language) charType(c rune) string {
	switch l {
	case Japanese:
		return japaneseTypeCodes[japaneseCharTypeID(c)]
	case Chinese:
		return chineseTypeCodes[chineseCharTypeID(c)]
	case Korean:
		return koreanTypeCodes[koreanCharTypeID(c)]
	default:
		return "O"
	}
}

// punctLatinDigit returns the shared type id for punctuation, Latin, and digit
// characters, or false if the character belongs to none of these classes.
func punctLatinDigit(c rune) (uint8, bool) {
	switch {
	case c >= 0x3000 && c <= 0x303F,
		c >= 0xFF01 && c <= 0xFF0F,
		c >= 0xFF1A && c <= 0xFF20,
		c >= 0xFF3B && c <= 0xFF40,
		c >= 0xFF5B && c <= 0xFF65:
		return punctTypeID, true
	case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z',
		c >= 0xFF41 && c <= 0xFF5A, c >= 0xFF21 && c <= 0xFF3A:
		return latinTypeID, true
	case c >= '0' && c <= '9', c >= 0xFF10 && c <= 0xFF19:
		return digitTypeID, true
	default:
		return otherTypeID, false
	}
}

// japaneseCharTypeID classifies a rune for Japanese.
// "M"=4 Kanji numbers, "H"=5 Kanji, "I"=6 Hiragana, "K"=7 Katakana.
func japaneseCharTypeID(c rune) uint8 {
	switch c {
	case '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '億', '兆':
		return 4
	case '々', '〆', 'ヵ', 'ヶ':
		return 5
	}
	switch {
	case c >= 0x4E00 && c <= 0x9FFF: // CJK Unified Ideographs U+4E00..=U+9FFF
		return 5
	case c >= 0x3041 && c <= 0x3093: // Hiragana ぁ-ん
		return 6
	case (c >= 0x30A1 && c <= 0x30F4) || c == 'ー' || (c >= 0xFF71 && c <= 0xFF9D) || c == 'ﾞ' || c == 'ﾟ':
		return 7
	}
	if t, ok := punctLatinDigit(c); ok {
		return t
	}
	return otherTypeID
}

// chineseCharTypeID classifies a rune for Chinese.
// "F"=4 function words, "C"=5 CJK, "X"=6 CJK Ext A, "R"=7 radicals, "B"=8 Bopomofo.
func chineseCharTypeID(c rune) uint8 {
	switch c {
	case '的', '地', '得', '了', '着', '过', '吗', '呢', '吧', '啊', '嘛', '和',
		'与', '或', '但', '而', '且', '及', '在', '从', '到', '把', '被', '对',
		'向', '给', '是', '有', '不', '也', '都', '就', '要', '会', '能', '可':
		return 4
	}
	switch {
	case c >= 0x4E00 && c <= 0x9FFF:
		return 5
	case c >= 0x3400 && c <= 0x4DBF:
		return 6
	case c >= 0x2E80 && c <= 0x2FDF:
		return 7
	case (c >= 0x3100 && c <= 0x312F) || (c >= 0x31A0 && c <= 0x31BF):
		return 8
	}
	if t, ok := punctLatinDigit(c); ok {
		return t
	}
	return otherTypeID
}

// koreanCharTypeID classifies a rune for Korean.
// "E"=4 particles, "SN"=5 no batchim, "SF"=6 with batchim, "J"=7 Jamo,
// "G"=8 compatibility Jamo, "H"=9 Hanja.
func koreanCharTypeID(c rune) uint8 {
	switch c {
	case '은', '는', '을', '를', '의', '에':
		return 4
	}
	switch {
	case c >= 0xAC00 && c <= 0xD7AF:
		if (c-0xAC00)%28 == 0 {
			return 5
		}
		return 6
	case c >= 0x1100 && c <= 0x11FF:
		return 7
	case c >= 0x3130 && c <= 0x318F:
		return 8
	case c >= 0x4E00 && c <= 0x9FFF:
		return 9
	}
	if t, ok := punctLatinDigit(c); ok {
		return t
	}
	return otherTypeID
}

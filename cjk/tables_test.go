package cjk

import "testing"

// charType is the string-view counterpart of charTypeID (used by litsea's
// feature templates); pinning the two against each other keeps the tables
// honest — a reordering of a language's type codes shows up here as a mismatch
// between the numeric id and its code.
func TestLanguage_charType(t *testing.T) {
	for _, tc := range []struct {
		lang Language
		r    rune
		want string
	}{
		{Japanese, 'あ', "I"}, // hiragana
		{Japanese, '漢', "H"}, // kanji
		{Japanese, 'ア', "K"}, // katakana
		{Japanese, '한', "O"}, // hangul is other, for Japanese
		{Chinese, '漢', "C"},
		{Korean, '한', "SF"}, // syllable block
		{Japanese, 'a', "A"},
		{Japanese, '5', "N"},
		{Japanese, '。', "P"},
		{Japanese, 'x', "A"},
	} {
		if got := tc.lang.charType(tc.r); got != tc.want {
			t.Errorf("%d.charType(%q) = %q, want %q", tc.lang, tc.r, got, tc.want)
		}
	}
}

func TestAdaBoost_weight(t *testing.T) {
	a := &AdaBoost{weights: map[string]float64{"w3:1": 0.5}}
	if got := a.weight("w3:1"); got != 0.5 {
		t.Errorf("weight = %v, want 0.5", got)
	}
	if got := a.weight("missing"); got != 0 {
		t.Errorf("weight of unknown attr = %v, want 0", got)
	}
	var nilB *AdaBoost
	if got := nilB.weight("w3:1"); got != 0 {
		t.Errorf("nil model weight = %v, want 0", got)
	}
}

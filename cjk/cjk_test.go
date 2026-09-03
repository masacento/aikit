package cjk

import (
	"math"
	"testing"
)

func TestLoadAdaBoostBiasBeforeTrailingWeights(t *testing.T) {
	m, err := LoadAdaBoostFromBytes([]byte("UW1:a\t1.5\n0.25\nUW2:b\t-0.75\n"))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.Bias()-0.25) > 1e-12 {
		t.Fatalf("Bias() = %g, want the serialized bias 0.25", m.Bias())
	}
	if got := m.weight("UW2:b"); got != -0.75 {
		t.Fatalf("trailing weight = %g, want -0.75", got)
	}
}

func TestSegmentSegmentWordsJapanese(t *testing.T) {
	m, err := LoadAdaBoost("models/japanese.model")
	if err != nil {
		t.Fatal(err)
	}
	seg := NewSegmenter(Japanese, m)
	text := "アップルは2007年にiPhoneを発表した。"
	ws := seg.SplitWords(text)
	if len(ws) < 5 {
		t.Fatalf("too few words: %d", len(ws))
	}
	pos := 0
	for i, w := range ws {
		if text[w.Start:w.End] != w.Text {
			t.Fatalf("word %d: text %q != text[%d:%d]", i, w.Text, w.Start, w.End)
		}
		if w.Start != pos {
			t.Fatalf("word %d: start %d, want %d (words must partition the input)", i, w.Start, pos)
		}
		pos = w.End
	}
	if pos != len(text) {
		t.Fatalf("last end %d, want %d", pos, len(text))
	}
	if ws[0].Text != "アップル" {
		t.Fatalf("first word = %q, want アップル", ws[0].Text)
	}
}

func TestDefaultJapanese(t *testing.T) {
	seg, err := Default(Japanese)
	if err != nil {
		t.Fatal(err)
	}
	if seg == nil {
		t.Fatal("Default(Japanese) = nil")
	}
	// Shared instance across calls.
	seg2, err := Default(Japanese)
	if err != nil || seg2 != seg {
		t.Fatal("Default(Japanese) must return the same segmenter")
	}
	ws := seg.SplitWords("権藤三峰は武将である")
	if len(ws) < 3 {
		t.Fatalf("too few words: %d", len(ws))
	}
	if ws[0].Text != "権藤" {
		t.Fatalf("first word = %q, want 権藤", ws[0].Text)
	}
}

func TestDefaultChineseKorean(t *testing.T) {
	zh, err := Default(Chinese)
	if err != nil {
		t.Fatal(err)
	}
	if ws := zh.Segment("我是中国人"); len(ws) < 2 {
		t.Fatalf("chinese segmentation too coarse: %q", ws)
	}
	ko, err := Default(Korean)
	if err != nil {
		t.Fatal(err)
	}
	if ws := ko.Segment("한국어 텍스트입니다"); len(ws) < 2 {
		t.Fatalf("korean segmentation too coarse: %q", ws)
	}
}

func TestDetectCJK(t *testing.T) {
	cases := []struct {
		text string
		want Language
		ok   bool
	}{
		{"Steve Jobs founded Apple", 0, false},
		{"権藤三峰は武将である", Japanese, true}, // kana -> Japanese
		{"漢字だけのテキスト", Japanese, true},  // Han + kana
		{"한국어 텍스트입니다", Korean, true},   // Hangul -> Korean
		{"数字123", Japanese, true},      // kana (字 counts as Han, and 数 is Han too)
		{"abc123", 0, false},           // ASCII digits/letters are not CJK
	}
	for _, c := range cases {
		got, ok := DetectCJK(c.text)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("DetectCJK(%q) = (%v, %v), want (%v, %v)", c.text, got, ok, c.want, c.ok)
		}
	}
}

func TestHasCJK(t *testing.T) {
	if HasCJK("plain english 123") {
		t.Error("HasCJK(english) = true")
	}
	if !HasCJK("日本語") || !HasCJK("한국어") {
		t.Error("HasCJK must detect kana, Han, and Hangul")
	}
}

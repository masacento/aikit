package ner

import "testing"

func TestSplitWordsCJK(t *testing.T) {
	ja := "アップルは2007年にiPhoneを発表した。"
	ws := SplitWordsCJK(ja)
	if len(ws) < 5 {
		t.Fatalf("Japanese text not segmented: %d words", len(ws))
	}
	for i, w := range ws {
		if ja[w.Start:w.End] != w.Text {
			t.Fatalf("word %d offsets: %q vs %q", i, ja[w.Start:w.End], w.Text)
		}
	}
	if w0 := ws[0].Text; w0 != "アップル" {
		t.Fatalf("first word = %q, want アップル", w0)
	}

	en := "Steve Jobs founded Apple in Hawaii."
	ws = SplitWordsCJK(en)
	want := SplitWords(en)
	if len(ws) != len(want) {
		t.Fatalf("latin text: %d words, want %d", len(ws), len(want))
	}
	for i := range ws {
		if ws[i] != want[i] {
			t.Fatalf("latin text word %d = %+v, want %+v", i, ws[i], want[i])
		}
	}
}

func TestSplitWordsCKorean(t *testing.T) {
	ko := "한국어 텍스트를 분석합니다"
	ws := SplitWordsCJK(ko)
	if len(ws) == 0 {
		t.Fatal("no words")
	}
	for i, w := range ws {
		if ko[w.Start:w.End] != w.Text {
			t.Fatalf("word %d offsets: %q vs %q", i, ko[w.Start:w.End], w.Text)
		}
	}
	t.Logf("%+v", ws)
}

func TestSplitWords2CJKOffsets(t *testing.T) {
	text := "権藤三峰は武将である"
	prevEnd := 0
	for i, w := range SplitWords2CJK(text) {
		if text[w.Start:w.End] != w.Text {
			t.Fatalf("word %d: text %q != text[%d:%d]=%q", i, w.Text, w.Start, w.End, text[w.Start:w.End])
		}
		if w.Start != prevEnd {
			t.Fatalf("word %d: start %d, want %d (cumulative word?)", i, w.Start, prevEnd)
		}
		prevEnd = w.End
	}
}

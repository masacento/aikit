package ner

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

const (
	tokenclassModelDir   = "../testdata/distilbert-secret-masker-v3.3a-rs"
	tokenclassGoldenPath = "../testdata/tokenclassification_golden.json"
)

type goldenSpan struct {
	Start int     `json:"start"`
	End   int     `json:"end"`
	Line  int     `json:"line"`
	Value string  `json:"value"`
	Score float64 `json:"score"`
}

type tokenclassGolden struct {
	Model string  `json:"model"`
	Tau   float64 `json:"tau"`
	Mask  string  `json:"mask"`
	Cases []struct {
		Text     string       `json:"text"`
		Spans    []goldenSpan `json:"spans"`
		SpansTau []goldenSpan `json:"spans_tau"`
		Masked   string       `json:"masked"`
		Invalid  int          `json:"invalid_bio_transitions"`
		Pieces   []struct {
			Start int     `json:"start"`
			End   int     `json:"end"`
			Label int     `json:"label"`
			P     float64 `json:"p"`
		} `json:"pieces"`
	} `json:"cases"`
}

func loadTokenClassifierFixture(t *testing.T) (*TokenClassifier, *tokenclassGolden) {
	t.Helper()
	if _, err := os.Stat(tokenclassModelDir + "/model.safetensors"); err != nil {
		t.Skipf("no distilbert-secret-masker at %s — uvx --from huggingface_hub hf download "+
			"AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs", tokenclassModelDir)
	}
	raw, err := os.ReadFile(tokenclassGoldenPath)
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_tokenclassification.py")
	}
	var g tokenclassGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := LoadTokenClassifier(tokenclassModelDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, &g
}

// TestTokenClassifier_parity certifies the whole token-classification path —
// tokenizer with offsets, manual overflow windows with first-window-wins
// dedup, DistilBERT trunk, classifier head, lenient BIO decode, threshold
// mode — against the model card's span_infer.py
// (scripts/oracle/pin_tokenclassification.py). The masked-text half of the golden is
// asserted by the secret-specific consumer, examples/secretmasker.
//
// Like TestGLiNER_parity it asserts two levels. The per-piece tensor (stored
// for the first few cases) catches a forward that is wrong but still ranks
// entity pieces first; the span sets — argmax AND tau-thresholded — are the
// artifact that ships, and the tau set's membership is exactly where fp drift
// could flip a span across the threshold (the fixture holds one at 0.986 vs
// tau 0.99 on purpose).
func TestTokenClassifier_parity(t *testing.T) {
	m, g := loadTokenClassifierFixture(t)
	if labels := m.Labels(); !equalStrings(labels, []string{"O", "B-SECRET", "I-SECRET"}) {
		t.Fatalf("Labels() = %v, want the pinned checkpoint's [O B-SECRET I-SECRET]", labels)
	}

	fail := func(ci int, format string, args ...any) {
		t.Errorf("case %d (%q): %s", ci, truncate(g.Cases[ci].Text), fmt.Sprintf(format, args...))
	}

	const pieceTol, spanTol = 5e-3, 5e-3
	for ci, c := range g.Cases {
		pieces, invalid, err := m.Pieces(c.Text, TokenOpts{})
		if err != nil {
			t.Fatalf("case %d: Pieces: %v", ci, err)
		}
		if invalid != c.Invalid {
			fail(ci, "invalid BIO transitions %d, want %d", invalid, c.Invalid)
		}

		if c.Pieces != nil {
			if len(pieces) != len(c.Pieces) {
				fail(ci, "%d pieces, want %d", len(pieces), len(c.Pieces))
			} else {
				for k, want := range c.Pieces {
					got := pieces[k]
					ws, we := charToByte(c.Text, want.Start), charToByte(c.Text, want.End)
					if got.Start != ws || got.End != we {
						fail(ci, "piece %d at bytes [%d,%d), want [%d,%d)",
							k, got.Start, got.End, ws, we)
						continue
					}
					if got.Label != want.Label {
						fail(ci, "piece %d label %d, want %d", k, got.Label, want.Label)
					}
					if d := math.Abs(got.PEntity - want.P); d > pieceTol {
						fail(ci, "piece %d p_entity %.6f, want %.6f (diff %.3g)",
							k, got.PEntity, want.P, d)
					}
				}
			}
		}

		got, err := m.Predict(c.Text, TokenOpts{})
		if err != nil {
			t.Fatalf("case %d: Predict: %v", ci, err)
		}
		compareSpans(fail, ci, c.Text, got, c.Spans)

		gotTau, err := m.Predict(c.Text, TokenOpts{Tau: g.Tau})
		if err != nil {
			t.Fatalf("case %d: Predict(tau): %v", ci, err)
		}
		compareSpans(fail, ci, c.Text, gotTau, c.SpansTau)
	}
}

func compareSpans(fail func(int, string, ...any), ci int, text string, got []TokenEntity, want []goldenSpan) {
	const spanTol = 5e-3
	if len(got) != len(want) {
		fail(ci, "%d spans, want %d: %+v vs %+v", len(got), len(want), got, want)
		return
	}
	for i, w := range want {
		e := got[i]
		// The reference reports CHARACTER offsets (Python string indices);
		// this package reports byte offsets, converted here like GLiNER's.
		ws, we := charToByte(text, w.Start), charToByte(text, w.End)
		if e.Text != w.Value || e.Start != ws || e.End != we {
			fail(ci, "span %d = {[%d,%d) %q}, want {[%d,%d) %q}",
				i, e.Start, e.End, e.Text, ws, we, w.Value)
		}
		// Line is not part of the Go API (a caller concern) — but the golden
		// pins the reference's 1-based count of newlines up to the span, and
		// examples/secretmasker reuses exactly that definition, so verify it
		// holds for the byte offsets this package produced.
		if line := strings.Count(text[:e.Start], "\n") + 1; line != w.Line {
			fail(ci, "span %d line %d, want %d", i, line, w.Line)
		}
		if e.Label != "SECRET" {
			fail(ci, "span %d label %q, want \"SECRET\" (the B-/I- prefix must be stripped)", i, e.Label)
		}
		if d := math.Abs(e.Score - w.Score); d > spanTol {
			fail(ci, "span %d score %.6f, want %.6f (diff %.3g)", i, e.Score, w.Score, d)
		}
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func truncate(s string) string {
	if len(s) > 40 {
		return s[:40] + "…"
	}
	return s
}

// TestTokenClassifier_decode pins decode decisions the parity golden happens
// not to exercise: whole-entity threshold drops (never a truncated prefix),
// the lenient I-open and its counter, and label-name extraction.
func TestTokenClassifier_decode(t *testing.T) {
	m := &TokenClassifier{
		labels: []string{"O", "B-SECRET", "I-SECRET"},
		role:   []byte{'O', 'B', 'I'},
	}
	// Five pieces: an O, a two-piece entity, a closing O, then an orphan I
	// (the lenient-BIO invalid transition).
	pieces := []Piece{
		{Start: 0, End: 1, Label: 0, PEntity: 0.1},
		{Start: 2, End: 5, Label: 1, PEntity: 0.90},
		{Start: 5, End: 8, Label: 2, PEntity: 0.80},
		{Start: 9, End: 10, Label: 0, PEntity: 0.1},
		{Start: 11, End: 12, Label: 2, PEntity: 0.70},
	}
	const text = "abcdefghijkl"

	spans := m.decode(text, pieces, 0)
	if len(spans) != 2 {
		t.Fatalf("argmax mode: %d spans, want 2: %+v", len(spans), spans)
	}
	if spans[0].Start != 2 || spans[0].End != 8 || spans[0].Text != text[2:8] {
		t.Errorf("entity span = [%d,%d) %q, want [2,8) %q",
			spans[0].Start, spans[0].End, spans[0].Text, text[2:8])
	}
	if spans[0].Label != "SECRET" {
		t.Errorf("entity label = %q, want \"SECRET\"", spans[0].Label)
	}
	if spans[1].Start != 11 || spans[1].End != 12 {
		t.Errorf("orphan span = [%d,%d), want [11,12)", spans[1].Start, spans[1].End)
	}
	if want := (0.90 + 0.80) / 2; math.Abs(spans[0].Score-want) > 1e-12 {
		t.Errorf("entity score %.6f, want %.6f (mean over pieces)", spans[0].Score, want)
	}
	if n := countInvalid(pieces, m.labels, m.role); n != 1 {
		t.Errorf("countInvalid = %d, want 1 (the orphan I)", n)
	}

	// Threshold 0.82 keeps the entity (mean 0.85) and drops the orphan (0.70)
	// — whole-entity filtering, never a truncated prefix.
	if spans := m.decode(text, pieces, 0.82); len(spans) != 1 || spans[0].Start != 2 {
		t.Errorf("tau 0.82: %+v, want only the entity", spans)
	}
	// Threshold 0.65 keeps both.
	if spans := m.decode(text, pieces, 0.65); len(spans) != 2 {
		t.Errorf("tau 0.65: %d spans, want 2", len(spans))
	}
}

func TestTokenClassifierDecodeSplitsMismatchedIType(t *testing.T) {
	m := &TokenClassifier{
		labels: []string{"O", "B-PER", "I-PER", "B-ORG", "I-ORG"},
		role:   []byte{'O', 'B', 'I', 'B', 'I'},
	}
	pieces := []Piece{
		{Start: 0, End: 4, Label: 1, PEntity: 0.9},
		{Start: 5, End: 9, Label: 4, PEntity: 0.8},
	}
	const text = "John Acme"
	got := m.decode(text, pieces, 0)
	if len(got) != 2 || got[0].Label != "PER" || got[1].Label != "ORG" || got[1].Start != 5 {
		t.Fatalf("decode = %+v, want separate PER and ORG entities", got)
	}
	if invalid := countInvalid(pieces, m.labels, m.role); invalid != 1 {
		t.Fatalf("countInvalid = %d, want mismatched I-ORG transition counted", invalid)
	}
}

// TestTokenClassifier_windowOpts pins the window arithmetic's guard rails: the
// MaxLength ceiling is the trunk's position table, because bert.go forward
// truncates ids to that table and a longer window would silently break the
// piece↔offset alignment instead of failing.
func TestTokenClassifier_windowOpts(t *testing.T) {
	const trunkMax = 512
	ml, st, err := (TokenOpts{}).window(trunkMax)
	if err != nil {
		t.Fatal(err)
	}
	if ml != trunkMax || st != 128 {
		t.Errorf("zero opts = (%d, %d), want (512, 128)", ml, st)
	}
	if _, _, err := (TokenOpts{MaxLength: trunkMax + 1}).window(trunkMax); err == nil {
		t.Error("MaxLength past the position table must error, not silently truncate")
	}
	for _, bad := range []TokenOpts{{MaxLength: 2}, {MaxLength: -1}, {Stride: 0, MaxLength: -1}} {
		if _, _, err := bad.window(trunkMax); err == nil {
			t.Errorf("%+v must error", bad)
		}
	}
	if _, _, err := (TokenOpts{MaxLength: 3}).window(trunkMax); err != nil {
		t.Errorf("MaxLength 3 (CLS + one token + SEP) must be accepted: %v", err)
	}
	// A stride ≥ body collapses to step 1 — the reference does the same via
	// max(1, body - stride) — valid, just quadratic.
	if _, st, err := (TokenOpts{Stride: 1 << 20}).window(trunkMax); err != nil || st != 1<<20 {
		t.Errorf("oversized stride: (%d, %v), want accepted", st, err)
	}
}

package gate

import (
	"strings"
	"testing"
)

func TestField_String(t *testing.T) {
	cases := []struct {
		name string
		f    Field
		want string
	}{
		{"no detail", Field{Key: "status", State: "ok"}, "status=ok"},
		{"with detail", Field{Key: "status", State: "FAIL", Detail: "boom"}, "status=FAIL(boom)"},
		{"empty state", Field{Key: "k"}, "k="},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.f.String(); got != c.want {
				t.Errorf("String() = %q, want %q", got, c.want)
			}
		})
	}
}

func TestCell_Field(t *testing.T) {
	c := Cell{Fields: []Field{
		{Key: "status", State: "ok"},
		{Key: "count", State: "3"},
	}}
	if got := c.Field("status"); got != "ok" {
		t.Errorf("Field(status) = %q, want ok", got)
	}
	if got := c.Field("count"); got != "3" {
		t.Errorf("Field(count) = %q, want 3", got)
	}
	if got := c.Field("missing"); got != "" {
		t.Errorf("Field(missing) = %q, want empty", got)
	}
	if got := (Cell{}).Field("anything"); got != "" {
		t.Errorf("Field on a zero-value Cell = %q, want empty", got)
	}
}

func TestRunAll_preservesOrder(t *testing.T) {
	checks := []Check{
		{Name: "a", Run: func() Cell { return Cell{Name: "a", Outcome: OK} }},
		{Name: "b", Run: func() Cell { return Cell{Name: "b", Outcome: Fail} }},
		{Name: "c", Run: func() Cell { return Cell{Name: "c", Outcome: NA} }},
	}
	cells := RunAll(checks)
	if len(cells) != 3 {
		t.Fatalf("len(cells) = %d, want 3", len(cells))
	}
	wantNames := []string{"a", "b", "c"}
	for i, want := range wantNames {
		if cells[i].Name != want {
			t.Errorf("cells[%d].Name = %q, want %q", i, cells[i].Name, want)
		}
	}
}

// TestRunAll_panicBecomesFail confirms the count is never lost to a panicking check — the
// doc comment's central promise. A gate with N checks must always report N cells.
func TestRunAll_panicBecomesFail(t *testing.T) {
	checks := []Check{
		{Name: "ok-one", Run: func() Cell { return Cell{Name: "ok-one", Outcome: OK} }},
		{Name: "boom", Run: func() Cell { panic("kaboom") }},
		{Name: "ok-two", Run: func() Cell { return Cell{Name: "ok-two", Outcome: OK} }},
	}
	cells := RunAll(checks)
	if len(cells) != 3 {
		t.Fatalf("len(cells) = %d, want 3 (the panic must not drop the count)", len(cells))
	}
	if cells[1].Outcome != Fail {
		t.Errorf("panicking check's Outcome = %v, want Fail", cells[1].Outcome)
	}
	if cells[1].Name != "boom" {
		t.Errorf("panicking check's Name = %q, want %q (must come from the Check, not the panic)", cells[1].Name, "boom")
	}
	var panicField Field
	for _, f := range cells[1].Fields {
		if f.Key == "panic" {
			panicField = f
		}
	}
	if !strings.Contains(panicField.Detail, "kaboom") {
		t.Errorf("panic field detail = %q, want it to contain the recovered value", panicField.Detail)
	}
	// The checks either side of the panic must be unaffected — one panicking Run must not
	// poison its neighbors.
	if cells[0].Outcome != OK || cells[2].Outcome != OK {
		t.Errorf("neighboring checks were affected by the panic: %v, %v", cells[0].Outcome, cells[2].Outcome)
	}
}

func cellsOf(outcomes ...Outcome) []Cell {
	cells := make([]Cell, len(outcomes))
	for i, o := range outcomes {
		cells[i] = Cell{Outcome: o}
	}
	return cells
}

// TestReconcile_precedence is the test the package's own doc comment asks for: "a
// regression in Reconcile's precedence logic... would silently propagate everywhere". Every
// row is a case from Precedence's own doc comment, made explicit rather than left implicit
// in a caller's behavior.
func TestReconcile_precedence(t *testing.T) {
	cases := []struct {
		name        string
		outcomes    []Outcome
		precedence  Precedence
		wantOutcome Outcome
		wantExit    int
	}{
		{"empty is OK", nil, InconclusiveWins, OK, 0},
		{"all ok", []Outcome{OK, OK, OK}, InconclusiveWins, OK, 0},
		{"all na is OK (na excluded, nothing left to fail on)", []Outcome{NA, NA}, InconclusiveWins, OK, 0},
		{"one fail, inconclusive-wins precedence, no incon present", []Outcome{OK, Fail}, InconclusiveWins, Fail, 1},
		{"one incon, inconclusive-wins precedence", []Outcome{OK, Inconclusive}, InconclusiveWins, Inconclusive, 2},
		{
			"fail AND incon together, InconclusiveWins: incon dominates",
			[]Outcome{OK, Fail, Inconclusive}, InconclusiveWins, Inconclusive, 2,
		},
		{
			"fail AND incon together, FailWins: fail dominates",
			[]Outcome{OK, Fail, Inconclusive}, FailWins, Fail, 1,
		},
		{"one fail, FailWins precedence, no incon present", []Outcome{OK, Fail}, FailWins, Fail, 1},
		{"one incon, FailWins precedence, no fail present", []Outcome{OK, Inconclusive}, FailWins, Inconclusive, 2},
		{"na cells never flip an all-ok run", []Outcome{OK, NA, OK, NA}, InconclusiveWins, OK, 0},
		{"na alongside fail: na still excluded, fail still wins", []Outcome{NA, Fail}, InconclusiveWins, Fail, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rep := ReconcileWith(cellsOf(c.outcomes...), c.precedence)
			if rep.Outcome != c.wantOutcome {
				t.Errorf("Outcome = %v, want %v", rep.Outcome, c.wantOutcome)
			}
			if rep.Exit != c.wantExit {
				t.Errorf("Exit = %d, want %d", rep.Exit, c.wantExit)
			}
		})
	}
}

// TestReconcile_isReconcileWithInconclusiveWins pins Reconcile's documented default so a
// future edit can't silently change which precedence it wraps.
func TestReconcile_isReconcileWithInconclusiveWins(t *testing.T) {
	cells := cellsOf(OK, Fail, Inconclusive)
	got := Reconcile(cells)
	want := ReconcileWith(cells, InconclusiveWins)
	if got != want {
		t.Errorf("Reconcile(cells) = %+v, want ReconcileWith(cells, InconclusiveWins) = %+v", got, want)
	}
}

func TestReconcile_tally(t *testing.T) {
	cells := cellsOf(OK, OK, Fail, NA, Inconclusive, NA, OK)
	rep := Reconcile(cells)
	if rep.Total != 7 {
		t.Errorf("Total = %d, want 7", rep.Total)
	}
	if rep.Pass != 3 {
		t.Errorf("Pass = %d, want 3", rep.Pass)
	}
	if rep.Fail != 1 {
		t.Errorf("Fail = %d, want 1", rep.Fail)
	}
	if rep.Incon != 1 {
		t.Errorf("Incon = %d, want 1", rep.Incon)
	}
	if rep.NA != 2 {
		t.Errorf("NA = %d, want 2", rep.NA)
	}
	if got, want := rep.Applicable(), 5; got != want { // Total - NA = 7 - 2
		t.Errorf("Applicable() = %d, want %d", got, want)
	}
}

func TestVerdict(t *testing.T) {
	cases := []struct {
		o    Outcome
		msg  string
		want string
	}{
		{OK, "5/5 passed", "VERDICT: PASS — 5/5 passed"},
		{Fail, "2 defects", "VERDICT: FAIL — 2 defects"},
		{Inconclusive, "proxy unreachable", "VERDICT: INCONCLUSIVE — proxy unreachable"},
	}
	for _, c := range cases {
		if got := Verdict(c.o, c.msg); got != c.want {
			t.Errorf("Verdict(%v, %q) = %q, want %q", c.o, c.msg, got, c.want)
		}
	}
}

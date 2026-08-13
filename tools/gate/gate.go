// Package gate is a small runner for verification gates. It executes a data-defined
// matrix of checks, never aborting on one failure, and reconciles their outcomes into a
// single exit code under one precedence rule. The point is that the tally, the n/a
// arithmetic, and the exit-code contract live in ONE place, so a gate is defined by the
// []Check it supplies rather than by re-implementing "how do I count and what do I exit".
//
// THE SEAM. consumergate is the first user. gpu_gate.sh and gpu_device.sh are the intended
// next ones: their groups are already the same shape — a named unit that is ok, n/a, or
// FAIL, tallied so that a skip is never a pass and no single machine has to cover a matrix
// alone. Moving them onto this runner means writing their groups as []Check and deleting
// their bespoke reconciliation; it does not mean changing what they decide. This package
// migrates nothing on its own — it is the shape the migration would target.
package gate

import "fmt"

// Outcome is the classification of a Cell (and, after Reconcile, of a whole run).
//
//   - OK   — the unit passed.
//   - NA   — not applicable here (e.g. wrong OS for a platform-gated module). EXCLUDED
//     from the applicable tally: neither a pass nor a failure, so that "no single machine
//     covers the matrix" is honest arithmetic rather than a green tick over half the work.
//   - FAIL — a real defect.
//   - INCONCLUSIVE — the gate could not judge (tooling or network unavailable). Never a
//     pass; it dominates the run so a green cannot be reported over an unjudged unit.
type Outcome string

const (
	OK           Outcome = "ok"
	NA           Outcome = "n/a"
	Fail         Outcome = "FAIL"
	Inconclusive Outcome = "INCONCLUSIVE"
)

// Field is one labeled sub-result printed on a Cell's scope line, e.g. `resolve=ok` or
// `compile=n/a(no packages on darwin)`. A Cell may carry several — one per tier.
type Field struct {
	Key    string
	State  string
	Detail string
}

func (f Field) String() string {
	if f.Detail == "" {
		return f.Key + "=" + f.State
	}
	return f.Key + "=" + f.State + "(" + f.Detail + ")"
}

// Cell is the result of one unit of work in the matrix.
type Cell struct {
	Name    string
	Fields  []Field
	Outcome Outcome
}

// Check is one unit of work — data, not control flow. A gate is a []Check.
type Check struct {
	Name string
	Run  func() Cell
}

// RunAll executes every check IN ORDER and collects the cells. A check that panics becomes
// a Fail cell instead of aborting the run: the count is never lost. Sequential by design —
// these checks shell out to the network and determinism is worth more than the wall-clock.
func RunAll(checks []Check) []Cell {
	cells := make([]Cell, 0, len(checks))
	for _, c := range checks {
		cells = append(cells, runOne(c))
	}
	return cells
}

func runOne(c Check) (cell Cell) {
	defer func() {
		if r := recover(); r != nil {
			cell = Cell{
				Name:    c.Name,
				Outcome: Fail,
				Fields:  []Field{{Key: "panic", State: "FAIL", Detail: fmt.Sprint(r)}},
			}
		}
	}()
	return c.Run()
}

// Report is the reconciled tally of a run.
type Report struct {
	Total, Pass, Fail, Incon, NA int
	Outcome                      Outcome
	Exit                         int
}

// Applicable is Total minus the NA cells — the denominator a verdict should report, so a
// run's fraction describes what this machine could actually judge.
func (r Report) Applicable() int { return r.Total - r.NA }

// Reconcile applies the one precedence rule that keeps a skip from reading as a pass:
// any INCONCLUSIVE makes the whole run INCONCLUSIVE (exit 2); else any FAIL is FAIL
// (exit 1); else OK (exit 0). NA cells are excluded from the tally entirely.
func Reconcile(cells []Cell) Report {
	r := Report{Total: len(cells)}
	for _, c := range cells {
		switch c.Outcome {
		case Inconclusive:
			r.Incon++
		case Fail:
			r.Fail++
		case NA:
			r.NA++
		default:
			r.Pass++
		}
	}
	switch {
	case r.Incon > 0:
		r.Outcome, r.Exit = Inconclusive, 2
	case r.Fail > 0:
		r.Outcome, r.Exit = Fail, 1
	default:
		r.Outcome, r.Exit = OK, 0
	}
	return r
}

// Verdict formats the paste-able one-liner every gate ends with. The word after the dash
// is PASS for an OK run so it reads as an assertion rather than a bare "ok".
func Verdict(o Outcome, msg string) string {
	word := string(o)
	if o == OK {
		word = "PASS"
	}
	return "VERDICT: " + word + " — " + msg
}

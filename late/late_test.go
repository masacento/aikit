package late

import (
	"math"
	"testing"
)

// unit is a 2-D unit vector at angle theta (radians) — lets test cases specify
// exact cosine similarities instead of hand-normalizing arbitrary vectors.
func unit(theta float64) []float32 {
	return []float32{float32(math.Cos(theta)), float32(math.Sin(theta))}
}

func TestMaxSim_basic(t *testing.T) {
	// Two query tokens, two doc tokens. q0 is closest to d1 (same angle);
	// q1 is closest to d0 (60° apart, cos(60°)=0.5) vs d1 (90° apart, cos=0).
	q := [][]float32{unit(0), unit(math.Pi / 3)} // 0°, 60°
	d := [][]float32{unit(0), unit(math.Pi / 2)} // 0°, 90°

	got := MaxSim(q, d)
	// q0·d0 = cos(0)=1 (best for q0); q1·d0 = cos(60°)=0.5, q1·d1 = cos(-30°)=cos(30°)≈0.866 (best for q1)
	want := 1.0 + math.Cos(math.Pi/6)
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("MaxSim = %v, want %v", got, want)
	}
}

func TestMaxSim_negativeSimilarity(t *testing.T) {
	// Every doc token is a POOR match (negative cosine) for the query token —
	// the regression case for seeding best at 0 instead of the first candidate:
	// a naive `best := float32(0); if s > best` would wrongly report 0 (as if
	// no comparison had happened) instead of the true negative best.
	q := [][]float32{{1, 0}}
	d := [][]float32{{-1, 0}, {-0.7071068, 0.7071068}} // cos = -1, cos = -0.707
	got := MaxSim(q, d)
	want := -0.7071068 // the LESS negative of the two, i.e. the true max
	if math.Abs(got-want) > 1e-5 {
		t.Fatalf("MaxSim = %v, want %v (must not silently floor at 0)", got, want)
	}
}

func TestMaxSim_degenerate(t *testing.T) {
	q := [][]float32{unit(0), unit(1)}
	d := [][]float32{unit(0)}
	if got := MaxSim(nil, d); got != 0 {
		t.Errorf("MaxSim(nil query) = %v, want 0", got)
	}
	if got := MaxSim(q, nil); got != 0 {
		t.Errorf("MaxSim(nil doc) = %v, want 0 (no candidate to match)", got)
	}
	if got := MaxSim(nil, nil); got != 0 {
		t.Errorf("MaxSim(nil, nil) = %v, want 0", got)
	}
}

func TestScoreBatch_matchesMaxSim(t *testing.T) {
	q := [][]float32{unit(0), unit(0.4), unit(1.1)}
	docs := make([][][]float32, 40) // more than NumCPU, so every worker handles several
	for i := range docs {
		docs[i] = [][]float32{unit(float64(i) * 0.1), unit(float64(i)*0.1 + 0.3)}
	}
	got := ScoreBatch(q, docs, 0)
	for i, d := range docs {
		want := MaxSim(q, d)
		if got[i] != want {
			t.Fatalf("ScoreBatch[%d] = %v, want %v (serial MaxSim)", i, got[i], want)
		}
	}

	// concurrency=1 must agree too — the split, not the math, changes.
	serial := ScoreBatch(q, docs, 1)
	for i := range docs {
		if serial[i] != got[i] {
			t.Fatalf("concurrency=1 disagrees with concurrency=0 at [%d]: %v vs %v", i, serial[i], got[i])
		}
	}
}

func TestScoreBatch_empty(t *testing.T) {
	if got := ScoreBatch([][]float32{unit(0)}, nil, 0); got != nil {
		t.Errorf("ScoreBatch(no docs) = %v, want nil", got)
	}
}

func TestIndex_Query(t *testing.T) {
	// Three single-token docs at increasing angular distance from the query —
	// Query must return them nearest-first.
	q := [][]float32{unit(0)}
	docs := [][][]float32{
		{unit(0.9)}, // index 0: far
		{unit(0.1)}, // index 1: near
		{unit(0.4)}, // index 2: middle
	}
	ix := New(docs)
	if n := ix.Len(); n != 3 {
		t.Fatalf("Len() = %d, want 3", n)
	}
	hits := ix.Query(q, 3)
	if len(hits) != 3 {
		t.Fatalf("Query returned %d hits, want 3", len(hits))
	}
	wantOrder := []int{1, 2, 0} // near, middle, far
	for i, h := range hits {
		if h.Index != wantOrder[i] {
			t.Errorf("hits[%d].Index = %d, want %d (order %v)", i, h.Index, wantOrder[i], hits)
		}
	}
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Fatalf("hits not sorted best-first: %v", hits)
		}
	}

	if got := ix.Query(q, 0); got != nil {
		t.Errorf("Query(k=0) = %v, want nil", got)
	}
	if got := New(nil).Query(q, 5); got != nil {
		t.Errorf("Query on empty Index = %v, want nil", got)
	}
}

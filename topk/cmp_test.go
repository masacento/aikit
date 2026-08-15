package topk

import "testing"

func TestCmp_scoreDescThenIDAsc(t *testing.T) {
	cases := []struct {
		name           string
		aID, bID       int
		aScore, bScore float64
		want           int
	}{
		{"higher score first", 5, 1, 2.0, 1.0, -1},
		{"lower score last", 1, 5, 1.0, 2.0, 1},
		{"tie: lower id first", 1, 2, 1.0, 1.0, -1},
		{"tie: higher id last", 2, 1, 1.0, 1.0, 1},
		{"tie: equal id", 3, 3, 1.0, 1.0, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Cmp(c.aID, c.bID, c.aScore, c.bScore); got != c.want {
				t.Errorf("Cmp(%d, %d, %v, %v) = %d, want %d", c.aID, c.bID, c.aScore, c.bScore, got, c.want)
			}
		})
	}
}

func TestItemCmp_matchesCmp(t *testing.T) {
	a := ItemWithScore[int]{Item: 5, Score: 2.0}
	b := ItemWithScore[int]{Item: 1, Score: 1.0}
	if got, want := ItemCmp(a, b), Cmp(a.Item, b.Item, a.Score, b.Score); got != want {
		t.Errorf("ItemCmp = %d, want %d (matching direct Cmp)", got, want)
	}
}

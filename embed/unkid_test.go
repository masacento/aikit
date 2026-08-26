package embed

import "testing"

// TestResolveUnkID covers the <unk>-id helper extracted from the BPE tokenizer
// parser (docs/internal/task-code-health.md §4.4). Like encoder's order_test,
// it exists because the tests that would otherwise reach this code are all
// checkpoint-gated and skip without testdata — the precedence rule below was
// unverified by any running test before this.
func TestResolveUnkID(t *testing.T) {
	cases := []struct {
		name  string
		added map[string]int32
		vocab map[string]int32
		want  int32
	}{
		{"added wins over vocab", map[string]int32{"<unk>": 7}, map[string]int32{"<unk>": 3}, 7},
		{"falls back to vocab", map[string]int32{}, map[string]int32{"<unk>": 3}, 3},
		{"absent everywhere is 0", map[string]int32{"a": 1}, map[string]int32{"b": 2}, 0},
		{"nil maps are 0", nil, nil, 0},
		{"a real id of 0 is indistinguishable from absent, by design", map[string]int32{"<unk>": 0}, nil, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := resolveUnkID(c.added, c.vocab); got != c.want {
				t.Errorf("got %d, want %d", got, c.want)
			}
		})
	}
}

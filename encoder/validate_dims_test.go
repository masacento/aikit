package encoder

import "testing"

// TestValidateDims covers the shared helper directly rather than only through
// each Load path's own switch — bert.go/gte.go/weights.go have no unit tests
// of their own rejection switches (only full-checkpoint parity tests, which
// always pass a valid config), so this is the actual gate on validateDims's
// four cases.
func TestValidateDims(t *testing.T) {
	// A valid RoPE config: 768/12 = 64, even.
	if err := validateDims("T", 768, 12, 12, 3072, true); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	// Same, non-RoPE caller.
	if err := validateDims("T", 768, 12, 12, 3072, false); err != nil {
		t.Fatalf("valid config rejected (no RoPE): %v", err)
	}

	for _, tc := range []struct {
		name                                string
		hidden, heads, layers, intermediate int
		requireEvenHeadDim                  bool
	}{
		{"zero hidden", 0, 12, 12, 3072, false},
		{"zero heads", 768, 0, 12, 3072, false},
		{"zero layers", 768, 12, 0, 3072, false},
		{"zero intermediate", 768, 12, 12, 0, false},
		{"not divisible", 769, 12, 12, 3072, false},
		// 768/256 = 3, odd; RoPE required.
		{"odd head dim, RoPE required", 768, 256, 12, 3072, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateDims("T", tc.hidden, tc.heads, tc.layers, tc.intermediate, tc.requireEvenHeadDim); err == nil {
				t.Errorf("expected an error, got nil")
			}
		})
	}

	// The one case that must NOT reject: an odd head dim when the caller
	// doesn't need RoPE (BERT's learned absolute positions never touch
	// rope.go, so nothing panics on an odd head dim).
	if err := validateDims("T", 768, 256, 12, 3072, false); err != nil {
		t.Errorf("odd head dim rejected for a non-RoPE caller: %v", err)
	}
}

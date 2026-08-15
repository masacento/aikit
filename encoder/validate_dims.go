package encoder

import "fmt"

// validateDims checks the handful of dimension assumptions every encoder Load
// path shares, regardless of which config.json vocabulary it reads them from
// (bertConfig's Hidden/Heads, gteConfig's same, or Config's HiddenDim/
// NumHeads): hidden/heads/layers/intermediate must all be positive, and
// hidden must divide evenly by heads.
//
// requireEvenHeadDim additionally requires hidden/heads to be even. Every
// RoPE-using forward needs this — rope.go's newRopeTable panics on an odd
// head-dim (rotate_half splits it in half) — so GTE and CodeRankEmbed/
// NomicBert (both RoPE) pass true; BERT's learned absolute positions never
// touch rope.go, so it passes false.
//
// who names the caller in the returned error ("BERT", "GTE", "CodeRankEmbed")
// so a validation failure says which loader rejected the checkpoint.
func validateDims(who string, hidden, heads, layers, intermediate int, requireEvenHeadDim bool) error {
	switch {
	case hidden == 0 || heads == 0 || layers == 0 || intermediate == 0:
		return fmt.Errorf("encoder: %s config missing a required dim (hidden=%d heads=%d layers=%d intermediate=%d)",
			who, hidden, heads, layers, intermediate)
	case hidden%heads != 0:
		return fmt.Errorf("encoder: %s hidden %d not divisible by heads %d", who, hidden, heads)
	case requireEvenHeadDim && (hidden/heads)%2 != 0:
		return fmt.Errorf("encoder: %s head dim %d must be even for RoPE", who, hidden/heads)
	}
	return nil
}

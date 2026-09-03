package encoder

import (
	"os"
	"path/filepath"
	"testing"
)

// TestModernBertConfig_ropeParameters pins how the two RoPE thetas are resolved.
//
// This is the plan's silent-corruption case: transformers ≥5.7 dropped the flat
// global_rope_theta / local_rope_theta keys for a nested rope_parameters block, and
// the loader defaults a missing theta to 10000 rather than erroring. Ettin declares
// 160000, so before this resolution order existed the checkpoint loaded clean and
// computed every rotary position at a 16×-wrong base. No weights are involved, so
// this runs unconditionally rather than asset-gated.
func TestModernBertConfig_ropeParameters(t *testing.T) {
	base := `"model_type":"modernbert","hidden_size":256,"num_hidden_layers":7,` +
		`"num_attention_heads":4,"intermediate_size":384,"hidden_activation":"gelu",` +
		`"position_embedding_type":"sans_pos","global_attn_every_n_layers":3,` +
		`"local_attention":128`

	load := func(t *testing.T, body string) (modernBertConfig, error) {
		t.Helper()
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte("{"+base+","+body+"}"), 0o644); err != nil {
			t.Fatal(err)
		}
		return loadModernBertConfig(dir)
	}

	t.Run("nested", func(t *testing.T) {
		// The Ettin shape.
		c, err := load(t, `"rope_parameters":{
			"full_attention":{"rope_theta":160000.0,"rope_type":"default"},
			"sliding_attention":{"rope_theta":160000.0,"rope_type":"default"}}`)
		if err != nil {
			t.Fatal(err)
		}
		if c.GlobalRopeTheta != 160000 || c.LocalRopeTheta != 160000 {
			t.Fatalf("thetas = %g/%g, want 160000/160000", c.GlobalRopeTheta, c.LocalRopeTheta)
		}
	})

	t.Run("nested split thetas", func(t *testing.T) {
		// ModernBERT-base's actual values, in the 5.x spelling: the two classes
		// differ, so a single shared field would be wrong.
		c, err := load(t, `"rope_parameters":{
			"full_attention":{"rope_theta":160000.0},
			"sliding_attention":{"rope_theta":10000.0}}`)
		if err != nil {
			t.Fatal(err)
		}
		if c.GlobalRopeTheta != 160000 || c.LocalRopeTheta != 10000 {
			t.Fatalf("thetas = %g/%g, want 160000/10000", c.GlobalRopeTheta, c.LocalRopeTheta)
		}
	})

	t.Run("flat wins", func(t *testing.T) {
		// bekko / ruri-v3 keep working, and a checkpoint carrying both spellings
		// resolves to the flat one rather than depending on map ordering.
		c, err := load(t, `"global_rope_theta":160000.0,"local_rope_theta":10000.0,
			"rope_parameters":{"full_attention":{"rope_theta":1.0},
			"sliding_attention":{"rope_theta":2.0}}`)
		if err != nil {
			t.Fatal(err)
		}
		if c.GlobalRopeTheta != 160000 || c.LocalRopeTheta != 10000 {
			t.Fatalf("thetas = %g/%g, want 160000/10000", c.GlobalRopeTheta, c.LocalRopeTheta)
		}
	})

	t.Run("absent defaults", func(t *testing.T) {
		c, err := load(t, `"vocab_size":1`)
		if err != nil {
			t.Fatal(err)
		}
		if c.GlobalRopeTheta != 10000 || c.LocalRopeTheta != 10000 {
			t.Fatalf("thetas = %g/%g, want 10000/10000", c.GlobalRopeTheta, c.LocalRopeTheta)
		}
	})

	t.Run("local falls back to global", func(t *testing.T) {
		c, err := load(t, `"rope_parameters":{"full_attention":{"rope_theta":160000.0}}`)
		if err != nil {
			t.Fatal(err)
		}
		if c.LocalRopeTheta != 160000 {
			t.Fatalf("local theta = %g, want 160000 (inherited)", c.LocalRopeTheta)
		}
	})

	t.Run("scaled rope rejected", func(t *testing.T) {
		_, err := load(t, `"rope_parameters":{
			"full_attention":{"rope_theta":160000.0,"rope_type":"yarn"},
			"sliding_attention":{"rope_theta":160000.0,"rope_type":"default"}}`)
		if err == nil {
			t.Fatal("rope_type=yarn accepted; position scaling is not implemented")
		}
	})
}

// TestModernBertConfig_ettin loads the real checkpoint's config.json, so the shape
// above stays tied to the artifact rather than to a hand-written approximation.
func TestModernBertConfig_ettin(t *testing.T) {
	const dir = "../testdata/ettin-reranker-17m"
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skip("no ettin checkpoint — run scripts/fetch_ettin.sh")
	}
	c, err := loadModernBertConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if c.GlobalRopeTheta != 160000 || c.LocalRopeTheta != 160000 {
		t.Errorf("thetas = %g/%g, want 160000/160000", c.GlobalRopeTheta, c.LocalRopeTheta)
	}
	if c.Hidden != 256 || c.Layers != 7 || c.Heads != 4 || c.Intermediate != 384 {
		t.Errorf("dims = %d/%d/%d/%d, want 256/7/4/384", c.Hidden, c.Layers, c.Heads, c.Intermediate)
	}
	if c.LocalAttention != 128 || c.GlobalEvery != 3 {
		t.Errorf("local_attention=%d global_every=%d, want 128/3", c.LocalAttention, c.GlobalEvery)
	}
}

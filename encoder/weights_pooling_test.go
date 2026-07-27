package encoder

import (
	"os"
	"testing"
)

// TestLoadWeights_readsPooling is the regression for AUDIT #1: the mmap disk
// loader must derive pooling from 1_Pooling/config.json, not leave it "" (which
// poolOne treats as CLS). nomic-embed-text-v1.5 is mean-pooled, so a CLS result
// here is the silent-wrong the finding describes. LoadWeightsFromFS is checked
// alongside so the two loaders can't diverge again.
func TestLoadWeights_readsPooling(t *testing.T) {
	const dir = "../testdata/nomic-embed"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no nomic-embed fixture at %s", dir)
	}
	w, err := LoadWeights(dir)
	if err != nil {
		t.Fatal(err)
	}
	if w.Cfg.pooling != poolMean {
		t.Errorf("LoadWeights pooling = %q, want %q (mean-pooled checkpoint loaded as CLS)", w.Cfg.pooling, poolMean)
	}

	wf, err := LoadWeightsFromFS(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	if wf.Cfg.pooling != w.Cfg.pooling {
		t.Errorf("LoadWeights pooling %q != LoadWeightsFromFS pooling %q — the two loaders diverge", w.Cfg.pooling, wf.Cfg.pooling)
	}

	// LoadQ8 wraps LoadWeights, so it inherits the fix.
	q8, err := LoadWeightsQ8(dir)
	if err != nil {
		t.Fatal(err)
	}
	if q8.Cfg.pooling != poolMean {
		t.Errorf("LoadWeightsQ8 pooling = %q, want %q", q8.Cfg.pooling, poolMean)
	}
}

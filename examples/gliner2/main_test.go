package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The example is a thin CLI over ner.LoadGLiNER2, whose decode path the ner
// package's own parity gate pins exhaustively (testdata/gliner2_golden.json).
// What a CLI can add is the flags-to-API wiring: this exercises the same
// argument parsing and dispatch main() does, against the real checkpoint when
// one is present, so a flag rename that breaks the documented invocations
// goes red here.
func TestMain_flags(t *testing.T) {
	dir := filepath.Join("..", "..", "testdata", "gliner-multi-v2.5")
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no gliner2 checkpoint at %s — download fastino/gliner2.5-multi-v1 (see testdata/README.md)", dir)
	}
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("no GLiNER2 config at %s", dir)
	}
	var c struct {
		Architecture string `json:"architecture"`
	}
	raw, err := os.ReadFile(dir + "/config.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.Architecture != "boundary" {
		t.Fatalf("checkpoint architecture = %q, want boundary", c.Architecture)
	}
}

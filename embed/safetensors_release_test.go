package embed

import (
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

// TestReleaseTensorsIsLossless is the correctness contract behind
// ReleaseTensors: releasing a tensor's pages must not change what reading it
// afterwards returns. The mapping is read-only and file-backed, so a released
// page re-faults identical bytes — that is exactly what makes it safe for
// LoadWeightsQ8 to drop a tensor it *thinks* it is done with.
//
// The fixture is deliberately several pages wide; a sub-page tensor has an empty
// PageAlignedInterior, so a one-element tensor would exercise nothing.
func TestReleaseTensorsIsLossless(t *testing.T) {
	const n = 1 << 15 // 128 KiB of f32 — spans many pages on any target
	want := make([]float32, n)
	for i := range want {
		want[i] = float32(i) * 0.5
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	blob := buildF32Safetensors(map[string][]float32{"big.weight": want, "small.weight": {1, 2, 3}})
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}

	sf, err := OpenSafetensorsMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sf.Close() }()

	checkTensor(t, sf, "big.weight", want) // fault it in first
	if err := sf.ReleaseTensors("big.weight", "small.weight"); err != nil {
		t.Fatalf("ReleaseTensors: %v", err)
	}
	checkTensor(t, sf, "big.weight", want) // re-faults; must be byte-identical
	checkTensor(t, sf, "small.weight", []float32{1, 2, 3})
	// Releasing twice is not an error, and neither is releasing something never read.
	if err := sf.ReleaseTensors("big.weight"); err != nil {
		t.Errorf("second ReleaseTensors: %v", err)
	}
}

// TestReleaseTensorsUnknownName pins the one thing ReleaseTensors does report.
// Advisory madvise failures are swallowed; a name that doesn't resolve is a
// caller bug, and silently ignoring it would let a tensor rename turn a
// footprint fix into a no-op with every test still green (encoder's
// TestLayerQ8TensorNamesResolve is the consumer of this signal).
func TestReleaseTensorsUnknownName(t *testing.T) {
	blob := buildF32Safetensors(map[string][]float32{"a.weight": {1, 2}})
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	sf, err := OpenSafetensorsMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sf.Close() }()

	if err := sf.ReleaseTensors("a.weight", "nope.weight"); err == nil {
		t.Error("ReleaseTensors accepted an unknown tensor name")
	}
	// A bad name must not stop the good ones in the same call from being released,
	// and must not corrupt them either.
	checkTensor(t, sf, "a.weight", []float32{1, 2})
}

// TestReleaseTensorsHeapBacked: the heap opens have no pages to drop, so
// ReleaseTensors is a lookup-only no-op there. It must NOT reach madvise — the
// bytes are Go heap memory, where MADV_DONTNEED would zero them.
func TestReleaseTensorsHeapBacked(t *testing.T) {
	want := []float32{1, 2, 3, 4}
	blob := buildF32Safetensors(map[string][]float32{"a.weight": want})
	sf, err := OpenSafetensorsFromFS(fstest.MapFS{"m.safetensors": &fstest.MapFile{Data: blob}}, "m.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if len(sf.mmapped) != 0 {
		t.Fatal("fixture is not heap-backed")
	}
	if err := sf.ReleaseTensors("a.weight"); err != nil {
		t.Errorf("ReleaseTensors on a heap-backed file: %v", err)
	}
	checkTensor(t, sf, "a.weight", want)
	if err := sf.ReleaseTensors("nope"); err == nil {
		t.Error("unknown name should error on a heap-backed file too")
	}
}

//go:build linux

package gpu

import "testing"

// TestCUDA_graphReplay verifies the Graph primitive: a captured segment replays bit-identically to
// direct launches, AND it reads the CURRENT buffer contents on each replay (the property the MoE
// expert cache relies on — the graph topology is fixed, only the DMA'd slot contents change). If
// replay froze the inputs at capture time, the second replay would return the first result.
func TestCUDA_graphReplay(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(smokePTX)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "vadd")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	a := d.NewBufferFloats([]float32{1, 2, 3, 4, 5})
	b := d.NewBufferFloats([]float32{10, 20, 30, 40, 50})
	out := d.NewBufferLen(5)
	nb := d.NewBufferU32(5)

	// Capture vadd(a,b,out,n) — a pure async launch, no sync inside the closure.
	g, err := q.Capture(func() error { return q.launch(p, 5, 5, []Buffer{a, b, out, nb}) })
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	defer g.Close()

	read := func() []float32 {
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, 5)
		if err := out.ReadFloats(got); err != nil {
			t.Fatalf("ReadFloats: %v", err)
		}
		return got
	}

	// Replay 1: out = a + b.
	if err := g.Replay(); err != nil {
		t.Fatalf("Replay: %v", err)
	}
	got := read()
	want := []float32{11, 22, 33, 44, 55}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("replay1 out[%d]=%v want %v", i, got[i], want[i])
		}
	}

	// Change a's CONTENTS (same buffer pointer) and replay: the graph must read the new bytes.
	if err := Upload(a, []float32{100, 200, 300, 400, 500}); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if err := g.Replay(); err != nil {
		t.Fatalf("Replay2: %v", err)
	}
	got = read()
	want = []float32{110, 220, 330, 440, 550}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("replay2 out[%d]=%v want %v — graph froze inputs at capture (contents-vary broken)", i, got[i], want[i])
		}
	}
	t.Logf("Graph: captured vadd replays bit-identically and reads current buffer contents across replays")
}

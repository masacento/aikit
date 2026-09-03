package encoder

import (
	"encoding/json"
	"math"
	"os"
	"runtime"
	"testing"
)

// TestModernBERTCrossEncoder_parity pins the reranker against the Python golden
// (scripts/oracle/pin_ettin_reranker.py) on two independent legs:
//
//	forward — feed the golden's exact input_ids; scoreIDs must reproduce the logit.
//	          Isolates the trunk + head arithmetic.
//	live    — call Score(query, doc) and let aikit do its own pair framing. This is
//	          the leg that proves the byte-level BPE work in embed: a tokenizer that
//	          drops [CLS]/[SEP] or skips NFC passes the forward leg and fails here.
//
// Plus a ranking assertion, because both legs above are absolute-value comparisons
// and a head wired up backwards can still be numerically close on a single pair.
func TestModernBERTCrossEncoder_parity(t *testing.T) {
	const dir = "../testdata/ettin-reranker-17m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("no ettin checkpoint — run scripts/fetch_ettin.sh")
	}
	ce, err := LoadModernBERTCrossEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()

	raw, err := os.ReadFile("../testdata/ettin_reranker_golden.json")
	if err != nil {
		t.Skip("no golden — run scripts/oracle/pin_ettin_reranker.py")
	}
	var gld struct {
		Labels int `json:"labels"`
		MaxSeq int `json:"max_seq"`
		Cases  []struct {
			Query    string    `json:"query"`
			Doc      string    `json:"doc"`
			InputIDs []int32   `json:"input_ids"`
			Score    []float32 `json:"score"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &gld); err != nil {
		t.Fatal(err)
	}
	if ce.labels != gld.Labels {
		t.Fatalf("labels %d != golden %d", ce.labels, gld.Labels)
	}
	// sentence_bert_config.json declares no max_seq_length, so the cap is
	// max_position_embeddings. If the two sides disagree they read different amounts
	// of a long document, which the truncation cases below would show only as a
	// confusing score drift.
	if ce.mb.maxSeq != gld.MaxSeq {
		t.Fatalf("maxSeq %d != golden %d", ce.mb.maxSeq, gld.MaxSeq)
	}

	var worstFwd, worstLive float64
	var sawTrim bool
	for _, c := range gld.Cases {
		// (1) forward parity: the golden's exact ids.
		got := ce.scoreIDs(c.InputIDs)
		for l := range got {
			if d := math.Abs(float64(got[l]) - float64(c.Score[l])); d > worstFwd {
				worstFwd = d
			}
		}

		// (2) live parity: aikit's own tokenization and pair framing.
		live, err := ce.ScoreAll(c.Query, c.Doc)
		if err != nil {
			t.Fatalf("%.40q: ScoreAll: %v", c.Query, err)
		}
		if ids := ce.pairIDs(c.Query, c.Doc); !eqIDs(ids, c.InputIDs) {
			t.Errorf("%.40q/%.40q: pair ids differ from golden (len %d vs %d)",
				c.Query, c.Doc, len(ids), len(c.InputIDs))
		}
		for l := range live {
			if d := math.Abs(float64(live[l]) - float64(c.Score[l])); d > worstLive {
				worstLive = d
			}
		}
		if len(c.InputIDs) == gld.MaxSeq {
			sawTrim = true
		}
		t.Logf("L=%5d score %+8.4f  %.32q / %.32q", len(c.InputIDs), c.Score[0], c.Query, c.Doc)
	}
	if worstFwd > 5e-3 {
		t.Errorf("forward parity: worst |Δ| %.2e > 5e-3", worstFwd)
	}
	if worstLive > 5e-3 {
		t.Errorf("live parity: worst |Δ| %.2e > 5e-3 (pair framing or tokenizer)", worstLive)
	}
	if !sawTrim {
		t.Error("no case hit the max-length trim; the longest_first path is uncovered")
	}
	t.Logf("ettin reranker parity over %d cases: forward %.2e, live %.2e", len(gld.Cases), worstFwd, worstLive)
}

// TestModernBERTCrossEncoder_ranking asserts the reranker actually ranks. The parity
// test compares magnitudes against a golden, which a head wired up backwards can
// satisfy; this asserts the ORDER a caller depends on, and does it through the
// public surface only.
func TestModernBERTCrossEncoder_ranking(t *testing.T) {
	const dir = "../testdata/ettin-reranker-17m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("no ettin checkpoint — run scripts/fetch_ettin.sh")
	}
	ce, err := LoadModernBERTCrossEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()

	const query = "What is the capital of France?"
	docs := []string{
		"Paris is the capital and most populous city of France.",
		"Berlin is the capital of Germany.",
		"Sourdough needs a starter, flour, water and salt.",
	}
	scores, err := ce.ScoreBatch(query, docs, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(scores); i++ {
		if scores[i] >= scores[i-1] {
			t.Errorf("score[%d]=%.4f >= score[%d]=%.4f — expected relevant > related > unrelated",
				i, scores[i], i-1, scores[i-1])
		}
	}
	// ScoreBatch dispatches longest-first and writes results by index; a mix-up
	// there would reorder the output, so cross-check against the serial path.
	for i, d := range docs {
		one, err := ce.Score(query, d)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(one-scores[i])) > 1e-5 {
			t.Errorf("doc %d: ScoreBatch %.6f != Score %.6f — batch result misordered?", i, scores[i], one)
		}
	}
	t.Logf("scores: %v", scores)
}

// TestModernBERTCrossEncoder_rejectsBadChain asserts the module-chain validation
// fires. A checkpoint with a different chain is a different computation, and this
// forward would otherwise score it and return plausible numbers.
func TestModernBERTCrossEncoder_rejectsBadChain(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(dir+"/modules.json",
		[]byte(`[{"idx":0,"path":"","type":"sentence_transformers.models.Transformer"},
		         {"idx":1,"path":"1_Pooling","type":"sentence_transformers.models.Pooling"}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadModernBERTCrossEncoder(dir); err == nil {
		t.Fatal("a 2-module chain loaded; the head this forward applies is not present")
	}
}

func TestModernBERTCrossEncoder_rejectsUnsupportedDenseBias(t *testing.T) {
	dir := t.TempDir()
	modules := `[
		{"path":"","type":"sentence_transformers.models.Transformer"},
		{"path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
		{"path":"2_Dense","type":"sentence_transformers.models.Dense"},
		{"path":"3_LayerNorm","type":"sentence_transformers.models.LayerNorm"},
		{"path":"4_Dense","type":"sentence_transformers.models.Dense"}
	]`
	if err := os.WriteFile(dir+"/modules.json", []byte(modules), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, module := range []string{"2_Dense", "4_Dense"} {
		if err := os.MkdirAll(dir+"/"+module, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(dir+"/2_Dense/config.json",
		[]byte(`{"activation_function":"GELU","bias":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir+"/4_Dense/config.json",
		[]byte(`{"activation_function":"Identity","bias":true}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := checkModuleChain(dir); err == nil {
		t.Fatal("2_Dense bias=true accepted even though scoreIDs never applies that bias")
	}
}

// TestModernBERTCrossEncoder_footprint pins the loaded model's own memory.
//
// 77% of this checkpoint's parameters are the word-embedding table (50368×256 of
// 16.8M), so the number here is dominated by one tensor and a regression almost
// certainly means it stopped being read straight off the mmap. The head is heap-owned
// on purpose (~0.3 MB across three files, cloned so the mappings can close), and the
// gate is loose enough not to fail on allocator noise while still catching a
// widening of the trunk.
func TestModernBERTCrossEncoder_footprint(t *testing.T) {
	const dir = "../testdata/ettin-reranker-17m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skip("no ettin checkpoint — run scripts/fetch_ettin.sh")
	}
	// Three GCs, not one, on BOTH sides. Earlier tests in this package leave
	// hundreds of MiB of forward scratch in a sync.Pool, and pooled objects are
	// only reclaimed on the second GC after they go idle (the first demotes them
	// to the victim cache). With a single GC the baseline reads high and the
	// delta comes out negative — a measurement that silently passes any threshold.
	settle := func() uint64 {
		for range 3 {
			runtime.GC()
		}
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return m.HeapAlloc
	}
	before := settle()

	ce, err := LoadModernBERTCrossEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()
	after := settle()
	runtime.KeepAlive(ce) // the model must be live when `after` is sampled

	heapMiB := (float64(after) - float64(before)) / (1 << 20)
	fi, err := os.Stat(dir + "/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	onDiskMiB := float64(fi.Size()) / (1 << 20)
	t.Logf("on-disk %.1f MiB, heap retained by the loaded model %.2f MiB", onDiskMiB, heapMiB)
	if heapMiB < 0 {
		t.Fatalf("negative heap delta %.2f MiB — the baseline did not settle, so this gate is measuring nothing", heapMiB)
	}

	// The trunk's 64 MiB stays in the mapping, so the heap holds the tokenizer
	// (~6 MiB of vocab + merge tables — the dominant term), the head, the rotary
	// tables and bookkeeping. The ceiling is set just under two tokenizers: the
	// trunk and the cross-encoder each need one, and loading rather than sharing
	// the second is precisely the regression worth catching here.
	if heapMiB > 10 {
		t.Errorf("heap grew %.2f MiB on load, want < 10 — trunk weights copied, or the tokenizer loaded twice?", heapMiB)
	}
}

func eqIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

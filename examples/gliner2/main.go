// Command gliner is a standalone zero-shot extraction example built on GLiNER2
// (the "boundary" architecture): it extracts entities of whatever types you name
// at the command line, or classifies the text over classes you name — with no
// fine-tuning and no label schema baked into the weights.
//
// That is the whole point of the model, and the thing to try first is changing the
// --labels flag on the same sentence. The types go through the tokenizer and the
// backbone like any other text, so "person" and "famous scientist" are different
// prompts and give different answers.
//
//	go run ./examples/gliner2 --model testdata/gliner-multi-v2.5 \
//	    --text "織田信長は日本の武将である" --labels person
//	go run ./examples/gliner2 --model testdata/gliner-multi-v2.5 \
//	    --text "社員総会で合併が承認された" --task sentiment \
//	    --labels positive,negative,neutral
//
// The checkpoint comes from the hub:
//
//	uvx --from huggingface_hub hf download fastino/gliner2.5-multi-v1 \
//	    --local-dir testdata/gliner-multi-v2.5
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/ner"
)

func main() {
	modelDir := flag.String("model", "", "dir with a GLiNER2 checkpoint")
	text := flag.String("text", "Barack Obama was born in Honolulu, Hawaii on August 4, 1961.", "text to extract from")
	labels := flag.String("labels", "person,location,date", "comma-separated entity types (or classes with --task)")
	task := flag.String("task", "", "classification task name (runs --labels as classes)")
	threshold := flag.Float64("threshold", 0.5, "minimum score for a span to be reported")
	nested := flag.Bool("nested", false, "allow overlapping spans (nested NER)")
	flag.Parse()

	if *modelDir == "" {
		fmt.Println(`gliner — zero-shot extraction (GLiNER2 boundary).

Needs one local model:
  --model <dir>   a GLiNER2 checkpoint (e.g. testdata/gliner-multi-v2.5)

Download:
  uvx --from huggingface_hub hf download fastino/gliner2.5-multi-v1 \
      --local-dir testdata/gliner-multi-v2.5

Then try changing --labels on the same --text: the entity types are a PROMPT, not
a schema, so the model's answer changes with them and no retraining is involved.`)
		return
	}

	if *task != "" {
		runClassify(*modelDir, *text, *task, *labels)
		return
	}
	runExtract(*modelDir, *text, *labels, *threshold, *nested)
}

func runExtract(dir, text, labels string, threshold float64, nested bool) {
	t0 := time.Now()
	m, err := load(dir)
	check(err, "load model")
	defer func() { check(m.Close(), "close model") }()
	tLoad := time.Since(t0)
	fmt.Printf("model loaded: GLiNER2 boundary, max %d words\n\n", m.MaxLen())

	types := splitList(labels)
	t0 = time.Now()
	ents, err := m.Predict(text, types, ner.Opts{Threshold: threshold, Nested: nested, CJKSplit: true})
	check(err, "predict")
	tPredict := time.Since(t0)

	fmt.Printf("text:   %q\n", text)
	fmt.Printf("labels: %v\n\n", types)
	if len(ents) == 0 {
		fmt.Println("(no entities above the threshold)")
	}
	for _, e := range ents {
		fmt.Printf("  %-24s %-12s [%3d,%3d)  %.4f\n", e.Text, e.Label, e.Start, e.End, e.Score)
		if text[e.Start:e.End] != e.Text {
			fmt.Fprintln(os.Stderr, "gliner: offsets do not re-slice to the span text")
			os.Exit(1)
		}
	}
	fmt.Printf("\nmodel load: %10.2f ms\n", float64(tLoad.Microseconds())/1000)
	fmt.Printf("predict:    %10.2f ms\n", float64(tPredict.Microseconds())/1000)
}

func runClassify(dir, text, task, labels string) {
	t0 := time.Now()
	m, err := load(dir)
	check(err, "load model")
	defer func() { check(m.Close(), "close model") }()
	tLoad := time.Since(t0)

	classes := splitList(labels)
	t0 = time.Now()
	res, err := m.Classify(text, task, classes, false)
	check(err, "classify")
	tPredict := time.Since(t0)

	fmt.Printf("text:  %q\n", text)
	fmt.Printf("task:  %s %v\n\n", task, classes)
	best := 0
	for i, r := range res {
		mark := ""
		if r.Prob > res[best].Prob {
			best = i
			mark = "  <-"
		}
		fmt.Printf("  %-14s %.4f%s\n", r.Label, r.Prob, mark)
	}
	fmt.Printf("\nmodel load: %10.2f ms\n", float64(tLoad.Microseconds())/1000)
	fmt.Printf("classify:   %10.2f ms\n", float64(tPredict.Microseconds())/1000)
}

func splitList(s string) []string {
	parts := strings.Split(s, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

// load wraps LoadGLiNER2 with a download hint when the checkpoint directory is
// absent, so the first-run failure points at the fix rather than at the loader.
func load(dir string) (*ner.GLiNER2, error) {
	m, err := ner.LoadGLiNER2(dir)
	if err != nil {
		if _, statErr := os.Stat(dir); statErr != nil {
			fmt.Fprintf(os.Stderr, "gliner: no checkpoint at %s — download it first:\n"+
				"  uvx --from huggingface_hub hf download fastino/gliner2.5-multi-v1 \\\n"+
				"      --local-dir %s\n", dir, dir)
			os.Exit(1)
		}
	}
	return m, err
}

func check(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "gliner: %s: %v\n", what, err)
		os.Exit(1)
	}
}

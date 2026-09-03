// Command secretmasker is a CLI wrapper over the pinned DistilBERT secret
// detector (AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs), mirroring the
// model card's own cli.py: read text from a file or stdin, report the secret
// spans (or a JSON dump), and optionally write the masked text out.
//
// All model machinery is the generic token-classification path in the ner
// package; this file owns only the secret-flavored presentation: line numbers,
// the mask replacement, and the flags.
//
//	# spans, like cli.py's default output
//	go run ./examples/secretmasker --model testdata/distilbert-secret-masker-v3.3a-rs \
//	    config.env
//	echo "password = hunter2!" | go run ./examples/secretmasker \
//	    --model testdata/distilbert-secret-masker-v3.3a-rs
//
//	# JSON span dump (byte offsets — the Python reference reports character
//	# offsets; identical on ASCII)
//	go run ./examples/secretmasker --model … --json secret.txt
//
//	# masked text to a file (span replacement, right-to-left so offsets hold)
//	go run ./examples/secretmasker --model … --mask-output clean.txt secret.txt
//
// The checkpoint comes from the hub:
//
//	uvx --from huggingface_hub hf download AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs \
//	    --local-dir testdata/distilbert-secret-masker-v3.3a-rs
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/ner"
)

const defaultMask = "***MASKED***"

func main() {
	model := flag.String("model", filepath.Join("testdata", "distilbert-secret-masker-v3.3a-rs"),
		"dir with the DistilBERT secret-masker checkpoint")
	tau := flag.Float64("tau", 0.99, "span-confidence threshold; 0 keeps every decoded span")
	mask := flag.String("mask", defaultMask, "replacement token for --mask-output")
	jsonOut := flag.Bool("json", false, "output spans as JSON")
	maskOutput := flag.String("mask-output", "", "write the masked text to this file")
	flag.Parse()

	text, err := readInput(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	m, err := ner.LoadTokenClassifier(*model)
	if err != nil {
		if _, statErr := os.Stat(*model); statErr != nil {
			fmt.Fprintf(os.Stderr, "secretmasker: no checkpoint at %s — download it first:\n"+
				"  uvx --from huggingface_hub hf download AndrewAndrewsen/distilbert-secret-masker-v3.3a-rs \\\n"+
				"      --local-dir %s\n", *model, *model)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "secretmasker: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := m.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "secretmasker: close model: %v\n", err)
		}
	}()

	spans, err := m.Predict(text, ner.TokenOpts{Tau: *tau})
	if err != nil {
		fmt.Fprintf(os.Stderr, "secretmasker: %v\n", err)
		os.Exit(1)
	}

	if *jsonOut {
		if err := dumpJSON(text, spans); err != nil {
			fmt.Fprintf(os.Stderr, "secretmasker: %v\n", err)
			os.Exit(1)
		}
	} else {
		for _, s := range spans {
			fmt.Printf("L%d [%d:%d] score=%.3f %q\n", lineOf(text, s.Start), s.Start, s.End, s.Score, s.Text)
		}
	}

	if *maskOutput != "" {
		if err := os.WriteFile(*maskOutput, []byte(maskSpans(text, spans, *mask)), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "secretmasker: write masked text: %v\n", err)
			os.Exit(1)
		}
	}
}

// readInput reads the operand file, or stdin for "-" / no operand — cli.py's
// contract.
func readInput(arg string) (string, error) {
	if arg == "" || arg == "-" {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return "", fmt.Errorf("read stdin: %w", err)
		}
		return string(b), nil
	}
	b, err := os.ReadFile(arg)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", arg, err)
	}
	return string(b), nil
}

// dumpJSON prints the spans as a JSON array with the reference's field names,
// scores rounded to 6 decimals like round(score, 6).
func dumpJSON(text string, spans []ner.TokenEntity) error {
	type jsonSpan struct {
		Start int     `json:"start"`
		End   int     `json:"end"`
		Line  int     `json:"line"`
		Value string  `json:"value"`
		Score float64 `json:"score"`
	}
	out := make([]jsonSpan, len(spans))
	for i, s := range spans {
		out[i] = jsonSpan{
			Start: s.Start, End: s.End, Line: lineOf(text, s.Start), Value: s.Text,
			Score: math.Round(s.Score*1e6) / 1e6,
		}
	}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(b))
	return nil
}

// lineOf is the reference's line field: a 1-based count of newlines up to the
// span start.
func lineOf(text string, start int) int {
	return strings.Count(text[:start], "\n") + 1
}

// maskSpans replaces every span with mask, right-to-left so the byte offsets
// of the not-yet-applied spans stay valid — the reference's mask_text.
func maskSpans(text string, spans []ner.TokenEntity, mask string) string {
	out := text
	for i := len(spans) - 1; i >= 0; i-- {
		s := spans[i]
		out = out[:s.Start] + mask + out[s.End:]
	}
	return out
}

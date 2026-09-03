package cjk

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// AveragedPerceptron is a POS-tagging model (Averaged Perceptron) loaded from
// a trained model file. It holds the class list and per-feature class weights.
type AveragedPerceptron struct {
	classes []string             // sorted class names (labels like "B-NOUN", "O")
	weights map[string][]float64 // feature -> weights indexed by class index
}

// LoadAveragedPerceptron reads an Averaged Perceptron model file.
//
// The model format matches litsea's save_model output: the first line is the
// number of classes, the next N lines are the class names, and the remaining
// lines are "feature\tclass\tweight" (only non-zero weights are saved).
func LoadAveragedPerceptron(path string) (*AveragedPerceptron, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	if !scanner.Scan() {
		return nil, fmt.Errorf("empty model file")
	}
	numClasses, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil {
		return nil, fmt.Errorf("invalid class count: %s", scanner.Text())
	}

	classes := make([]string, 0, numClasses)
	for i := 0; i < numClasses; i++ {
		if !scanner.Scan() {
			return nil, fmt.Errorf("unexpected end of model file while reading classes")
		}
		classes = append(classes, strings.TrimSpace(scanner.Text()))
	}

	weights := make(map[string][]float64)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) != 3 {
			return nil, fmt.Errorf("invalid weight line: %q", line)
		}
		feat, class, weightStr := parts[0], parts[1], parts[2]
		classIdx := indexOf(classes, class)
		if classIdx < 0 {
			return nil, fmt.Errorf("unknown class in weight line: %q", line)
		}
		w, err := strconv.ParseFloat(weightStr, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid weight value: %s", weightStr)
		}
		slot, ok := weights[feat]
		if !ok {
			slot = make([]float64, numClasses)
			weights[feat] = slot
		}
		slot[classIdx] = w
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return &AveragedPerceptron{classes: classes, weights: weights}, nil
}

// best returns the class name with the highest score, or "" if no classes are
// registered. Ties resolve to the lowest class index.
func (p *AveragedPerceptron) best(scores []float64) string {
	if p == nil || len(p.classes) == 0 {
		return ""
	}
	bestIdx := 0
	for i, s := range scores {
		if s > scores[bestIdx] {
			bestIdx = i
		}
	}
	return p.classes[bestIdx]
}

func indexOf(classes []string, target string) int {
	for i, c := range classes {
		if c == target {
			return i
		}
	}
	return -1
}

package cjk

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// AdaBoost is a word-segmentation model loaded from a trained AdaBoost model
// file. It holds the feature weights and the bias term used for scoring.
type AdaBoost struct {
	weights map[string]float64
	bias    float64
}

// LoadAdaBoost reads an AdaBoost model file.
//
// The model format matches litsea's save_model output: each line is
// "feature\tweight" (feature names may embed any non-tab character), and the
// last line is a single bias value. Weight lines after the bias line are
// accepted for compatibility with legacy models (e.g. RWCP.model).
func LoadAdaBoost(path string) (*AdaBoost, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return loadAdaBoost(f)
}

// LoadAdaBoostFromBytes parses an AdaBoost model file already in memory.
func LoadAdaBoostFromBytes(data []byte) (*AdaBoost, error) {
	return loadAdaBoost(bytes.NewReader(data))
}

func loadAdaBoost(r io.Reader) (*AdaBoost, error) {
	m := make(map[string]float64)
	biasSeen := false
	bias := 0.0

	scanner := bufio.NewScanner(r)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if line == "" {
			return nil, fmt.Errorf("empty line at line %d", lineNum)
		}
		if strings.ContainsRune(line, '\t') {
			parts := strings.SplitN(line, "\t", 2)
			feat := parts[0]
			value, err := strconv.ParseFloat(parts[1], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid value at line %d: %s", lineNum, parts[1])
			}
			if _, dup := m[feat]; dup {
				return nil, fmt.Errorf("duplicate feature at line %d: %q", lineNum, feat)
			}
			m[feat] = value
		} else {
			if biasSeen {
				return nil, fmt.Errorf("duplicate bias line at line %d", lineNum)
			}
			b, err := strconv.ParseFloat(line, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid bias at line %d: %s", lineNum, line)
			}
			biasSeen = true
			bias = b
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if lineNum == 0 {
		return nil, fmt.Errorf("empty model file")
	}
	if !biasSeen {
		return nil, fmt.Errorf("model file has no bias line; the file may be truncated")
	}

	return &AdaBoost{weights: m, bias: bias}, nil
}

// Bias returns the model's bias term.
func (a *AdaBoost) Bias() float64 {
	if a == nil {
		return 0
	}
	return a.bias
}

// weight returns the model weight of a single attribute (0.0 if unknown).
func (a *AdaBoost) weight(attr string) float64 {
	if a == nil {
		return 0
	}
	return a.weights[attr]
}

# CJK word segmentation

The `cjk` package is a Go port of the compact inference core from
[mosuka/litsea](https://github.com/mosuka/litsea). Litsea is a Rust word
segmenter inspired by TinySegmenter and TinySegmenterMaker. It predicts word
boundaries from character n-gram features and a compact trained model instead of
loading a morphological dictionary.

This fork uses the port to give word-level NER finer boundaries on Japanese,
Chinese, and Korean text while keeping the root module pure Go.

## Port scope

The package ports the parts needed for local inference rather than the whole
Litsea project.

| Capability | This Go package |
|---|---|
| Japanese, Chinese, Korean segmentation | Supported |
| Embedded default segmentation models | Supported |
| External AdaBoost-format models | Supported from a path or bytes |
| Averaged-perceptron model loading and POS inference | Supported with a caller-supplied model |
| English | Not exposed as a package language |
| Training, corpus preparation, CLI, and language bindings | Not ported |

The embedded `japanese.model`, `chinese.model`, and `korean.model` files come
from Litsea. `Default` parses each model once and returns a shared, read-only
segmenter.

## Basic use

```go
seg, err := cjk.Default(cjk.Japanese)
if err != nil {
    return err
}

words := seg.Segment("これはテストです。")
// []string{"これ", "は", "テスト", "です", "。"}
```

Use `SplitWords` when byte offsets into the original string are required:

```go
text := "アップルは2007年にiPhoneを発表した。"
tokens := seg.SplitWords(text)
for _, token := range tokens {
    surface := text[token.Start:token.End]
    _ = surface
}
```

`Token.Start` and `Token.End` are UTF-8 byte offsets. The returned tokens
partition the complete input without rewriting it, so concatenating their
surfaces reconstructs the original string.

## Language selection

The package exposes three language constants:

```go
cjk.Japanese
cjk.Chinese
cjk.Korean
```

`ParseLanguage` accepts the lowercase names `"japanese"`, `"chinese"`, and
`"korean"`.

For automatic routing, `DetectCJK` applies a script-based rule:

- Hangul selects Korean.
- Hiragana or Katakana selects Japanese.
- Bare Han text selects Japanese because script inspection alone cannot
  distinguish Japanese from Chinese.
- Text without kana, Hangul, or Han returns `ok == false`.

Callers that know the document language should select it explicitly, especially
for Chinese text containing only Han characters.

`HasCJK` detects kana, Hangul, or Han, while `HasHangul` performs the narrower
Hangul check.

## Custom segmentation models

Load a Litsea-compatible AdaBoost model from disk:

```go
model, err := cjk.LoadAdaBoost("models/custom.model")
if err != nil {
    return err
}
seg := cjk.NewSegmenter(cjk.Chinese, model)
words := seg.Segment("我是中国人")
```

`LoadAdaBoostFromBytes` provides the same parser for embedded or otherwise
in-memory model data. The accepted text format is Litsea's segmentation model
format: feature and weight pairs separated by a tab, plus one serialized bias
line. Legacy files with weight rows after the bias are accepted.

The language passed to `NewSegmenter` must match the feature templates used to
train the model.

## POS tagging

`LoadAveragedPerceptron` loads a compatible class/feature-weight model, and
`NewPosSegmenter` enables `SegmentWithPos`:

```go
model, err := cjk.LoadAveragedPerceptron("models/japanese-pos.model")
if err != nil {
    return err
}
seg := cjk.NewPosSegmenter(cjk.Japanese, model)
tagged, err := seg.SegmentWithPos("これはテストです。")
```

Each result contains a word and a `Upos` value. The package defines the 17
Universal Dependencies UPOS tags. POS models are not embedded, so callers must
supply a model whose format and feature templates match this port.

## Algorithm and implementation

For each possible boundary, the segmenter evaluates Litsea's character,
character-type, and preceding-boundary-tag feature templates. AdaBoost weights
and the model bias decide whether the current character begins a new word.

At load time, the Go port compiles string feature names into packed integer keys.
Inference then uses an immutable open-addressed table, avoiding feature-string
construction and allocation in the scoring loop. Sentence context, boundary
tags, and byte-offset tables remain local to each call, so a loaded segmenter can
be shared by concurrent readers.

## NER integration

`ner.Opts{CJKSplit: true}` enables CJK segmentation for GLiNER2. The option is
off by default because the Python reference splitter treats a contiguous CJK run
as one word; enabling it deliberately trades exact splitter parity for finer
entity boundaries.

When enabled, GLiNER2 segments only CJK word runs. Latin text, URLs, email
addresses, handles, and punctuation continue through the reference-compatible
splitter. If no language applies or a default model cannot be loaded, NER falls
back to the unsplit run.

## Provenance and license

The implementation and embedded models are derived from
[Litsea](https://github.com/mosuka/litsea). Litsea is distributed under the MIT
License and includes material originally developed by Taku Kudo under the BSD
3-Clause License. The notices shipped with this port are preserved in
[`cjk/LICENSE`](../cjk/LICENSE).

This document describes the Go port in this repository; refer to the upstream
Litsea project for its Rust CLI, training workflow, current model metrics, and
other language bindings.

## Tests

```bash
GOWORK=off go test ./cjk ./ner
GOWORK=off go test -race ./cjk
```

The `cjk` suite checks embedded-model loading, segmentation, language detection,
byte-offset partitioning, feature tables, and legacy model parsing. NER has
separate tests for the optional CJK splitting path.

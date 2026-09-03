package encoder

import (
	"math"
	"os"
	"testing"
)

// TestModernBERTQ8_ruriParity certifies that the int8-quantized ModernBERT
// forward produces embeddings with cosine ≥ 0.99 vs the f32 forward on
// cl-nagoya/ruri-v3-30m. The Q8 path quantizes Wqkv/AttnWo/MLPWi/MLPWo to
// per-row int8; LN weights and the embedding table stay f32.
func TestModernBERTQ8_ruriParity(t *testing.T) {
	const dir = "../testdata/ruri-v3-30m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no ruri-v3-30m at %s", dir)
	}

	f32m, err := LoadModernBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f32m.Close()

	q8m, err := LoadModernBERTQ8(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer q8m.Close()

	cases := []string{
		"寿司の特徴は何ですか？",
		"天ぷらは魚や野菜に衣をつけて揚げた料理です。",
		"ラーメンは小麦粉の麺をスープに入れた日本の料理です。",
		"Python is a popular programming language for data science.",
		"量子化はモデルのメモリを削減する",
		"",
		"hello world",
	}

	worstCos := 1.0
	for _, tc := range cases {
		f32Emb, err := f32m.Encode(tc)
		if err != nil {
			t.Fatalf("f32 encode %q: %v", tc, err)
		}
		q8Emb, err := q8m.Encode(tc)
		if err != nil {
			t.Fatalf("q8 encode %q: %v", tc, err)
		}
		cos := cosineSim(f32Emb, q8Emb)
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-50q  cosine=%.6f", tc, cos)
	}

	const floor = 0.99
	if worstCos < floor {
		t.Errorf("worst cosine %.6f < floor %.2f", worstCos, floor)
	}
	t.Logf("ruri Q8 parity over %d cases: worst cosine %.6f", len(cases), worstCos)
}

// TestModernBERTQ8_bekkoParity certifies the same Q8 parity on bekko
// (hotchpotch/bekko-embedding-v1-a8m), the other ModernBERT spelling.
func TestModernBERTQ8_bekkoParity(t *testing.T) {
	const dir = "../testdata/bekko-embedding-v1-a8m"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no bekko-embedding-v1-a8m at %s", dir)
	}

	f32m, err := LoadModernBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f32m.Close()

	q8m, err := LoadModernBERTQ8(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer q8m.Close()

	cases := []string{
		"compute the sha256 hash of a file",
		"識別子を検索する",
		"Bonjour le monde, ça va?",
		"The quick brown fox jumps over the lazy dog.",
	}

	worstCos := 1.0
	for _, tc := range cases {
		f32Emb, err := f32m.Encode(tc)
		if err != nil {
			t.Fatalf("f32 encode %q: %v", tc, err)
		}
		q8Emb, err := q8m.Encode(tc)
		if err != nil {
			t.Fatalf("q8 encode %q: %v", tc, err)
		}
		cos := cosineSim(f32Emb, q8Emb)
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-50q  cosine=%.6f", tc, cos)
	}

	const floor = 0.99
	if worstCos < floor {
		t.Errorf("worst cosine %.6f < floor %.2f", worstCos, floor)
	}
	t.Logf("bekko Q8 parity over %d cases: worst cosine %.6f", len(cases), worstCos)
}

// TestModernBERTQ8_prequantizedI8 certifies the pre-quantized checkpoint path:
// a model stored I8-on-disk (per-tensor symmetric int8 + companion scales)
// loads with the projections wrapped directly off the mmap — no load-time
// quantization — and stays within the cosine floor of the f32 forward.
func TestModernBERTQ8_prequantizedI8(t *testing.T) {
	const dir = "../testdata/ruri-v3-30m-int8"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no ruri-v3-30m-int8 at %s", dir)
	}

	q8m, err := LoadModernBERTQ8(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer q8m.Close()

	// The projections must be int8-resident straight off the disk bytes, with
	// the mmap kept alive to back them.
	for i, l := range q8m.layers {
		if kind := l.Wqkv.Kind(); kind != "int8" {
			t.Fatalf("layer %d Wqkv kind = %q, want int8", i, kind)
		}
	}
	if q8m.st == nil {
		t.Fatal("pre-quantized load released the mmap; the int8 weights alias it")
	}

	// Parity vs the f32 model, when it is also present.
	const f32dir = "../testdata/ruri-v3-30m"
	if _, err := os.Stat(f32dir + "/model.safetensors"); err != nil {
		t.Skipf("no ruri-v3-30m at %s for the parity check", f32dir)
	}
	f32m, err := LoadModernBERT(f32dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f32m.Close()

	cases := []string{
		"寿司の特徴は何ですか？",
		"天ぷらは魚や野菜に衣をつけて揚げた料理です。",
		"ラーメンは小麦粉の麺をスープに入れた日本の料理です。",
		"Python is a popular programming language for data science.",
		"量子化はモデルのメモリを削減する",
		"hello world",
	}

	worstCos := 1.0
	for _, tc := range cases {
		f32Emb, err := f32m.Encode(tc)
		if err != nil {
			t.Fatalf("f32 encode %q: %v", tc, err)
		}
		q8Emb, err := q8m.Encode(tc)
		if err != nil {
			t.Fatalf("q8 encode %q: %v", tc, err)
		}
		cos := cosineSim(f32Emb, q8Emb)
		if cos < worstCos {
			worstCos = cos
		}
		t.Logf("%-50q  cosine=%.6f", tc, cos)
	}

	const floor = 0.99
	if worstCos < floor {
		t.Errorf("worst cosine %.6f < floor %.2f", worstCos, floor)
	}
	t.Logf("pre-quantized I8 parity over %d cases: worst cosine %.6f", len(cases), worstCos)
}

func cosineSim(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

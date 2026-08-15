package ann

import "testing"

// TestFlatBinaryI8_recallReal_Model2Vec is TestFlatBinary_recallReal_Model2Vec
// for the int8-reranked sibling — same real Model2Vec corpus (realVecs, see
// flat_binary_real_test.go), same overquery sweep, ground truth still Flat's
// exact float32 scan. This is the "prefilter approximation + int8
// quantization combined" measurement FlatBinaryI8's package doc points at.
func TestFlatBinaryI8_recallReal_Model2Vec(t *testing.T) {
	vecs, queries := realVecs(t)
	flat := New(vecs)
	t.Logf("real corpus: %d Model2Vec embeddings, dim %d, %d queries", len(vecs), len(vecs[0]), len(queries))

	for _, k := range []int{10, 50} {
		var atDefault float64
		for _, over := range []int{1, 2, 4, 8, 16, 32} {
			r := binRecallAt(flat, NewFlatBinaryI8Overquery(vecs, over), queries, k)
			t.Logf("k=%2d overquery %2d: recall %.4f", k, over, r)
			if over == DefaultOverquery {
				atDefault = r
			}
		}
		if atDefault < 0.85 {
			t.Errorf("k=%d: recall at DefaultOverquery (%d) = %.4f, want >= 0.85", k, DefaultOverquery, atDefault)
		}
	}
}

package bm25

import (
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
)

func TestTokens_roundTrip(t *testing.T) {
	docs := [][]string{
		Tokenize("func readLines(path string) ([]string, error)"),
		Tokenize("the quick brown fox"),
		{}, // an empty document must round-trip as empty, not nil-vs-empty flaky
		Tokenize("unicode café résumé"),
	}
	blob, err := MarshalTokens(docs)
	if err != nil {
		t.Fatalf("MarshalTokens: %v", err)
	}
	got, err := UnmarshalTokens(blob)
	if err != nil {
		t.Fatalf("UnmarshalTokens: %v", err)
	}
	if len(got) != len(docs) {
		t.Fatalf("got %d docs, want %d", len(got), len(docs))
	}
	for i := range docs {
		if !reflect.DeepEqual(got[i], docs[i]) {
			// An empty doc round-trips as a nil slice (docs[i] with len 0
			// decodes with ntok==0, which the loop leaves as the zero value).
			// TopK/Build treat nil and empty identically, so require only
			// that both are empty, not that nilness matches exactly.
			if len(got[i]) != 0 || len(docs[i]) != 0 {
				t.Errorf("doc %d: got %v, want %v", i, got[i], docs[i])
			}
		}
	}
}

func TestTokens_buildFromRoundTripMatchesDirectBuild(t *testing.T) {
	raw := []string{
		"func readLines(path string) ([]string, error)",
		"the quick brown fox jumps",
		"quick quick quick sort algorithm",
		"unrelated content about databases",
	}
	var docs [][]string
	for _, s := range raw {
		docs = append(docs, Tokenize(s))
	}
	want := Build(docs)

	blob, err := MarshalTokens(docs)
	if err != nil {
		t.Fatal(err)
	}
	cachedDocs, err := UnmarshalTokens(blob)
	if err != nil {
		t.Fatal(err)
	}
	got := Build(cachedDocs)

	for _, q := range [][]string{
		Tokenize("read a file line by line"),
		Tokenize("quick sort"),
		Tokenize("databases"),
	} {
		if !reflect.DeepEqual(got.TopK(q, 10), want.TopK(q, 10)) {
			t.Fatalf("query %v: TopK differs after a MarshalTokens/UnmarshalTokens/Build round trip\n got %v\nwant %v",
				q, got.TopK(q, 10), want.TopK(q, 10))
		}
	}
}

func TestTokens_emptyCorpus(t *testing.T) {
	blob, err := MarshalTokens(nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := UnmarshalTokens(blob)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want empty", got)
	}
}

func TestTokens_badMagic(t *testing.T) {
	if _, err := UnmarshalTokens([]byte("not a token cache blob, but long enough")); err == nil {
		t.Fatal("expected an error for bad magic")
	} else if !errors.Is(err, ErrFormat) {
		t.Errorf("error %v does not wrap ErrFormat", err)
	}
}

func TestTokens_wrongVersion(t *testing.T) {
	docs := [][]string{{"a", "b"}}
	blob, err := MarshalTokens(docs)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(blob[4:], tokensVersion+1)
	if _, err := UnmarshalTokens(blob); err == nil {
		t.Fatal("expected an error for an unsupported version")
	} else if !errors.Is(err, ErrFormat) {
		t.Errorf("error %v does not wrap ErrFormat", err)
	}
}

func TestTokens_truncated(t *testing.T) {
	docs := [][]string{{"one", "two", "three"}, {"four"}}
	blob, err := MarshalTokens(docs)
	if err != nil {
		t.Fatal(err)
	}
	for cut := 0; cut < len(blob); cut++ {
		if _, err := UnmarshalTokens(blob[:cut]); err == nil {
			t.Fatalf("truncated to %d/%d bytes: expected an error, got none", cut, len(blob))
		} else if !errors.Is(err, ErrFormat) {
			t.Fatalf("truncated to %d bytes: error %v does not wrap ErrFormat", cut, err)
		}
	}
}

func TestTokens_trailingBytes(t *testing.T) {
	blob, err := MarshalTokens([][]string{{"x"}})
	if err != nil {
		t.Fatal(err)
	}
	blob = append(blob, 0xFF)
	if _, err := UnmarshalTokens(blob); err == nil {
		t.Fatal("expected an error for trailing bytes")
	} else if !errors.Is(err, ErrFormat) {
		t.Errorf("error %v does not wrap ErrFormat", err)
	}
}

func TestTokens_hostileCountDoesNotOOM(t *testing.T) {
	// A declared ndocs of ~4 billion over a 12-byte blob must be rejected by
	// the remaining-bytes bound, not attempt a multi-GB make([][]string, ...).
	blob := make([]byte, 12)
	binary.LittleEndian.PutUint32(blob[0:], tokensMagic)
	binary.LittleEndian.PutUint32(blob[4:], tokensVersion)
	binary.LittleEndian.PutUint32(blob[8:], 0)  // flags
	blob = append(blob, 0xFF, 0xFF, 0xFF, 0x7F) // ndocs = 0x7FFFFFFF
	if _, err := UnmarshalTokens(blob); err == nil {
		t.Fatal("expected an error for a hostile document count")
	} else if !errors.Is(err, ErrFormat) {
		t.Errorf("error %v does not wrap ErrFormat", err)
	}
}

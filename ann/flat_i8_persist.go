package ann

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
	"runtime"
	"unsafe"
)

// FlatI8 serialization — the //go:embed-an-index entry point for the int8 index,
// the one you'd most want to embed (¼ the float32 memory at ~equal recall). Like
// the HNSW format it is versioned from day one so the on-disk layout can evolve
// without silently mis-reading old blobs:
//
//	magic uint32 | version uint32
//	dim int32 | n int32
//	codes:  n × dim int8 (row-major, one byte each)
//	scales: n × float32 (little-endian)
//
// All integers little-endian. The int8 codes come first (1 byte each, no alignment
// constraint) so LoadFlatI8Mmap can alias them straight from a read-only mapping;
// the small scales block is always copied. flatI8Layout validates the payload size
// against the remaining bytes before any allocation, so a corrupt or hostile blob
// returns an error rather than panicking or over-allocating.
// Format-stability policy (pre-1.0): rebuild-per-minor — Load rejects any other
// version with ErrFormat (loud, never a silent misread). See README "Serialized blob
// formats". FORMAT-BUMP CHECKLIST: the next version bump should add a reserved uint32
// flags word right after the version, so later additive changes extend via flags
// without a version bump (the anti-churn mechanism). FlatI8 already zero-copy-mmaps
// its int8 codes (1-byte, no alignment); the f32 scales are copied, so no alignment
// change is needed here (unlike HNSW's float32 vectors).
const (
	flatI8Magic   uint32 = 0x46493800 // "FI8\0"
	flatI8Version uint32 = 1
)

// MarshalBinary serializes the int8 index (codes + per-vector scales + shape) into
// a versioned blob that LoadFlatI8 / LoadFlatI8Mmap turn back into a query-ready
// *FlatI8. It implements encoding.BinaryMarshaler, so the index also round-trips
// through gob.
//
// The point is the //go:embed pattern: quantize the corpus once offline, embed the
// bytes, and Load at startup — no float32 vectors, no re-quantization per process.
func (f *FlatI8) MarshalBinary() ([]byte, error) {
	// Allocated at exact capacity but LENGTH 16, then appended into. Using
	// make([]byte, total) instead would zero the whole payload before overwriting
	// every byte of it — a wasted pass over ~19 MB on a 50k×384 index. Appending
	// into spare capacity is a memmove with no pre-zeroing.
	//
	// The previous version appended the code block ONE BYTE AT A TIME and pushed
	// every scale through a `put32` closure that captured `b` by reference, so a
	// 1M×768 index cost 768M appends plus 1M non-inlinable indirect calls.
	b := make([]byte, 16, 16+len(f.bq)+len(f.scales)*4)
	binary.LittleEndian.PutUint32(b[0:], flatI8Magic)
	binary.LittleEndian.PutUint32(b[4:], flatI8Version)
	binary.LittleEndian.PutUint32(b[8:], uint32(int32(f.dim)))
	binary.LittleEndian.PutUint32(b[12:], uint32(int32(f.n)))
	if len(f.bq) > 0 {
		// int8 → byte is the two's-complement round-trip, and the two have identical
		// layout, so the whole code block is one memmove. This is the same aliasing
		// LoadFlatI8Mmap already relies on (flat_i8_mmap.go), in the other direction.
		b = append(b, unsafe.Slice((*byte)(unsafe.Pointer(&f.bq[0])), len(f.bq))...)
	}
	for _, s := range f.scales {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(s))
	}
	return b, nil
}

// WriteTo streams the same bytes MarshalBinary returns, without building them.
//
// It implements io.WriterTo, so os.File, bufio.Writer, gzip.Writer and
// io.Copy all pick it up automatically.
//
// WHY IT EXISTS. MarshalBinary allocates the whole blob — 16 bytes plus the code
// block plus four bytes per vector — and a caller writing an index to disk
// therefore holds the index AND a second full copy of it at once. On a
// 1M×768 index that is ~770 MB of transient heap to write 770 MB of file. The
// blob is not needed as a value by anyone who is only going to write it; it was
// simply the only shape offered (lens doc §4.3).
//
// Streaming changes the transient cost from a full copy to a 4 KB buffer. The
// code block is written straight from the index's own memory in one Write — the
// same int8/byte aliasing MarshalBinary and LoadFlatI8Mmap both already rely on —
// so only the scales pass through the buffer.
//
// Byte-for-byte identical to MarshalBinary; TestFlatI8_WriteToMatchesMarshal
// asserts it, so the two cannot drift into different formats.
func (f *FlatI8) WriteTo(w io.Writer) (int64, error) {
	// f.bq may alias an mmap owned by f (LoadFlatI8Mmap); keep f reachable for
	// the whole write so a finalizer cannot unmap it mid-Write.
	defer runtime.KeepAlive(f)
	if f.closed {
		return 0, errors.New("ann: WriteTo on a closed FlatI8 (mmap released by Close)")
	}
	var total int64
	var hdr [16]byte
	binary.LittleEndian.PutUint32(hdr[0:], flatI8Magic)
	binary.LittleEndian.PutUint32(hdr[4:], flatI8Version)
	binary.LittleEndian.PutUint32(hdr[8:], uint32(int32(f.dim)))
	binary.LittleEndian.PutUint32(hdr[12:], uint32(int32(f.n)))
	n, err := w.Write(hdr[:])
	total += int64(n)
	if err != nil {
		return total, err
	}
	if len(f.bq) > 0 {
		n, err = w.Write(unsafe.Slice((*byte)(unsafe.Pointer(&f.bq[0])), len(f.bq)))
		total += int64(n)
		if err != nil {
			return total, err
		}
	}
	// 4 KB of scales at a time: large enough that the syscall count is
	// negligible next to the code block's single Write, small enough to stay a
	// stack-sized buffer rather than a second allocation proportional to n.
	var buf [4096]byte
	for i := 0; i < len(f.scales); {
		k := min(len(buf)/4, len(f.scales)-i)
		for j := range k {
			binary.LittleEndian.PutUint32(buf[j*4:], math.Float32bits(f.scales[i+j]))
		}
		n, err = w.Write(buf[:k*4])
		total += int64(n)
		if err != nil {
			return total, err
		}
		i += k
	}
	return total, nil
}

// fcur is a bounds-checked little-endian reader over a FlatI8 header (the
// per-format cursor convention, alongside HNSW's hcur and gguf's gcur). Every read
// goes through need(), so a truncated input sets err instead of panicking.
type fcur struct {
	b   []byte
	pos int
	err error
}

func (c *fcur) need(n int) bool {
	if c.err != nil {
		return false
	}
	if n < 0 || n > len(c.b)-c.pos {
		c.err = errFormatf("ann: FlatI8 blob truncated (need %d at %d of %d)", n, c.pos, len(c.b))
		return false
	}
	return true
}

func (c *fcur) u32() uint32 {
	if !c.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(c.b[c.pos:])
	c.pos += 4
	return v
}

// nonNeg reads an int32 shape scalar (dim, n) and rejects a negative value.
func (c *fcur) nonNeg(name string) int {
	v := int32(c.u32())
	if c.err != nil {
		return 0
	}
	if v < 0 {
		c.err = errFormatf("ann: FlatI8 %s %d is negative", name, v)
		return 0
	}
	return int(v)
}

// flatI8Layout validates the header (magic, version, dim, n) and the exact payload
// size, returning the dims and the byte offset where the int8 codes begin. Shared
// by LoadFlatI8 (copies the codes) and LoadFlatI8Mmap (aliases them).
func flatI8Layout(data []byte) (dim, n, codesAt int, err error) {
	c := &fcur{b: data}
	if c.u32() != flatI8Magic {
		return 0, 0, 0, errFormatf("ann: not a FlatI8 blob (bad magic)")
	}
	if v := c.u32(); v != flatI8Version {
		return 0, 0, 0, errFormatf("ann: unsupported FlatI8 format version %d (want %d)", v, flatI8Version)
	}
	dim = c.nonNeg("dim")
	n = c.nonNeg("n")
	if c.err != nil {
		return 0, 0, 0, c.err
	}
	// An index is either empty (n=0, dim=0, like NewFlatI8(nil)) or non-empty
	// (both > 0). Reject the mixed cases — in particular n=0 with a huge dim, which
	// the size check below would otherwise wave through (n*dim = 0), leaving a
	// loaded index whose dim could drive a gigantic query-vector allocation.
	if (n == 0) != (dim == 0) {
		return 0, 0, 0, errFormatf("ann: FlatI8 inconsistent shape (n=%d, dim=%d): both must be zero or both nonzero", n, dim)
	}
	codesAt = c.pos
	// Payload must be exactly n×dim code bytes + n×4 scale bytes. Computed in int64
	// so a hostile (n, dim) can't overflow into a small allocation; the exact-match
	// check also rejects truncation and trailing bytes in one shot.
	want := int64(n)*int64(dim) + int64(n)*4
	if got := int64(len(data) - codesAt); want != got {
		return 0, 0, 0, errFormatf("ann: FlatI8 payload size %d != n*dim + n*4 = %d (n=%d dim=%d)", got, want, n, dim)
	}
	return dim, n, codesAt, nil
}

// readScales reads n little-endian float32 from b (which must hold ≥ n*4 bytes).
// Always a copy — the scales are tiny (n floats) and copying sidesteps the 4-byte
// alignment an aliased float32 view would require.
func readScales(b []byte, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return s
}

// LoadFlatI8 reconstructs an index from MarshalBinary's output, copying the codes
// into the Go heap. The returned *FlatI8 is query-ready and read-only-safe for
// concurrent Query; the bytes are not retained. Returns an error for a bad magic,
// an unsupported version, or any truncated/inconsistent blob — never a panic. Use
// LoadFlatI8Mmap to avoid the copy for a large embedded index.
func LoadFlatI8(data []byte) (*FlatI8, error) {
	dim, n, at, err := flatI8Layout(data)
	if err != nil {
		return nil, err
	}
	bq := make([]int8, n*dim)
	for i := range bq {
		bq[i] = int8(data[at+i])
	}
	return &FlatI8{bq: bq, scales: readScales(data[at+n*dim:], n), n: n, dim: dim}, nil
}

package ann

import (
	"encoding/binary"
	"io"
	"math"
	"math/rand/v2"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/cursor"
)

// HNSW serialization. The format is versioned from day one so the on-disk /
// //go:embed-ed layout can evolve without silently mis-reading old blobs:
//
//	magic uint32 | version uint32
//	dim, ndocs, m, m0, efConstruction, efSearch, entry, maxLayer  (int32 each)
//	mL float64 | seed uint64 | heuristic uint8 (0/1) | int8 uint8 (0/1)
//	vectors:  int8 mode → ndocs×dim int8 codes + ndocs float32 scales;
//	          else      → ndocs × dim float32 (little-endian, row-major)
//	graph:    per node — layer int32, then for l in 0..layer:
//	          nbrCount int32, then nbrCount × neighbor id int32
//
// All integers little-endian. entry is int32 (it is -1 for an empty index); every
// other count is non-negative. Load validates every length against the bytes that
// remain, so a corrupt or hostile blob returns an error rather than panicking or
// over-allocating.
// Format-stability policy (pre-1.0): rebuild-per-minor — a blob is not a stable
// cross-version interchange format; Load rejects any other version with ErrFormat
// (loud, never a silent misread), so callers regenerate. See README "Serialized blob
// formats". FORMAT-BUMP CHECKLIST — the next version bump should bundle, in ONE bump
// (to curb the v1→v2→v3 churn):
//  1. a reserved uint32 flags word right after the version, so later additive
//     changes extend via flags WITHOUT a version bump (the anti-churn mechanism);
//  2. pad the header so the float32 vector block starts 8-byte aligned, unblocking
//     zero-copy mmap aliasing of the vectors (the deferred LoadHNSWMmap, mirroring
//     LoadFlatI8Mmap's int8 aliasing — §3.2).
const (
	hnswMagic   uint32 = 0x484E5357 // "HNSW"
	hnswVersion uint32 = 3          // v3 added the int8 storage-mode byte (+ int8 codes/scales)
	// rngSplit matches NewHNSW's PCG seeding so a loaded index re-creates an
	// equivalently-seeded rng (for Add-after-load).
	rngSplit uint64 = 0x9e3779b97f4a7c15
)

// blobSize returns the exact serialized length, which is both MarshalBinary's
// allocation size and WriteTo's reported total. Header + vectors + graph, where
// each neighbor id is 4 bytes and each node costs 4 for its layer plus 4 per
// layer for that layer's count.
func (h *HNSW) blobSize() int {
	nNbr, nLayers := 0, 0
	for _, nd := range h.nodes {
		nLayers += len(nd.nbrs)
		for _, l := range nd.nbrs {
			nNbr += len(l)
		}
	}
	vecBytes := h.count() * h.dim * 4
	if h.int8 {
		vecBytes = len(h.bq) + len(h.scales)*4
	}
	return hnswHeaderBytes + vecBytes + len(h.nodes)*4 + nLayers*4 + nNbr*4
}

// hnswHeaderBytes is the fixed prefix: magic+version+8 int32 config scalars
// (10×4), mL and seed (2×8), heuristic and int8 mode (2×1).
const hnswHeaderBytes = 10*4 + 2*8 + 2

// MarshalBinary serializes the built index — graph, vectors, and config — into a
// versioned byte blob that Load turns back into a query-ready *HNSW. It implements
// encoding.BinaryMarshaler, so the index also round-trips through gob and friends.
//
// The point is the //go:embed pattern: build the graph once offline, embed the
// bytes in the binary, and Load them at startup instead of rebuilding per process.
//
// This is a thin wrapper over WriteTo into an exactly-sized buffer, so the two
// cannot encode different bytes — there is only one encoder. Prefer WriteTo when
// the destination is a file or a network connection: this necessarily materializes
// a second full copy of the index (see WriteTo's doc for the numbers).
func (h *HNSW) MarshalBinary() ([]byte, error) {
	// Whole-blob mode: the destination IS the buffer, so the encoder writes each
	// byte exactly once — no staging copy, and one allocation at the exact size.
	// (The previous version's own capacity estimate was short — it budgeted 8
	// bytes of graph header per node when a node with L layers needs 4+4L — so
	// append doubled it: 131.0 MB allocated for a 58.3 MB blob.)
	bw := &hblobWriter{buf: make([]byte, h.blobSize())}
	h.encode(bw)
	if bw.err != nil {
		return nil, bw.err
	}
	return bw.buf[:bw.n], nil
}

// WriteTo streams the same bytes MarshalBinary returns, without building them.
//
// It implements io.WriterTo, so os.File, bufio.Writer, gzip.Writer and io.Copy
// all pick it up automatically.
//
// WHY IT EXISTS. MarshalBinary allocates the whole blob, so a caller writing an
// index to disk holds the index AND a second full copy of it at once. An HNSW
// index is the largest thing this package produces — at n=1M, dim=768 the f32
// vectors alone are 3.1 GB, and MarshalBinary doubles the peak to save it. The
// blob is not needed as a value by anyone who is only going to write it; it was
// simply the only shape offered (lens doc §4.3).
//
// Streaming changes the transient cost from a full copy to one 64 KiB buffer,
// whatever the index's size. In int8 mode the code block is written straight from
// the index's own memory in a single Write — the same int8/byte aliasing
// MarshalBinary and LoadFlatI8Mmap both already rely on — so only the scales and
// the graph pass through the buffer.
func (h *HNSW) WriteTo(w io.Writer) (int64, error) {
	bw := &hblobWriter{w: w, buf: make([]byte, hblobBufSize)}
	h.encode(bw)
	return bw.finish()
}

// hblobBufSize is the streaming staging buffer. Big enough that the syscall count
// is negligible (a 3 GB index is ~48k writes, and callers wrapping in a
// bufio.Writer pay nothing extra), small enough to be a fixed cost independent of
// index size — which is the entire point of WriteTo.
const hblobBufSize = 64 << 10

// encode is the single encoder both serialization surfaces run. There is
// deliberately no second copy of the format: MarshalBinary passes a writer whose
// buffer is the whole output, WriteTo passes one that flushes every 64 KiB, and
// the bytes are identical because the code producing them is identical.
func (h *HNSW) encode(bw *hblobWriter) {
	bw.u32(hnswMagic)
	bw.u32(hnswVersion)
	bw.puti(h.dim)
	bw.puti(h.count())
	bw.puti(h.m)
	bw.puti(h.m0)
	bw.puti(h.efConstruction)
	bw.puti(h.efSearch)
	bw.puti(h.entry)
	bw.puti(h.maxLayer)
	bw.u64(math.Float64bits(h.mL))
	bw.u64(h.seed)
	bw.boolByte(h.heuristic)
	bw.boolByte(h.int8) // v3: storage mode

	// Vectors: int8 codes + per-vector scales (int8 mode) or row-major f32.
	if h.int8 {
		if len(h.bq) > 0 {
			// int8 → byte is the two's-complement round-trip and the two have
			// identical layout, so the whole code block is one memmove out. Same
			// aliasing argument as flat_i8_persist.go:64.
			bw.raw(unsafe.Slice((*byte)(unsafe.Pointer(&h.bq[0])), len(h.bq)))
		}
		bw.f32s(h.scales)
	} else {
		// f32 is encoded, not aliased: unlike int8 a float32 view would bake in
		// the host's endianness, and the format is little-endian by contract.
		for _, v := range h.vecs {
			bw.f32s(v)
		}
	}
	for _, nd := range h.nodes {
		bw.puti(nd.layer)
		for l := 0; l <= nd.layer; l++ {
			bw.puti(len(nd.nbrs[l]))
			bw.i32s(nd.nbrs[l])
		}
	}
}

// hblobWriter is the encoder target for both serialization surfaces.
//
//	w != nil — STREAMING: buf is a fixed staging buffer, flushed to w when full.
//	w == nil — WHOLE-BLOB: buf is the entire output and is never flushed, so each
//	           byte is written exactly once and MarshalBinary needs no staging copy.
//
// Errors are sticky: the first failed Write is remembered and every later call is
// a no-op, so encode() reads as straight-line code instead of an error check per
// field.
type hblobWriter struct {
	w     io.Writer
	buf   []byte
	n     int
	total int64
	err   error
}

func (b *hblobWriter) flush() {
	if b.err != nil || b.n == 0 || b.w == nil {
		return
	}
	n, err := b.w.Write(b.buf[:b.n])
	b.total += int64(n)
	b.n = 0
	if err != nil {
		b.err = err
	}
}

// ensure makes room for at least need bytes and reports whether the caller may
// write them.
func (b *hblobWriter) ensure(need int) bool {
	if b.err != nil {
		return false
	}
	if b.n+need <= len(b.buf) {
		return true
	}
	if b.w == nil {
		// Whole-blob mode ran out, i.e. blobSize() under-counted. Grow rather than
		// fail: a miscount must not corrupt a caller's blob, and
		// TestHNSW_WriteToMatchesMarshal is what catches it as a defect.
		b.buf = append(b.buf, make([]byte, b.n+need-len(b.buf))...)
		return true
	}
	b.flush()
	return b.err == nil
}

func (b *hblobWriter) u32(v uint32) {
	if !b.ensure(4) {
		return
	}
	binary.LittleEndian.PutUint32(b.buf[b.n:], v)
	b.n += 4
}

func (b *hblobWriter) u64(v uint64) {
	if !b.ensure(8) {
		return
	}
	binary.LittleEndian.PutUint64(b.buf[b.n:], v)
	b.n += 8
}

func (b *hblobWriter) puti(v int) { b.u32(uint32(int32(v))) }

func (b *hblobWriter) boolByte(v bool) {
	if !b.ensure(1) {
		return
	}
	if v {
		b.buf[b.n] = 1
	} else {
		b.buf[b.n] = 0
	}
	b.n++
}

// f32s and i32s encode a whole slice at a time. The bulk form matters: the
// per-value entry point costs a space check and a sticky-error check per element,
// and at 12.8 M floats that overhead was 2× the encode itself. Here both checks
// are paid once per buffer-full, and the inner loop is a bounds-check-hoistable
// run of stores.
func (b *hblobWriter) f32s(vs []float32) {
	for len(vs) > 0 {
		if !b.ensure(4) {
			return
		}
		k := min(len(vs), (len(b.buf)-b.n)/4)
		dst := b.buf[b.n : b.n+k*4]
		for j, f := range vs[:k] {
			binary.LittleEndian.PutUint32(dst[j*4:], math.Float32bits(f))
		}
		b.n += k * 4
		vs = vs[k:]
	}
}

func (b *hblobWriter) i32s(vs []int32) {
	for len(vs) > 0 {
		if !b.ensure(4) {
			return
		}
		k := min(len(vs), (len(b.buf)-b.n)/4)
		dst := b.buf[b.n : b.n+k*4]
		for j, v := range vs[:k] {
			binary.LittleEndian.PutUint32(dst[j*4:], uint32(v))
		}
		b.n += k * 4
		vs = vs[k:]
	}
}

// raw emits p without staging it — for a block already contiguous in the index's
// own memory, where a copy through the buffer would be pointless. In whole-blob
// mode there is nowhere to stream to, so it copies into the output.
func (b *hblobWriter) raw(p []byte) {
	if b.w == nil {
		if !b.ensure(len(p)) {
			return
		}
		b.n += copy(b.buf[b.n:], p)
		return
	}
	b.flush()
	if b.err != nil {
		return
	}
	n, err := b.w.Write(p)
	b.total += int64(n)
	if err != nil {
		b.err = err
	}
}

func (b *hblobWriter) finish() (int64, error) {
	b.flush()
	return b.total, b.err
}

// hcur is a bounds-checked little-endian reader over the serialized blob —
// the shared primitive (internal/cursor.Cursor) plus HNSW-specific
// allocation-guarding readers. Counts are clamped to what the remaining
// bytes could hold before they drive an allocation.
type hcur struct {
	cursor.Cursor
}

func newHcur(data []byte) *hcur {
	return &hcur{cursor.Cursor{B: data, Context: "ann: HNSW blob", Errorf: errFormatf}}
}

// int8s reads n raw int8 code bytes (the int8 vector block).
func (c *hcur) int8s(n int) []int8 {
	raw := c.Bytes(n)
	if raw == nil {
		return nil
	}
	out := make([]int8, n)
	for i, b := range raw {
		out[i] = int8(b)
	}
	return out
}

// asInt reads a signed int32 (used for entry, which is -1 for an empty index).
func (c *hcur) asInt() int { return int(int32(c.U32())) }

// readLen reads an allocation-driving LENGTH (ndocs, dim, neighbor count) and
// rejects it if it can't fit the bytes that remain (every subsequent element is
// ≥4 bytes), so a hostile count can't drive a giant make() before the reads hit
// EOF.
func (c *hcur) readLen() int {
	v := int32(c.U32())
	if c.Err != nil {
		return 0
	}
	if v < 0 || int(v) > c.Remaining()/4 {
		c.Err = errFormatf("ann: HNSW blob count %d exceeds remaining bytes", v)
		return 0
	}
	return int(v)
}

// cfgMax bounds the config scalars (m, m0, efConstruction, efSearch). Unlike
// readLen() these aren't byte-sized lengths — they're tuning knobs — but efSearch
// sizes Query's candidate map, so an absurd value must not slip through and OOM.
const cfgMax = 1 << 20

// cfg reads a non-negative config scalar, rejecting negatives and absurd values.
func (c *hcur) cfg(name string) int {
	v := int32(c.U32())
	if c.Err != nil {
		return 0
	}
	if v < 0 || int(v) > cfgMax {
		c.Err = errFormatf("ann: HNSW %s %d out of [0,%d]", name, v, cfgMax)
		return 0
	}
	return int(v)
}

// i32Arena hands out non-overlapping sub-slices of one backing array. Each
// sub-slice is capped at its own length (a full three-index slice) so appending
// to one can never write into the next.
//
// take falls back to a fresh allocation when the arena is exhausted rather than
// panicking. That keeps the arena a pure optimization: a mis-sized or skipped
// pre-pass costs allocations, never correctness.
type i32Arena struct{ buf []int32 }

func (a *i32Arena) take(n int) []int32 {
	if n > cap(a.buf)-len(a.buf) {
		return make([]int32, n)
	}
	s := a.buf[len(a.buf) : len(a.buf)+n : len(a.buf)+n]
	a.buf = a.buf[:len(a.buf)+n]
	return s
}

// takeLayers is the same idea for the per-node [][]int32 header slices.
func takeLayers(arena *[][]int32, n int) [][]int32 {
	if n > cap(*arena)-len(*arena) {
		return make([][]int32, n)
	}
	s := (*arena)[len(*arena) : len(*arena)+n : len(*arena)+n]
	*arena = (*arena)[:len(*arena)+n]
	return s
}

// scanGraph walks the graph section from pos without allocating and returns the
// exact number of neighbor ids and per-layer slices it contains, so Load can size
// its two arenas in one shot.
//
// It cannot use a byte-count upper bound instead: ids are 4 bytes each in the blob
// so remaining/4 bounds them tightly, but each [][]int32 header is 24 bytes in
// memory, and sizing THAT from remaining/4 would turn a 6 GB blob into a 38 GB
// allocation — an amplification a hostile input could aim. An exact pre-pass has
// no such gap. It re-validates nothing: ok=false simply means "don't bother", and
// the real read reports the error.
func scanGraph(b []byte, pos, ndocs int) (nIDs, nLayers int, ok bool) {
	for range ndocs {
		if pos+4 > len(b) {
			return 0, 0, false
		}
		layer := int(int32(binary.LittleEndian.Uint32(b[pos:])))
		pos += 4
		if layer < 0 || layer > (len(b)-pos)/4 {
			return 0, 0, false
		}
		nLayers += layer + 1
		for range layer + 1 {
			if pos+4 > len(b) {
				return 0, 0, false
			}
			cnt := int(int32(binary.LittleEndian.Uint32(b[pos:])))
			pos += 4
			if cnt < 0 || cnt > (len(b)-pos)/4 {
				return 0, 0, false
			}
			nIDs += cnt
			pos += cnt * 4
		}
	}
	return nIDs, nLayers, true
}

// Load reconstructs an index from MarshalBinary's output — the //go:embed-an-index
// entry point. The returned *HNSW is query-ready and, like a freshly built one,
// read-only-safe for concurrent Query. The bytes are not retained (vectors are
// copied). Returns an error for a bad magic, an unsupported version, or any
// truncated/inconsistent blob — never a panic.
func Load(data []byte) (*HNSW, error) {
	c := newHcur(data)
	if c.U32() != hnswMagic {
		return nil, errFormatf("ann: not an HNSW blob (bad magic)")
	}
	if v := c.U32(); v != hnswVersion {
		return nil, errFormatf("ann: unsupported HNSW format version %d (want %d)", v, hnswVersion)
	}
	dim := c.readLen()
	ndocs := c.readLen()
	// match NewHNSW's clamp; a well-formed blob already has m ≥ 2
	m := max(c.cfg("m"), 2)
	m0 := c.cfg("m0")
	efc := c.cfg("efConstruction")
	efs := c.cfg("efSearch")
	entry := c.asInt()
	maxLayer := c.cfg("maxLayer")
	_ = c.U64() // consume the serialized mL to advance the cursor; recomputed below
	seed := c.U64()
	heuristic := c.U8() != 0
	int8mode := c.U8() != 0 // v3
	if c.Err != nil {
		return nil, c.Err
	}
	// Recompute mL from m rather than trust the serialized value: a crafted blob
	// could carry mL = +Inf/NaN/≤0, which randomLevel turns into an overflowing
	// level on Add-after-load (Load then Add is a documented capability) →
	// make([][]int32, level+1) panics. 1/ln(m) is deterministic and
	// round-trip-stable (NewHNSW stored exactly this), so a valid blob is
	// unaffected; Query never touches mL, so a load-only workflow was already safe.
	mL := 1.0 / math.Log(float64(m))

	var vecs [][]float32
	var bq []int8
	var scales []float32
	if int8mode {
		// Overflow-safe: the int8 codes + scales must fit the bytes before the
		// graph (computed in int64 so a hostile ndocs/dim can't wrap to a small
		// allocation).
		if int64(ndocs)*int64(dim)+int64(ndocs)*4 > int64(len(c.B)-c.Pos) {
			return nil, errFormatf("ann: HNSW int8 vector block (ndocs=%d dim=%d) exceeds remaining bytes", ndocs, dim)
		}
		bq = c.int8s(ndocs * dim)
		scales = make([]float32, ndocs)
		for i := range scales {
			scales[i] = c.F32()
		}
	} else {
		// Overflow-safe product guard, mirroring the int8 branch above:
		// count() clamps ndocs and dim individually but not their product, so
		// without this a crafted ~1 MB header (dim≈ndocs≈250k) drives ~250 GB
		// of row allocations. 4 bytes per f32.
		if int64(ndocs)*int64(dim)*4 > int64(len(c.B)-c.Pos) {
			return nil, errFormatf("ann: HNSW f32 vector block (ndocs=%d dim=%d) exceeds remaining bytes", ndocs, dim)
		}
		// One arena, sub-sliced per row, instead of ndocs separate allocations —
		// 1M rows was 1M allocations for a block that is contiguous in the blob
		// anyway. The guard above already proved ndocs*dim*4 fits the remaining
		// bytes, so this allocation is bounded by the input either way.
		vecs = make([][]float32, ndocs)
		arena := make([]float32, ndocs*dim)
		for d := range vecs {
			row := arena[d*dim : (d+1)*dim : (d+1)*dim]
			for j := range row {
				row[j] = c.F32()
			}
			vecs[d] = row
			if c.Err != nil { // stop reading once the input is exhausted
				return nil, c.Err
			}
		}
	}
	// The graph was 2 allocations per node — one [][]int32 of layer headers and one
	// []int32 per layer — so a 1M-doc index cost >2.1M allocations to load a
	// structure that is one contiguous run of int32s in the blob. Two arenas,
	// sized by a non-allocating pre-pass, make it 2. scanGraph returns ok=false on
	// a truncated blob, which leaves both arenas empty; take() then falls back to
	// make() and the read below produces exactly the error it always did.
	nIDs, nLayers, _ := scanGraph(c.B, c.Pos, ndocs)
	idArena := i32Arena{buf: make([]int32, 0, nIDs)}
	layerArena := make([][]int32, 0, nLayers)
	nodes := make([]hnswNode, ndocs)
	for d := range nodes {
		layer := c.readLen()
		nbrs := takeLayers(&layerArena, layer+1)
		for l := 0; l <= layer; l++ {
			cnt := c.readLen()
			ids := idArena.take(cnt)
			for i := range ids {
				ids[i] = int32(c.U32())
			}
			nbrs[l] = ids
		}
		nodes[d] = hnswNode{layer: layer, nbrs: nbrs}
	}
	if c.Err != nil {
		return nil, c.Err
	}
	if c.Pos != len(c.B) {
		return nil, errFormatf("ann: HNSW blob has %d trailing bytes", len(c.B)-c.Pos)
	}

	// Validate graph integrity: Query indexes vecs[id] and nodes[id].nbrs[layer]
	// directly, so a blob with out-of-range ids or layer-inconsistent edges would
	// panic mid-query. Reject it here instead.
	if ndocs == 0 {
		if entry != -1 {
			return nil, errFormatf("ann: empty HNSW must have entry -1, got %d", entry)
		}
	} else {
		if entry < 0 || entry >= ndocs {
			return nil, errFormatf("ann: HNSW entry %d out of [0,%d)", entry, ndocs)
		}
		if maxLayer > nodes[entry].layer {
			return nil, errFormatf("ann: HNSW maxLayer %d exceeds entry-node layer %d", maxLayer, nodes[entry].layer)
		}
	}
	for d := range nodes {
		for l := 0; l <= nodes[d].layer; l++ {
			for _, id := range nodes[d].nbrs[l] {
				if id < 0 || int(id) >= ndocs {
					return nil, errFormatf("ann: HNSW node %d layer %d neighbor id %d out of [0,%d)", d, l, id, ndocs)
				}
				if nodes[id].layer < l {
					return nil, errFormatf("ann: HNSW node %d layer %d links node %d, which exists only to layer %d", d, l, id, nodes[id].layer)
				}
			}
		}
	}

	return &HNSW{
		vecs:           vecs,
		int8:           int8mode,
		bq:             bq,
		scales:         scales,
		nodes:          nodes,
		dim:            dim,
		m:              m,
		m0:             m0,
		efConstruction: efc,
		efSearch:       efs,
		mL:             mL,
		entry:          entry,
		maxLayer:       maxLayer,
		seed:           seed,
		heuristic:      heuristic,
		rng:            rand.New(rand.NewPCG(seed, seed^rngSplit)),
	}, nil
}

//go:build darwin

// Package metal is the Phase-1 (Layer A) proof for the cgo-free native-Metal spike
// (docs/task-metal-cgofree-spike.md): can we reach Metal + compile/run MSL kernels
// through purego-objc with CGO_ENABLED=0 and correct output? This file is the
// binding skeleton — device, the runtime MSL compiler, and the ONE thing that must
// be right before any kernel: MTLCompileOptions.languageVersion.
//
// ⚠️ Risk #7 (the landmine): CGO_ENABLED=0 macOS binaries omit LC_BUILD_VERSION
// (golang/go#77917), so Metal's runtime compiler DEFAULTS languageVersion to MSL 2.4
// and silently strips modern types. We NEVER rely on the default: every library is
// compiled with an explicit MTLCompileOptions at MSL >=3.1, and we assert it took.
package gpu

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// MTLLanguageVersion values (NSUInteger). MSL 3.1 = (3<<16)|1. Rig is macOS 26, so
// >=3.1 is safe; bump to match features used.
const MSL3_1 uint = (3 << 16) | 1

var mtlCreateSystemDefaultDevice func() uintptr

// loadOnce guards the Metal.framework dlopen + symbol registration. It is a
// sync.Once rather than a package init so it does not depend on init ordering
// between files in this package (annbackend.go's init registers the backend and
// calls CreateSystemDefaultDevice, and Go runs inits in filename order — a plain
// init here would run after that call). Callers reach it via CreateSystemDefaultDevice.
var loadOnce sync.Once

func loadMetal() {
	loadOnce.Do(func() {
		h, err := purego.Dlopen("/System/Library/Frameworks/Metal.framework/Metal",
			purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			panic("metal: dlopen Metal.framework: " + err.Error())
		}
		purego.RegisterLibFunc(&mtlCreateSystemDefaultDevice, h, "MTLCreateSystemDefaultDevice")
	})
}

// cached selectors
var (
	selAlloc              = objc.RegisterName("alloc")
	selInit               = objc.RegisterName("init")
	selRelease            = objc.RegisterName("release")
	selSetLanguageVersion = objc.RegisterName("setLanguageVersion:")
	selLanguageVersion    = objc.RegisterName("languageVersion")
	selNewLibrarySource   = objc.RegisterName("newLibraryWithSource:options:error:")
	selStringWithUTF8     = objc.RegisterName("stringWithUTF8String:")
	selUTF8String         = objc.RegisterName("UTF8String")
	selLocalizedDesc      = objc.RegisterName("localizedDescription")
	selName               = objc.RegisterName("name")
)

// nsString wraps a Go string as an autoreleased NSString.
func nsString(s string) objc.ID {
	b := append([]byte(s), 0) // NUL-terminated C string
	id := objc.ID(objc.GetClass("NSString")).Send(selStringWithUTF8, unsafe.Pointer(&b[0]))
	_ = b // keep alive across the call
	return id
}

// goString reads a C `const char *` (id.UTF8String) back into a Go string.
func goString(id objc.ID) string {
	if id == 0 {
		return ""
	}
	p := objc.Send[*byte](id, selUTF8String)
	if p == nil {
		return ""
	}
	// Walk the NUL terminator with unsafe.Add (pointer arithmetic that stays *unsafe.Pointer),
	// then materialize with unsafe.Slice — avoids the uintptr→unsafe.Pointer round-trip that
	// `go vet` flags as "possible misuse" (and that CI's vet rejects). The bytes are C-managed
	// (objc UTF8String), so GC movement is not a concern; this is purely the vet-clean idiom.
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// Device wraps an MTLDevice and OWNS every MTLBuffer allocated through it.
//
// purego means no ARC: an MTLBuffer that is never sent -release leaks by construction. There is
// no Metal equivalent of CUDA's "destroy the context and reclaim everything", so the only way to
// free a resident model is to release each buffer — hence this ledger. Every allocation goes
// through MustBuf, which records the id; ReleaseAll frees them (see Resident.Close).
type Device struct {
	id     objc.ID
	mu     sync.Mutex
	allocs []objc.ID // every MTLBuffer handed out by this Device, for ReleaseAll
	objs   []objc.ID // non-buffer +1-owned objc objects (command queue, pipelines, libraries) — M24
}

// TrackObj records a +1-owned objc object (command queue / pipeline / library) so ReleaseObjects
// frees it at Close. MTLFunctions are NOT tracked — they are released immediately once their
// pipeline state exists (a pipeline does not retain its function). See M24(b).
func (d *Device) TrackObj(id objc.ID) {
	if id == 0 {
		return
	}
	d.mu.Lock()
	d.objs = append(d.objs, id)
	d.mu.Unlock()
}

// ReleaseObjects releases the non-buffer objc objects (command queue, pipelines, libraries) this
// Device tracked, then the MTLDevice itself, and empties the ledgers. purego has no ARC, so these
// leak per load/unload otherwise (M24(b): ReleaseAll freed buffers only). Idempotent: it nils the
// device id, so a second call (or a double Close) is a no-op. Caller MUST ensure no GPU work is in
// flight (Resident.Close stops+waits the executor first). MTLCreateSystemDefaultDevice hands back a
// +1 retain PER CALL on the (shared) system device, so releasing here balances THIS Device's retain
// and leaves any other resident model's device retain intact.
func (d *Device) ReleaseObjects() {
	d.mu.Lock()
	objs := d.objs
	d.objs = nil
	id := d.id
	d.id = 0
	d.mu.Unlock()
	for _, o := range objs {
		o.Send(selRelease)
	}
	if id != 0 {
		id.Send(selRelease)
	}
}

// CreateSystemDefaultDevice reaches Metal cgo-free (MTLCreateSystemDefaultDevice is a
// plain C export in Metal.framework).
func CreateSystemDefaultDevice() (*Device, error) {
	loadMetal()
	p := mtlCreateSystemDefaultDevice()
	if p == 0 {
		return nil, fmt.Errorf("metal: MTLCreateSystemDefaultDevice returned nil (no Metal GPU?)")
	}
	return &Device{id: objc.ID(p)}, nil
}

// Name is the device's product name (e.g. "Apple M1 Pro").
func (d *Device) Name() string { return goString(d.id.Send(selName)) }

// CompileLibrary compiles MSL `src` at languageVersion `ver` — with the landmine
// defused: an explicit MTLCompileOptions, plus a read-back assertion that the option
// took (loud, not silent). Returns the MTLLibrary id.
func (d *Device) CompileLibrary(src string, ver uint) (objc.ID, error) {
	opts := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	defer opts.Send(selRelease)
	opts.Send(selSetLanguageVersion, ver)
	if got := uint(objc.Send[uintptr](opts, selLanguageVersion)); got != ver {
		return 0, fmt.Errorf("metal: languageVersion set to %#x but reads %#x — the LC_BUILD_VERSION landmine (golang/go#77917)", ver, got)
	}

	var nsErr objc.ID
	lib := d.id.Send(selNewLibrarySource, nsString(src), opts, unsafe.Pointer(&nsErr))
	if lib == 0 {
		return 0, fmt.Errorf("metal: newLibraryWithSource failed: %s", goString(nsErr.Send(selLocalizedDesc)))
	}
	d.TrackObj(lib) // +1-owned; released at Close (M24)
	return lib, nil
}

// Fast-math / math-mode selectors and the MTLMathMode raw values (probed empirically
// on macOS 26.5.2: MTLMathModeSafe=0, Relaxed=1, Fast=2, and the default is Fast).
// Metal 3.2 / macOS 15 introduced setMathMode: and DEPRECATED setFastMathEnabled:; we
// prefer the former where it exists and fall back to the latter on older OSes, and
// read the result back either way (see setPreciseMath).
var (
	selRespondsToSel = objc.RegisterName("respondsToSelector:")
	selSetMathMode   = objc.RegisterName("setMathMode:")        // Metal 3.2+
	selMathMode      = objc.RegisterName("mathMode")            // Metal 3.2+
	selSetFastMath   = objc.RegisterName("setFastMathEnabled:") // deprecated in Metal 3.2
	selFastMath      = objc.RegisterName("fastMathEnabled")     // getter is fastMathEnabled, NOT isFastMathEnabled
)

const mtlMathModeSafe uintptr = 0 // no fast-math: true divides, no f32 reassoc/contract

// respondsTo reports whether obj implements sel. BOOL is a signed char returned in the
// low byte of the result register, so mask before testing.
func respondsTo(obj objc.ID, sel objc.SEL) bool {
	return objc.Send[uintptr](obj, selRespondsToSel, uintptr(sel))&0xff != 0
}

// respondsToFn is the seam respondsTo goes through, so a test can force the
// "no verifiable API" path without needing a future OS. Production is respondsTo.
var respondsToFn = respondsTo

// setPreciseMath disables fast-math on opts and VERIFIES the demand took, mirroring the
// languageVersion read-back guard. Without the read-back a future OS could silently
// no-op the setter — setFastMathEnabled: is already deprecated (→ MTLMathMode in Metal
// 3.2 / macOS 15) — and hand back a fast-math library while claiming to be precise,
// breaking the ViT parity gate (its exact maxAbs/127 quant scale needs true divides)
// with no error and no red test. Prefers the non-deprecated MTLMathModeSafe; falls back
// to setFastMathEnabled:NO on older OSes.
//
// It requires that BOTH the setter AND the getter of the chosen path respond before it
// touches either — so a set is never issued through a selector we cannot read back —
// and if NEITHER pair is verifiable it returns an error naming both. "I could not
// verify" must fail loudly, not fall through to an unchecked "precise" library: that
// silent pass is the exact failure this guard exists to prevent.
func setPreciseMath(opts objc.ID) error {
	switch {
	case respondsToFn(opts, selSetMathMode) && respondsToFn(opts, selMathMode):
		opts.Send(selSetMathMode, mtlMathModeSafe)
		if got := objc.Send[uintptr](opts, selMathMode); got != mtlMathModeSafe {
			return fmt.Errorf("metal: setMathMode:MTLMathModeSafe did not take (mathMode reads %d, want %d) — the 'precise' library would be silently fast-math", got, mtlMathModeSafe)
		}
		return nil
	case respondsToFn(opts, selSetFastMath) && respondsToFn(opts, selFastMath):
		opts.Send(selSetFastMath, uintptr(0)) // fastMathEnabled = NO
		if got := objc.Send[uintptr](opts, selFastMath) & 0xff; got != 0 {
			return fmt.Errorf("metal: setFastMathEnabled:NO did not take (fastMathEnabled reads YES) — likely deprecated on this OS (→ MTLMathMode); the 'precise' library is silently fast-math")
		}
		return nil
	default:
		return fmt.Errorf("metal: cannot verify precise math — MTLCompileOptions responds to neither setMathMode:/mathMode (Metal 3.2+) nor setFastMathEnabled:/fastMathEnabled (deprecated); refusing to return an unverified 'precise' library")
	}
}

// CompileLibraryPrecise compiles MSL with fast-math DISABLED: divides are true
// divides (not reciprocal approximations) and the compiler won't reassociate or
// contract f32 — so a computation matches the CPU/CUDA reference where f32 can. The
// ViT kernels use this because their parity gate needs an exact per-row quant scale
// (maxAbs/127, which fast-math turns into maxAbs*rcp(127), off by a ULP). CompileLibrary
// keeps the default (fast-math on), which is what goinfer's tuned decode kernels want.
//
// Both the languageVersion and the fast-math demand are read back and asserted, so a
// silent no-op (the LC_BUILD_VERSION landmine, or a deprecated fast-math setter on a
// future OS) fails loudly here instead of producing a wrong-numerics library.
func (d *Device) CompileLibraryPrecise(src string, ver uint) (objc.ID, error) {
	opts := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	defer opts.Send(selRelease)
	opts.Send(selSetLanguageVersion, ver)
	if got := uint(objc.Send[uintptr](opts, selLanguageVersion)); got != ver {
		return 0, fmt.Errorf("metal: languageVersion set to %#x but reads %#x — the LC_BUILD_VERSION landmine (golang/go#77917)", ver, got)
	}
	if err := setPreciseMath(opts); err != nil {
		return 0, err
	}
	var nsErr objc.ID
	lib := d.id.Send(selNewLibrarySource, nsString(src), opts, unsafe.Pointer(&nsErr))
	if lib == 0 {
		return 0, fmt.Errorf("metal: newLibraryWithSource (precise) failed: %s", goString(nsErr.Send(selLocalizedDesc)))
	}
	d.TrackObj(lib)
	return lib, nil
}

// ---- compute Dispatch (Layer A phase-1 completion: queue → buffers → encode → run) ----

var (
	selNewCommandQueue = objc.RegisterName("newCommandQueue")
	selNewFunctionName = objc.RegisterName("newFunctionWithName:")
	selNewPipelineFn   = objc.RegisterName("newComputePipelineStateWithFunction:error:")
	selNewBufferBytes  = objc.RegisterName("newBufferWithBytes:length:options:")
	selNewBufferNoCopy = objc.RegisterName("newBufferWithBytesNoCopy:length:options:deallocator:")
	selNewBufferLen    = objc.RegisterName("newBufferWithLength:options:")
	selContents        = objc.RegisterName("contents")
	selCommandBuffer   = objc.RegisterName("commandBuffer")
	selComputeEncoder  = objc.RegisterName("computeCommandEncoder")
	selSetPipeline     = objc.RegisterName("setComputePipelineState:")
	selSetBuffer       = objc.RegisterName("setBuffer:offset:atIndex:")
	selSetBuffers      = objc.RegisterName("setBuffers:offsets:withRange:")
	selSetTgMem        = objc.RegisterName("setThreadgroupMemoryLength:atIndex:")
	selDispatchThreads = objc.RegisterName("dispatchThreads:threadsPerThreadgroup:")
	selDispatchTG      = objc.RegisterName("dispatchThreadgroups:threadsPerThreadgroup:")
	selEndEncoding     = objc.RegisterName("endEncoding")
	selCommit          = objc.RegisterName("commit")
	selWaitCompleted   = objc.RegisterName("waitUntilCompleted")
	selDrain           = objc.RegisterName("drain")
	selGPUStartTime    = objc.RegisterName("GPUStartTime")    // CFTimeInterval (double), post-completion
	selGPUEndTime      = objc.RegisterName("GPUEndTime")      // GPU busy window for this command buffer
	selKernelStartTime = objc.RegisterName("kernelStartTime") // incl scheduling; > GPU window
	selKernelEndTime   = objc.RegisterName("kernelEndTime")

	selNewSharedEvent     = objc.RegisterName("newSharedEvent")            // MTLDevice → id<MTLSharedEvent>
	selSignaledValue      = objc.RegisterName("signaledValue")             // MTLSharedEvent → uint64 (CPU read)
	selSetSignaledValue   = objc.RegisterName("setSignaledValue:")         // MTLSharedEvent uint64 (CPU signal)
	selEncodeSignalEvent  = objc.RegisterName("encodeSignalEvent:value:")  // MTLCommandBuffer: GPU sets value here
	selEncodeWaitForEvent = objc.RegisterName("encodeWaitForEvent:value:") // MTLCommandBuffer: GPU waits for value
)

// mtlSize mirrors MTLSize {NSUInteger width,height,depth}. At 24 bytes (>16) it is
// passed to objc_msgSend BY REFERENCE per AAPCS64 — so the call site passes
// unsafe.Pointer(&sz), which lands the pointer in the arg register exactly as the ABI
// wants. (Getting this wrong is the "MTLSize struct-by-value" hazard the doc flagged.)
type mtlSize struct{ w, h, d uint64 }

// Queue / Pipeline / Buffer are thin id wrappers.
type Queue struct{ id objc.ID }
type Pipeline struct{ id objc.ID }
type Buffer struct {
	id  objc.ID
	n   int     // element count (float32)
	off uintptr // byte offset for binding (a sub-view into a larger buffer; 0 = whole)
}

// At returns a view of the buffer bound at byteOff — for reading a slice of a combined
// buffer (e.g. q/k/v out of a fused QKV output). Zero-copy; only the bind offset changes.
func (b Buffer) At(byteOff int) Buffer { b.off = uintptr(byteOff); return b }

// Len reports the buffer's element count (float32s, or bytes for an int8 buffer) —
// mirrors cuda.go's Buffer.Len so a consumer can guard on an empty/optional buffer.
func (b Buffer) Len() int { return b.n }

// NewCommandQueue creates the command queue (built once, reused per token in the real backend).
func (d *Device) NewCommandQueue() Queue {
	q := d.id.Send(selNewCommandQueue)
	d.TrackObj(q) // +1-owned; released at Close (M24)
	return Queue{id: q}
}

// NewComputePipeline looks a kernel up in a compiled library and builds its pipeline state.
func (d *Device) NewComputePipeline(lib objc.ID, fn string) (Pipeline, error) {
	f := lib.Send(selNewFunctionName, nsString(fn))
	if f == 0 {
		return Pipeline{}, fmt.Errorf("metal: no kernel %q in library", fn)
	}
	var nsErr objc.ID
	p := d.id.Send(selNewPipelineFn, f, unsafe.Pointer(&nsErr))
	f.Send(selRelease) // the pipeline state does not retain its MTLFunction — release it now (M24b)
	if p == 0 {
		return Pipeline{}, fmt.Errorf("metal: pipeline %q: %s", fn, goString(nsErr.Send(selLocalizedDesc)))
	}
	d.TrackObj(p) // +1-owned; released at Close (M24)
	return Pipeline{id: p}, nil
}

// NewBufferFloats uploads data into a shared (UMA host-visible) MTLBuffer.
// MustBuf turns a FAILED MTLBuffer allocation into a loud panic instead of a silently
// zero-filled Buffer, and records the id so ReleaseAll can free it. objc returns nil on OOM and
// every constructor here used to keep it, so an out-of-memory condition surfaced as garbage
// numerics — an OOM wearing a parity bug's clothes. BuildResident recovers panics and declines
// to the CPU path, which is the right answer for OOM.
func (d *Device) MustBuf(id objc.ID, n int, what string) Buffer {
	if id == 0 {
		panic(fmt.Sprintf("metal: MTLBuffer allocation failed (%s, %d bytes) — out of memory", what, n))
	}
	d.mu.Lock()
	d.allocs = append(d.allocs, id)
	d.mu.Unlock()
	return Buffer{id: id, n: n}
}

// ReleaseAll releases every MTLBuffer this Device handed out and empties the ledger. Callers MUST
// ensure no GPU work is still in flight against them (Resident.Close stops the executor and waits
// first) — releasing a buffer a live command buffer references is a use-after-free.
func (d *Device) ReleaseAll() {
	d.mu.Lock()
	ids := d.allocs
	d.allocs = nil
	d.mu.Unlock()
	for _, id := range ids {
		id.Send(selRelease)
	}
}

// LedgerLen reports how many MTLBuffers and non-buffer objc objects this Device still owns —
// the observability hook for leak tests (Close/ReleaseAll must drive both to 0). Held under the
// same mutex the ledger mutations use.
func (d *Device) LedgerLen() (allocs, objs int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.allocs), len(d.objs)
}

// ReleaseBuf releases ONE MTLBuffer and removes it from the ledger, so a later ReleaseAll won't
// double-free it (C5). For per-call scratch that must not accumulate on the ledger until Close —
// the caller MUST ensure no in-flight GPU work references it (PrefillLast commits+waits before its
// deferred releases run). O(n) swap-remove scan of the ledger: fine for coarse per-request scratch,
// never a per-token path. A zero id (empty Buffer) is a no-op.
func (d *Device) ReleaseBuf(b Buffer) {
	if b.id == 0 {
		return
	}
	d.mu.Lock()
	for i, id := range d.allocs {
		if id == b.id {
			d.allocs[i] = d.allocs[len(d.allocs)-1]
			d.allocs = d.allocs[:len(d.allocs)-1]
			break
		}
	}
	d.mu.Unlock()
	b.id.Send(selRelease)
}

func (d *Device) NewBufferFloats(data []float32) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&data[0]), uintptr(len(data)*4), uintptr(0))
	runtime.KeepAlive(data)
	return d.MustBuf(id, len(data), "floats")
}

// NewBufferLen allocates an uninitialized shared MTLBuffer of nFloats float32s.
func (d *Device) NewBufferLen(nFloats int) Buffer {
	return d.MustBuf(d.id.Send(selNewBufferLen, uintptr(nFloats*4), uintptr(0)), nFloats, "len")
}

// NewBufferBytes allocates an uninitialized shared MTLBuffer of n BYTES (n is the
// element count for the returned Buffer). For consumers that size a buffer in raw
// bytes rather than float32s — the device-layer primitive goinfer's kernels reach
// for when re-pointed onto this substrate.
func (d *Device) NewBufferBytes(n int) Buffer {
	return d.MustBuf(d.id.Send(selNewBufferLen, uintptr(n), uintptr(0)), n, "bytes")
}

// NewBufferInt8 uploads int8 data (n counts BYTES for this buffer). NewBufferU32
// uploads a single uint32 (kernel scalar arg). Both are shared/UMA.
func (d *Device) NewBufferInt8(data []int8) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&data[0]), uintptr(len(data)), uintptr(0))
	runtime.KeepAlive(data)
	return d.MustBuf(id, len(data), "int8")
}

// NewBufferNoCopy wraps caller-owned memory in a shared MTLBuffer WITHOUT copying
// (newBufferWithBytesNoCopy) — the resident-weights lever: alias an mmap'd .giw straight into
// Metal instead of uploading a second GB-scale copy. ptr MUST be page-aligned and nBytes SHOULD be
// a page multiple (Metal drops a trailing partial page); an mmap base satisfies both. The
// deallocator is nil — Metal never frees this memory; the caller owns the mapping and munmaps it
// (AND must keep it mapped for the buffer's whole life). n counts BYTES, like NewBufferBytes. Bind
// per-tensor sub-views with Buffer.At(byteOffset) over the single whole-mapping buffer.
func (d *Device) NewBufferNoCopy(ptr unsafe.Pointer, nBytes int) Buffer {
	id := d.id.Send(selNewBufferNoCopy, ptr, uintptr(nBytes), uintptr(0), uintptr(0))
	return d.MustBuf(id, nBytes, "nocopy")
}

func (d *Device) NewBufferU32(v uint32) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&v), uintptr(4), uintptr(0))
	runtime.KeepAlive(&v)
	return d.MustBuf(id, 1, "u32")
}

// NewBufferUint32s uploads packed u32 data (n counts u32 words). Shared/UMA.
func (d *Device) NewBufferUint32s(data []uint32) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&data[0]), uintptr(len(data)*4), uintptr(0))
	runtime.KeepAlive(data)
	return d.MustBuf(id, len(data), "uint32s")
}

// NewBufferU16s uploads u16 data (e.g. f16 group scales; n counts u16s). Shared/UMA.
func (d *Device) NewBufferU16s(data []uint16) Buffer {
	if len(data) == 0 {
		return d.NewBufferLen(1)
	}
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&data[0]), uintptr(len(data)*2), uintptr(0))
	runtime.KeepAlive(data)
	return d.MustBuf(id, len(data), "u16s")
}

// Floats / Int8s view the buffer's shared contents as a Go slice (zero-copy on UMA).
func (b Buffer) Floats() []float32 {
	return unsafe.Slice(objc.Send[*float32](b.id, selContents), b.n)
}

func (b Buffer) Int8s() []int8 {
	return unsafe.Slice(objc.Send[*int8](b.id, selContents), b.n)
}

// U16s views the buffer's shared contents as []uint16 (f16 bits) — zero-copy on UMA.
func (b Buffer) U16s() []uint16 {
	return unsafe.Slice(objc.Send[*uint16](b.id, selContents), b.n)
}

// U32s views the buffer's shared contents as a []uint32 (zero-copy on UMA) — e.g. the MoE
// router's idx[k] output.
func (b Buffer) U32s() []uint32 {
	return unsafe.Slice(objc.Send[*uint32](b.id, selContents), b.n)
}

// SetU32 overwrites a 1-word uniform buffer's contents in place (per-token: rope pos,
// nKeys). Zero-copy on UMA.
func (b Buffer) SetU32(v uint32) { *objc.Send[*uint32](b.id, selContents) = v }

// U32 reads a 1-word buffer's first uint32 (per-token: the fused-argmax token id). Zero-copy.
func (b Buffer) U32() uint32 { return *objc.Send[*uint32](b.id, selContents) }

// Run1D encodes and runs a 1-D kernel over n threads (threadgroup width tg), binding
// bufs at indices 0..len-1, and blocks until the GPU finishes. Manual autoreleasepool
// discipline (no ARC): the per-token commandBuffer/encoder are autoreleased into the
// pool and drained here, so the decode loop won't leak (doc risk #2).
func (q Queue) Run1D(p Pipeline, n, tg int, bufs ...Buffer) {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)

	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
}

// Run2D encodes and runs a 2-D kernel over gx×gy THREADGROUPS of tgx×tgy threads each,
// via dispatchThreadgroups (UNIFORM whole threadgroups) — the shape a tiled GEMM needs.
// Unlike Run1D's dispatchThreads (non-uniform, exactly n threads, no bounds check),
// every thread in the final edge tiles must EXIST so the tile can cooperatively stage
// its A/B sub-tiles into threadgroup memory and all reach the barrier; the kernel then
// bounds-checks its own global (m,n,k) index, exactly like the CUDA tiled kernels.
// Blocks until the GPU finishes. Manual autoreleasepool discipline, like Run1D.
func (q Queue) Run2D(p Pipeline, gx, gy, tgx, tgy int, bufs ...Buffer) {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)

	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	grid := mtlSize{w: uint64(gx), h: uint64(gy), d: 1}
	perTG := mtlSize{w: uint64(tgx), h: uint64(tgy), d: 1}
	enc.Send(selDispatchTG, unsafe.Pointer(&grid), unsafe.Pointer(&perTG))
	runtime.KeepAlive(&grid)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
}

// Encoder batches many DIFFERENT kernel dispatches into ONE command buffer — the
// per-token shape the decode loop needs (whole layer stack → one commit/wait, per the
// tax finding). The default serial compute encoder inserts barriers between dependent
// dispatches (like WGSL's storage barriers), so chained kernels see prior writes.
type Encoder struct {
	pool objc.ID
	cb   objc.ID
	enc  objc.ID
	// Encode-tax trims (measured: together only ~0.5ms of the ~2.4ms/token overhead — the
	// rest is GPU-side per-Dispatch latency, not Go-side msgSend): skip setComputePipelineState
	// when the pipeline is unchanged, and batch all buffer binds into ONE setBuffers call.
	curPipe objc.ID
	idScr   [16]objc.ID // reusable C-arrays for batched setBuffers:offsets:withRange:
	offScr  [16]uintptr
	// GPU timing of the last committed command buffer (seconds), captured in end() after
	// waitUntilCompleted (valid post-completion) and before the pool drains the cb.
	gpuStart, gpuEnd, kernStart, kernEnd float64
}

func (q Queue) Begin() *Encoder {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	cb := q.id.Send(selCommandBuffer)
	return &Encoder{pool: pool, cb: cb, enc: cb.Send(selComputeEncoder)}
}

// ARPool is an NSAutoreleasePool handle — the pipelined executor owns one long-lived pool
// (drained periodically) so encode-ahead can hold an un-committed command buffer across the
// loop without per-call pool nesting.
type ARPool struct{ id objc.ID }

func NewARPool() ARPool {
	return ARPool{id: objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)}
}
func (p ARPool) Drain() { p.id.Send(selDrain) }

// BeginNP creates a command buffer + encoder with NO per-call pool — the cb/encoder autorelease
// into whatever pool is active on the calling thread. Used by the pipelined executor, which
// owns one long-lived pool drained periodically (so encode-ahead can hold an un-committed next
// command buffer without the LIFO pool-nesting a per-call pool would cause).
func (q Queue) BeginNP() *Encoder {
	cb := q.id.Send(selCommandBuffer)
	return &Encoder{cb: cb, enc: cb.Send(selComputeEncoder)}
}

// FinishEncoding closes the encoder (ready to commit) without committing — the split that lets
// the executor encode token t+1 while token t's command buffer runs.
func (e *Encoder) FinishEncoding() { e.enc.Send(selEndEncoding) }
func (e *Encoder) Commit()         { e.cb.Send(selCommit) }

// waitDone blocks until the committed command buffer completes, then reads its GPU timestamps.
func (e *Encoder) WaitDone() {
	e.cb.Send(selWaitCompleted)
	e.ReadTimes()
}

// ReadTimes reads the command buffer's GPU/kernel timestamps (valid post-completion).
func (e *Encoder) ReadTimes() {
	// On arm64 objc_msgSend returns the double in d0 — objc.Send[float64] uses the fp-return path.
	e.gpuStart = objc.Send[float64](e.cb, selGPUStartTime)
	e.gpuEnd = objc.Send[float64](e.cb, selGPUEndTime)
	e.kernStart = objc.Send[float64](e.cb, selKernelStartTime)
	e.kernEnd = objc.Send[float64](e.cb, selKernelEndTime)
}

// Dispatch encodes one kernel over n threads (threadgroup width tg, clamped ≤ n), binding
// bufs at indices 0..len-1. dispatchThreads launches EXACTLY n threads (non-uniform), so
// no out-of-range writes.
func (e *Encoder) Dispatch(p Pipeline, n, tg int, bufs ...Buffer) {
	if tg > n {
		tg = n
	}
	if p.id != e.curPipe {
		e.enc.Send(selSetPipeline, p.id)
		e.curPipe = p.id
	}
	// Batch ALL buffer binds into one setBuffers:offsets:withRange: msgSend (vs one per
	// buffer) — the aggressive Go-side-encode-cost probe. NSRange{location,length} lowers to
	// two trailing word args.
	nb := len(bufs)
	for i, b := range bufs {
		e.idScr[i], e.offScr[i] = b.id, b.off
	}
	e.enc.Send(selSetBuffers, unsafe.Pointer(&e.idScr[0]), unsafe.Pointer(&e.offScr[0]), uintptr(0), uintptr(nb))
	runtime.KeepAlive(&e.idScr)
	runtime.KeepAlive(&e.offScr)
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	e.enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
}

// DispatchTG is Dispatch with dynamic threadgroup memory of tgBytes at index 0 — for the Stage A
// GEMV kernels whose activation-staging scratch is sized to K per call (so a small-K model keeps
// full occupancy while a large-K one, e.g. Qwen3 o-proj K=nHhd>1536, gets the room it needs).
func (e *Encoder) DispatchTG(p Pipeline, n, tg, tgBytes int, bufs ...Buffer) {
	if tg > n {
		tg = n
	}
	if p.id != e.curPipe {
		e.enc.Send(selSetPipeline, p.id)
		e.curPipe = p.id
	}
	nb := len(bufs)
	for i, b := range bufs {
		e.idScr[i], e.offScr[i] = b.id, b.off
	}
	e.enc.Send(selSetBuffers, unsafe.Pointer(&e.idScr[0]), unsafe.Pointer(&e.offScr[0]), uintptr(0), uintptr(nb))
	e.enc.Send(selSetTgMem, uintptr(tgBytes), uintptr(0))
	runtime.KeepAlive(&e.idScr)
	runtime.KeepAlive(&e.offScr)
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	e.enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
}

func (e *Encoder) End() {
	e.enc.Send(selEndEncoding)
	e.cb.Send(selCommit)
	e.cb.Send(selWaitCompleted)
	e.ReadTimes()
	e.pool.Send(selDrain)
}

// SharedEvent is an MTLSharedEvent — a monotonic counter both the GPU (encodeSignalEvent /
// encodeWaitForEvent inside a command buffer) and the CPU (Value / SetValue) can read and advance.
// It exists to let the host observe per-layer progress WITHOUT tearing a token into one command
// buffer per layer: encode the whole trunk once, have the GPU signal after each layer's router and
// wait for a CPU ack before the expert dispatches, and the CPU stage the routed experts in between —
// one submit per token, with per-layer host intervention. This is the Metal-idiomatic alternative to
// per-layer submit+wait for the Gemma-4 MoE expert-paging path; the Step-6 Step-0 probe prices it.
type SharedEvent struct{ id objc.ID }

// NewSharedEvent creates a shared event (initial signaledValue 0).
func (d *Device) NewSharedEvent() SharedEvent { return SharedEvent{id: d.id.Send(selNewSharedEvent)} }

// Value reads the event's current signaledValue (CPU side). SetValue advances it (CPU signal to the
// GPU's encodeWaitForEvent). Both are the documented CPU⇄GPU shared-event handshake ops on UMA.
func (ev SharedEvent) Value() uint64     { return objc.Send[uint64](ev.id, selSignaledValue) }
func (ev SharedEvent) SetValue(v uint64) { ev.id.Send(selSetSignaledValue, uintptr(v)) }

// EventBoundary closes the current compute segment, has the GPU signal `sig` (so a spinning CPU sees
// this layer finished), then has the GPU wait for `wait` (blocking the next segment until the CPU
// acks), then opens a fresh compute segment. curPipe is reset so the next Dispatch re-binds its
// pipeline. Called between encodeLayer calls to build the one-command-buffer handshake trunk.
func (e *Encoder) EventBoundary(ev SharedEvent, sig, wait uint64) {
	e.enc.Send(selEndEncoding)
	e.cb.Send(selEncodeSignalEvent, ev.id, uintptr(sig))
	e.cb.Send(selEncodeWaitForEvent, ev.id, uintptr(wait))
	e.enc = e.cb.Send(selComputeEncoder)
	e.curPipe = 0
}

// DrainPool drains the Encoder's autorelease pool — the tail of a hand-built command buffer (e.g.
// the shared-event handshake) that uses FinishEncoding + Commit + WaitDone directly instead of End()
// because the caller runs CPU handshake work between Commit and WaitDone.
func (e *Encoder) DrainPool() { e.pool.Send(selDrain) }

// Run1DBatch encodes `reps` dispatches of the same kernel into ONE command buffer and
// submits once — the shape a real token uses (a whole layer stack encoded into one
// command buffer, one commit/wait). Isolates the per-commit round-trip tax (reps=1)
// from the marginal per-encoded-Dispatch msgSend cost (the slope over reps).
func (q Queue) Run1DBatch(p Pipeline, n, tg, reps int, bufs ...Buffer) {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)

	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	for range reps {
		enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	}
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
}

// Run1DBatchTG is Run1DBatch with dynamic threadgroup memory of tgBytes at index 0 (for kernels
// with a `threadgroup T* x [[threadgroup(0)]]` param sized per-Dispatch — batch-k stages only
// what k needs, preserving occupancy).
func (q Queue) Run1DBatchTG(p Pipeline, n, tg, reps, tgBytes int, bufs ...Buffer) {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)
	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	enc.Send(selSetTgMem, uintptr(tgBytes), uintptr(0))
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	for range reps {
		enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	}
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
}

// Run1DTG runs ONE 1-D dispatch over n threads (threadgroup width tg) with tgBytes of
// dynamic threadgroup memory bound at index 0 — Run1D plus a `threadgroup T*
// [[threadgroup(0)]]` param (the ViT attention kernel stages its per-query score row
// there). It is Run1DBatchTG with reps=1, the shape the encoder kernels want.
func (q Queue) Run1DTG(p Pipeline, n, tg, tgBytes int, bufs ...Buffer) {
	q.Run1DBatchTG(p, n, tg, 1, tgBytes, bufs...)
}

// GPUStart/GPUEnd/KernStart/KernEnd expose the last committed command buffer's GPU/kernel
// timestamps (seconds), valid after WaitDone/End. Accessors, since the fields are unexported.
func (e *Encoder) GPUStart() float64  { return e.gpuStart }
func (e *Encoder) GPUEnd() float64    { return e.gpuEnd }
func (e *Encoder) KernStart() float64 { return e.kernStart }
func (e *Encoder) KernEnd() float64   { return e.kernEnd }

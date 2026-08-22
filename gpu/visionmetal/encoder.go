//go:build darwin

// Package visionmetal is the native-Metal vision.ResidentEncoder — a whole SigLIP
// tower running on Apple GPUs, cgo-free (docs/task-native-gpu.md, Phase 3). It is the
// Apple counterpart of gpu/visioncuda, and the vision analogue of what gpu/annmetal is
// for ANN: aikit's Metal device substrate plus one real consumer.
//
// Importing this package is the opt-in. aikit's core `vision` never imports `gpu`, so
// the default build stays pure-Go CPU; this package's init() plugs the factory into
// vision.RegisterResident, and Encoder.EnableResident then routes Forward here.
//
// The forward mirrors gpu/visioncuda/encoder.go op for op (same order, same
// formulations) — the ViT kernels (gpu/metal_vit.go) carry the arithmetic-matching
// detail. The device-level differences from the CUDA path are all here: Apple UMA
// means a buffer's Floats() is a zero-copy live view (no explicit upload/download);
// scalars bind as 1-element buffers (SetU32 / a written float view); and the whole
// forward runs under one runtime.LockOSThread, because each dispatch's Run1D allocates
// and drains an NSAutoreleasePool that objc requires on a single thread.
package visionmetal

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/vision"
)

func init() {
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		w, err := e.GPUWeights()
		if err != nil {
			return nil, err
		}
		return newEncoder(w)
	})
}

// mat is one resident W8A8 weight: int8 rows [Rows,Cols] plus per-row scales.
type mat struct {
	q, s       gpu.Buffer
	rows, cols int
}

// layer is one transformer block's resident weights.
type layer struct {
	ln1w, ln1b, ln2w, ln2b gpu.Buffer
	qb, kb, vb, ob         gpu.Buffer
	fc1b, fc2b             gpu.Buffer
	qw, kw, vw, ow         mat
	fc1w, fc2w             mat
}

// encoder is the device-resident SigLIP tower.
type encoder struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	w   vision.GPUWeights

	patchW, patchB, posEmb gpu.Buffer
	postLNw, postLNb       gpu.Buffer
	layers                 []layer

	// Per-forward scratch, allocated once and reused across layers/forwards.
	patches, h, n1, n2 gpu.Buffer
	qa, ka, va, att, o gpu.Buffer
	mid, mlp, out      gpu.Buffer
	qi8, qs            gpu.Buffer // quantized activation + per-row scale

	// Reusable scalar buffers (Metal has no by-value kernel arg): three uint32 slots
	// and two float slots, rewritten before each dispatch. Safe to reuse because each
	// Run1D commits and waits, so a scalar is fully consumed before it is overwritten.
	s0, s1, s2       gpu.Buffer
	epsBuf, scaleBuf gpu.Buffer

	cpp int
	mu  sync.Mutex
}

// newEncoder uploads the tower and allocates scratch. A device OOM surfaces from the
// gpu layer as a panic (MustBuf's loud-failure contract); recover it into an error so
// EnableResident declines and the caller keeps the CPU path.
func newEncoder(w vision.GPUWeights) (enc *encoder, err error) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	defer func() {
		if r := recover(); r != nil {
			dev.ReleaseAll()
			dev.ReleaseObjects()
			enc, err = nil, fmt.Errorf("visionmetal: device upload failed: %v", r)
		}
	}()
	kern, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	np, hidden, inter := w.NumPatches, w.Hidden, w.Inter
	if np <= 0 || hidden <= 0 || inter <= 0 {
		dev.ReleaseObjects()
		return nil, fmt.Errorf("visionmetal: degenerate tower (np=%d hidden=%d inter=%d)", np, hidden, inter)
	}
	cpp := w.NumChannels * w.PatchSize * w.PatchSize

	e := &encoder{dev: dev, q: dev.NewCommandQueue(), k: kern, w: w, cpp: cpp}
	up := func(x []float32) gpu.Buffer { return gpu.NewBufferOf(dev, x) }
	upMat := func(m vision.GPUMat) mat {
		return mat{q: gpu.NewBufferOf(dev, m.Q), s: gpu.NewBufferOf(dev, m.Scales), rows: m.Rows, cols: m.Cols}
	}
	e.patchW, e.patchB, e.posEmb = up(w.PatchW), up(w.PatchB), up(w.PosEmb)
	e.postLNw, e.postLNb = up(w.PostLNw), up(w.PostLNb)
	e.layers = make([]layer, len(w.Layers))
	for i, L := range w.Layers {
		e.layers[i] = layer{
			ln1w: up(L.LN1w), ln1b: up(L.LN1b), ln2w: up(L.LN2w), ln2b: up(L.LN2b),
			qb: up(L.Qb), kb: up(L.Kb), vb: up(L.Vb), ob: up(L.Ob),
			fc1b: up(L.FC1b), fc2b: up(L.FC2b),
			qw: upMat(L.Qw), kw: upMat(L.Kw), vw: upMat(L.Vw), ow: upMat(L.Ow),
			fc1w: upMat(L.FC1w), fc2w: upMat(L.FC2w),
		}
	}
	f32 := func(n int) gpu.Buffer { return dev.NewBufferLen(n) }
	e.patches = f32(np * cpp)
	e.h, e.n1, e.n2 = f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.qa, e.ka, e.va = f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.att, e.o, e.mlp, e.out = f32(np*hidden), f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.mid = f32(np * inter)
	wide := max(inter, hidden)
	e.qi8 = gpu.NewBufferOf(dev, make([]int8, np*wide))
	e.qs = f32(np)
	e.s0, e.s1, e.s2 = gpu.NewBufferOf(dev, []uint32{0}), gpu.NewBufferOf(dev, []uint32{0}), gpu.NewBufferOf(dev, []uint32{0})
	e.epsBuf, e.scaleBuf = gpu.NewBufferOf(dev, []float32{0}), gpu.NewBufferOf(dev, []float32{0})
	return e, nil
}

func vitTG(n int) int {
	if n < gpu.ViTBlock {
		return n
	}
	return gpu.ViTBlock
}

func (e *encoder) setI(b gpu.Buffer, v int) { b.SetU32(uint32(int32(v))) }

// The dispatch helpers below are all called under ForwardPatches's LockOSThread. Each
// is one Run1D (or Run1DTG) — the same op sequence as the CUDA launches, with the
// scalar buffers rewritten per call.

func (e *encoder) gemmF32(a, b, c gpu.Buffer, M, N, K int) {
	e.setI(e.s0, M)
	e.setI(e.s1, N)
	e.setI(e.s2, K)
	p, gx, gy, tgx, tgy := e.k.GEMMF32Plan(M, N, K) // aligned → sg_big, else sg
	e.q.Run2D(p, gx, gy, tgx, tgy, a, b, c, e.s0, e.s1, e.s2)
}

func (e *encoder) addBias(x, bias gpu.Buffer, rows, dim int) {
	e.setI(e.s0, rows)
	e.setI(e.s1, dim)
	e.q.Run1D(e.k.AddBias, rows*dim, vitTG(rows*dim), x, bias, e.s0, e.s1)
}

func (e *encoder) addVec(x, v gpu.Buffer, n int) {
	e.setI(e.s0, n)
	e.q.Run1D(e.k.AddVec, n, vitTG(n), x, v, e.s0)
}

func (e *encoder) norm(x, w, b, out gpu.Buffer, rows, dim int) {
	e.setI(e.s0, rows)
	e.setI(e.s1, dim)
	e.epsBuf.Floats()[0] = e.w.Eps
	e.q.Run1D(e.k.LayerNorm, rows*gpu.ViTBlock, gpu.ViTBlock, x, w, b, out, e.s0, e.s1, e.epsBuf)
}

func (e *encoder) gelu(x gpu.Buffer, n int) {
	e.setI(e.s0, n)
	e.q.Run1D(e.k.GELUTanh, n, vitTG(n), x, e.s0)
}

func (e *encoder) attn(q, k, v, out gpu.Buffer, np, nH, hd int, scale float32) {
	e.setI(e.s0, np)
	e.setI(e.s1, nH)
	e.setI(e.s2, hd)
	e.scaleBuf.Floats()[0] = scale
	e.q.Run1DTG(e.k.Attention, np*nH*gpu.ViTBlock, gpu.ViTBlock, np*4,
		q, k, v, out, e.s0, e.s1, e.s2, e.scaleBuf)
}

// proj runs one int8 projection: quantize(src[M,K]) → W8A8 matmul against m[N,K] →
// += bias[N], into dst[M,N]. The quantized activation reuses the shared scratch, so
// projections must not interleave — they don't, the forward is sequential.
func (e *encoder) proj(src gpu.Buffer, m mat, bias, dst gpu.Buffer, M int) {
	K, N := m.cols, m.rows
	e.setI(e.s0, M)
	e.setI(e.s1, K)
	e.q.Run1D(e.k.QuantRows, M*gpu.ViTBlock, gpu.ViTBlock, src, e.qi8, e.qs, e.s0, e.s1)
	e.setI(e.s0, M)
	e.setI(e.s1, N)
	e.setI(e.s2, K)
	gx, gy, tgx, tgy := gpu.TileDims(M, N)
	e.q.Run2D(e.k.GEMMW8A8Tiled, gx, gy, tgx, tgy, e.qi8, e.qs, m.q, m.s, dst, e.s0, e.s1, e.s2)
	e.setI(e.s0, M)
	e.setI(e.s1, N)
	e.q.Run1D(e.k.AddBias, M*N, vitTG(M*N), dst, bias, e.s0, e.s1)
}

// ForwardPatches runs the resident SigLIP forward on im2col patches [np*(C*P*P)]
// (vision.Encoder.GridPatches) and returns last_hidden_state [np*hidden] — op for op
// the same sequence as vision/encoder.go's Forward and gpu/visioncuda's.
func (e *encoder) ForwardPatches(patches []float32) ([]float32, error) {
	np, hidden, inter := e.w.NumPatches, e.w.Hidden, e.w.Inter
	nH, hd := e.w.NumHeads, e.w.HeadDim
	if len(patches) != np*e.cpp {
		return nil, fmt.Errorf("visionmetal: patches len %d, want %d (np=%d cpp=%d)", len(patches), np*e.cpp, np, e.cpp)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	// One OS-thread pin for the whole forward: every Run1D allocs+drains an
	// NSAutoreleasePool, which objc requires on one thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	copy(e.patches.Floats(), patches) // UMA upload (zero-copy view)
	// patch embed: h = patches[np,cpp]·patchW[hidden,cpp]ᵀ + patchB + posEmb
	e.gemmF32(e.patches, e.patchW, e.h, np, hidden, e.cpp)
	e.addBias(e.h, e.patchB, np, hidden)
	e.addVec(e.h, e.posEmb, np*hidden)

	scale := float32(1.0 / math.Sqrt(float64(hd)))
	for i := range e.layers {
		L := &e.layers[i]
		// attention block (pre-LN, residual)
		e.norm(e.h, L.ln1w, L.ln1b, e.n1, np, hidden)
		e.proj(e.n1, L.qw, L.qb, e.qa, np)
		e.proj(e.n1, L.kw, L.kb, e.ka, np)
		e.proj(e.n1, L.vw, L.vb, e.va, np)
		e.attn(e.qa, e.ka, e.va, e.att, np, nH, hd, scale)
		e.proj(e.att, L.ow, L.ob, e.o, np)
		e.addVec(e.h, e.o, np*hidden)
		// MLP block (pre-LN, residual): fc2(geluTanh(fc1(x)))
		e.norm(e.h, L.ln2w, L.ln2b, e.n2, np, hidden)
		e.proj(e.n2, L.fc1w, L.fc1b, e.mid, np)
		e.gelu(e.mid, np*inter)
		e.proj(e.mid, L.fc2w, L.fc2b, e.mlp, np)
		e.addVec(e.h, e.mlp, np*hidden)
	}
	e.norm(e.h, e.postLNw, e.postLNb, e.out, np, hidden)

	dst := make([]float32, np*hidden)
	copy(dst, e.out.Floats()) // UMA download (the last Run1D already waited)
	return dst, nil
}

// Close releases the device buffers and objects (buffers first, so no in-flight work
// references freed memory; the forward's last Run1D already waited).
func (e *encoder) Close() {
	e.dev.ReleaseAll()
	e.dev.ReleaseObjects()
}

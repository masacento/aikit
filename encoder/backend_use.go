package encoder

// backend_use.go is the OPT-IN half of the Backend seam (docs/task-native-gpu.md,
// Phase 4).
//
// Backend, NewBackend and RegisterBackend have existed for a while, and
// goinfer/gpu registers "webgpu" on init — but until this wiring landed nothing in
// the forward ever called Backend.MatmulBT. The hot path went straight to the pure-Go
// matmul, so every registered backend was inert: you could ask for one, get one, and
// have it do nothing. The dispatch point is now scratch.mm, and these setters are how
// a caller attaches one.
//
// Everything here is additive. Not setting a backend leaves the forward calling
// exactly the function it called before, so the pure-Go numerics are unchanged by
// construction rather than by argument (backend_wiring_test.go gates that).
//
// A backend sees EVERY f32 matmul in a forward, including the small per-head QKᵀ and
// scores·V. Choosing which shapes are worth a device round-trip belongs to the
// backend, not here — a backend that offloads unconditionally will lose on short
// sequences, where the pure-Go path already switches strategy at ~4 MFLOP.
//
// Scope note: this routes the **f32** path, which is what Backend's single
// MatmulBT(a, b, dst, M, K, N) can express. The int8 (LoadQ8 / WeightsQ8) projections
// go through matmulBTQ8Into, whose int8 weights and per-row scales do not fit that
// signature, and they stay on the CPU. Accelerating those needs a wider Backend — a
// deliberate, separate decision, since Backend is Hard-tier surface.

// UseBackend attaches a compute backend to this model's forward. Passing nil restores
// the pure-Go path. The model does not take ownership: the caller closes the backend.
func (m *Model) UseBackend(b Backend) {
	if m != nil && m.weights != nil {
		m.weights.be = b
	}
}

// UseBackend attaches a compute backend to this int8 model's forward. Only the f32
// matmuls in the forward (attention QKᵀ and scores·V) route through it; the int8
// projections stay on the CPU — see the package note above.
func (m *ModelQ8) UseBackend(b Backend) {
	if m != nil && m.weights != nil {
		m.weights.be = b
	}
}

// UseBackend attaches a compute backend to this BERT model's forward.
func (b *BERT) UseBackend(be Backend) {
	if b != nil {
		b.be = be
	}
}

// UseBackend attaches a compute backend to this GTE model's forward.
func (g *GTE) UseBackend(be Backend) {
	if g != nil {
		g.be = be
	}
}

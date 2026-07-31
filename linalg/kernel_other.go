//go:build !arm64

package linalg

const has2x8Kernel = false

// HasFusedQ8Kernel — see the arm64 definition. False here: the fused packed path has no
// NEON kernel, so the encoder keeps the AVX2 dequant-then-f32-GEMM path off-arm64.
const HasFusedQ8Kernel = false

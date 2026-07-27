// smoke.cu — the device layer's own end-to-end proof (cuda_smoke_test.go): reach the
// GPU, JIT PTX, alloc buffers, upload, dispatch, read back. The CUDA twin of the MSL
// `vadd` in smoke_test.go.
//
// Note the bounds guard: unlike Metal's dispatchThreads, a CUDA launch rounds up to
// whole blocks, so the tail block runs past n. Every kernel on this device layer
// carries its own count and checks it (gpu/cuda.go, divergence 2).
extern "C" __global__ void vadd(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    const unsigned int* __restrict__ n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n) return;
    out[i] = a[i] + b[i];
}

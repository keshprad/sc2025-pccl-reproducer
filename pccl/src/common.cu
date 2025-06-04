#include "common.h"


// Kernel for vector addition.
template<typename T>
__global__ void vectorAddKernel(T* a, const T* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}
// Explicit instantiation declaration for fp32
template __global__ void vectorAddKernel<float>(float* a, const float* b, int n);
// Add a specialized kernel for bfloat16 addition
template<>
__global__ void vectorAddKernel<__nv_bfloat16>(__nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Convert to float32, add, then convert back to bfloat16
        float a_val = __bfloat162float(a[idx]);
        float b_val = __bfloat162float(b[idx]);
        a[idx] = __float2bfloat16(a_val + b_val);
    }
}

// Function to launch the kernel.
template<typename T>
void vectorAdd(T* a, const T* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    vectorAddKernel<<<blocks, threads, 0, stream>>>(a, b, n);
    CUDA_CHECK(cudaGetLastError());
}
// Explicit instantiation for fp32
template void vectorAdd<float>(float* a, const float* b, int n, cudaStream_t stream);
// Explicit instantiation for bf16
template void vectorAdd<__nv_bfloat16>(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream);


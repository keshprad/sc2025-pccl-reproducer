#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>      // Include this for __nv_bfloat162 type
#include <iostream>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
      exit(1); \
  } \
} while(0)

// Kernel for vector addition.
template<typename T>
__global__ void vectorAddKernel(T* a, const T* b, int n);
// Explicit instantiation declaration for fp32
extern template __global__ void vectorAddKernel<float>(float* a, const float* b, int n);
// Add specialized declaration for bfloat16
template<>
__global__ void vectorAddKernel<__nv_bfloat16>(__nv_bfloat16* a, const __nv_bfloat16* b, int n);

// Function to launch the kernel.
template<typename T>
void vectorAdd(T* a, const T* b, int n, cudaStream_t stream);
// Explicit instantiation declaration for fp32
extern template void vectorAdd<float>(float* a, const float* b, int n, cudaStream_t stream);
// Explicit instantiation declaration for bf16
extern template void vectorAdd<__nv_bfloat16>(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream);


#endif // COMMON_H

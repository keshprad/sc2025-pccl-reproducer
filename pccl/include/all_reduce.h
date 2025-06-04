#ifndef ALL_REDUCE_H
#define ALL_REDUCE_H

#include <mpi.h>
#include <cuda_bf16.h>

template<typename T>
void recursiveHalvingDoublingAllReduceGPU(T* output, 
                                  const T* input, 
                                  int total_elems, 
                                  T* buf,  
                                  T* recv_buf,  
                                  T* intermediate_buf,  
                                  MPI_Comm comm);
// Explicit instantiation declaration for fp32
extern template void recursiveHalvingDoublingAllReduceGPU<float>(
    float* output,
    const float* input,
    int total_elems,
    float* buf,
    float* recv_buf,
    float* intermediate_buf,
    MPI_Comm comm);
// Explicit instantiation declaration for bf16
extern template void recursiveHalvingDoublingAllReduceGPU<__nv_bfloat16>(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int total_elems,
    __nv_bfloat16* buf,
    __nv_bfloat16* recv_buf,
    __nv_bfloat16* intermediate_buf,
    MPI_Comm comm);

#endif // ALL_REDUCE_H

#ifndef REDUCE_SCATTER_H
#define REDUCE_SCATTER_H

#include <mpi.h>
#include <cuda_bf16.h>      // Include this for __nv_bfloat162 type

template<typename T>
void recursiveHalvingReduceScatterGPU(T* output, 
    const T* input, 
    int total_elems,
    T* buf, 
    T* recv_buf, 
    MPI_Comm comm = MPI_COMM_WORLD);

// Explicit instantiation declaration for fp32
extern template void recursiveHalvingReduceScatterGPU<float>(float* output, 
    const float* input, 
    int total_elems,
    float* buf, 
    float* recv_buf, 
    MPI_Comm comm = MPI_COMM_WORLD);

// Explicit instantiation declaration for bf16
extern template void recursiveHalvingReduceScatterGPU<__nv_bfloat16>(__nv_bfloat16* output, 
    const __nv_bfloat16* input, 
    int total_elems,
    __nv_bfloat16* buf, 
    __nv_bfloat16* recv_buf, 
    MPI_Comm comm = MPI_COMM_WORLD);

template<typename T>
void ringReduceScatterGPU(T* output, 
    const T* input, 
    int total_elems, 
    T* d_buf, 
    T* d_send, 
    T* d_tmp, 
    MPI_Comm comm = MPI_COMM_WORLD);

// Explicit instantiation declaration for fp32
extern template void ringReduceScatterGPU<float>(float* output, 
    const float* input, 
    int total_elems, 
    float* d_buf, 
    float* d_send, 
    float* d_tmp, 
    MPI_Comm comm = MPI_COMM_WORLD);

// Explicit instantiation declaration for bf16
extern template void ringReduceScatterGPU<__nv_bfloat16>(__nv_bfloat16* output, 
    const __nv_bfloat16* input, 
    int total_elems, 
    __nv_bfloat16* d_buf, 
    __nv_bfloat16* d_send, 
    __nv_bfloat16* d_tmp, 
    MPI_Comm comm = MPI_COMM_WORLD);

#endif // REDUCE_SCATTER_H

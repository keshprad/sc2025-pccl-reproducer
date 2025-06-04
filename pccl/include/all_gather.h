#ifndef ALL_GATHER_H
#define ALL_GATHER_H

#include <mpi.h>
#include <cuda_bf16.h>

template<typename T>
void recursiveDoublingAllGatherGPU(T* output, 
                                  const T* input, 
                                  int total_elems, 
                                  T* recv_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);
// Explicit instantiation declaration for fp32
extern template void recursiveDoublingAllGatherGPU(float* output, 
                                  const float* input, 
                                  int total_elems, 
                                  float* recv_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);
// Explicit instantiation declaration for bf16
extern template void recursiveDoublingAllGatherGPU(__nv_bfloat16* output, 
    const __nv_bfloat16* input, 
    int total_elems, 
    __nv_bfloat16* recv_buf,  
    MPI_Comm comm = MPI_COMM_WORLD);

#endif // ALL_GATHER_H

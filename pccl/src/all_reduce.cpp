// #include <torch/extension.h>
// #include <torch/torch.h
#include <cassert>
#include <cmath>

#include "all_gather.h"
#include "all_reduce.h"
#include "common.h"
#include "reduce_scatter.h"


// Performs an all-reduce on GPU tensors via recursive-halving reduce-scatter followed by recursive-doubling all-gather.
//  - output: CUDA device pointer where the final gathered tensor will be stored.
//  - input: CUDA device pointer to the local block of size block_size.
//  - total_elems: total number of elements in output (P * block_size).
//  - buf: main working buffer for the algorithm
//  - recv_buf: buffer to receive data before it is processed
//  - comm: MPI communicator (default MPI_COMM_WORLD).
template<typename T>
void recursiveHalvingDoublingAllReduceGPU(T* output,
                                const T* input,
                                int total_elems,
                                T* buf,                 // Same as input size
                                T* recv_buf,            // Same as input size
                                T* intermediate_buf,    // Input size / world size
                                MPI_Comm comm) {
    recursiveHalvingReduceScatterGPU<T>(intermediate_buf, input, total_elems, buf, recv_buf, comm);

    recursiveDoublingAllGatherGPU<T>(output, intermediate_buf, total_elems, recv_buf, comm);
}
// Explicit instantiation for fp32
template void recursiveHalvingDoublingAllReduceGPU<float>(
    float* output,
    const float* input,
    int total_elems,
    float* buf,
    float* recv_buf,
    float* intermediate_buf,
    MPI_Comm comm);
// Explicit instantiation declaration for bf16
template void recursiveHalvingDoublingAllReduceGPU<__nv_bfloat16>(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int total_elems,
    __nv_bfloat16* buf,
    __nv_bfloat16* recv_buf,
    __nv_bfloat16* intermediate_buf,
    MPI_Comm comm);

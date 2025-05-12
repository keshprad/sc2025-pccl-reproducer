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
void recursiveDoublingAllReduceGPU(void* output,
                                const void* input,
                                int total_elems,
                                void* buf,          // Same as output size
                                void* recv_buf,     // Same as output size
                                MPI_Comm comm) {
    
    // Reduce Scatter + All Gather
    recursiveHalvingReduceScatterGPU(output, input, total_elems, buf, recv_buf, comm);
    recursiveDoublingAllGatherGPU(output, input, total_elems, recv_buf, comm);
}
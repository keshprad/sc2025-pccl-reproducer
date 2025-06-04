// #include <torch/extension.h>
// #include <torch/torch.h>
#include <cassert>
#include <cmath>

#include "all_gather.h"
#include "common.h"


// Performs a recursive doubling all-gather on GPU tensors.
//  - output: CUDA device pointer where the final gathered tensor will be stored.
//  - input: CUDA device pointer to the local block of size block_size.
//  - total_elems: total number of elements in output (P * block_size).
//  - comm: MPI communicator (default MPI_COMM_WORLD).
template<typename T>
void recursiveDoublingAllGatherGPU(T* output, 
                                  const T* input, 
                                  int total_elems, 
                                  T* recv_buf,  // Same as output size
                                  MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Copy local input into its designated block in the output buffer.
    CUDA_CHECK(cudaMemcpyAsync(static_cast<void*>(output + (rank * block_size)), 
                             static_cast<const void*>(input), 
                             block_size * sizeof(T), 
                             cudaMemcpyDeviceToDevice, 
                             stream));

    cudaEvent_t stream_sync_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    int seg_size = 1;  // Start with local block.
    while (seg_size < size) {
        int partner = rank ^ seg_size;
        int group_start = (rank / (2 * seg_size)) * (2 * seg_size);
        int send_offset, recv_offset;
        
        if (rank < partner) {
            send_offset = group_start * block_size;
            recv_offset = (group_start + seg_size) * block_size;
        } else {
            send_offset = (group_start + seg_size) * block_size;
            recv_offset = group_start * block_size;
        }

        int count = seg_size * block_size;
        
        // Record an event on the cuda stream.
        CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        // Wait for the copy to complete.
        CUDA_CHECK(cudaEventSynchronize(stream_sync_event));
        
        MPI_Sendrecv(output + send_offset, count * sizeof(T), MPI_BYTE, partner, 0,
                     output + recv_offset, count * sizeof(T), MPI_BYTE, partner, 0,
                     comm, MPI_STATUS_IGNORE);
        
        seg_size *= 2;
    }
    
    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}
// Explicit instantiation for fp32
template void recursiveDoublingAllGatherGPU(float* output, 
    const float* input, 
    int total_elems, 
    float* recv_buf,  // Same as output size
    MPI_Comm comm);
// Explicit instantiation for bf16
template void recursiveDoublingAllGatherGPU(__nv_bfloat16* output, 
    const __nv_bfloat16* input, 
    int total_elems, 
    __nv_bfloat16* recv_buf,  // Same as output size
    MPI_Comm comm);

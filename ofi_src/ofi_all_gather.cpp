/*
 * Implementation of recursive doubling all gather using OFI/libfabric
 * operations through the NCCL-Net OFI plugin
 */

#include "ofi_all_gather.h"
#include <vector>
#include <cstring>
#include <cassert>

// All-gather function using recursive doubling algorithm
int all_gather(void* input_buffer, void* output_buffer, int block_size) {
    // Use the global communication context
    OFICommContext& ctx = get_comm_context();
    
    if (!ctx.is_initialized()) {
        NCCL_OFI_WARN("Process group not initialized. Call init_process_group() first.");
        return -1;
    }
    
    int rank = ctx.rank;
    int num_ranks = ctx.num_ranks;
    int buffer_type = ctx.buffer_type;
    ncclNet_t *extNet = ctx.extNet;
    
    // Copy local data to the output buffer at position corresponding to rank
    if (buffer_type == NCCL_PTR_HOST) {
        memcpy((char*)output_buffer + (rank * block_size), input_buffer, block_size);
    } else {
        CUDACHECK(hipMemcpy((char*)output_buffer + (rank * block_size), input_buffer, block_size, hipMemcpyDeviceToDevice));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  // Make sure all ranks are ready
    
    // Implement recursive doubling
    int seg_size = 1;  // Start with segments of size 1 (in terms of blocks)
    
    while (seg_size < num_ranks) {
        int partner = rank ^ seg_size;
        int group_start = (rank / (2 * seg_size)) * (2 * seg_size);
        int send_offset, recv_offset;
        
        if (partner >= num_ranks) {  // Make sure partner is valid
            return -1;
        }
        // For now, assume power of 2 ranks, so all partners are valid

        if (rank < partner) {
            send_offset = group_start * block_size;
            recv_offset = (group_start + seg_size) * block_size;
        } else {
            send_offset = (group_start + seg_size) * block_size;
            recv_offset = group_start * block_size;
        }
        
        int count = seg_size * block_size;
        
        // Register memory for the send and receive operations
        void *send_handle = NULL;
        void *recv_handle = NULL;
        OFINCCLCHECK(extNet->regMr((void *)ctx.sComms[rank][partner], 
                    (void *)((char*)output_buffer + send_offset), 
                    count, buffer_type, &send_handle));
        OFINCCLCHECK(extNet->regMr((void *)ctx.rComms[rank][partner], 
                    (void *)((char*)output_buffer + recv_offset), 
                    count, buffer_type, &recv_handle));
        
        // Send and receive data
        nccl_ofi_req_t *send_req = NULL;
        nccl_ofi_req_t *recv_req = NULL;
        
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0))
        // For NCCL v2.12 and later
        
        // For grouped recvs
        // Use group of 1 here since each step of recursive allgather has ranks send/recv 1 msg each
        // TODO: properly define TAG per collective / per "channel". This may not work if multiple collectives simultaneously take place
        int nrecv = 1, tag = 1;
        int counts[1] = {count};
        int tags[1] = {tag};
        void *recv_ptr = (char*)output_buffer + recv_offset;
        void **recv_buffs = &recv_ptr;

        while (send_req == NULL) {
            OFINCCLCHECK(extNet->isend((void *)ctx.sComms[rank][partner], 
                        (void *)((char*)output_buffer + send_offset), 
                        count, tag, send_handle, (void **)&send_req));
        }

        while (recv_req == NULL) {
            OFINCCLCHECK(extNet->irecv((void *)ctx.rComms[rank][partner], 
                        nrecv, 
                        recv_buffs,
                        counts, tags, &recv_handle, (void **)&recv_req));
        }
#else
        // For earlier NCCL versions
        while (send_req == NULL) {
            OFINCCLCHECK(extNet->isend((void *)ctx.sComms[rank][partner], 
                        (void *)((char*)output_buffer + send_offset), 
                        count, send_handle, (void **)&send_req));
        }
        
        while (recv_req == NULL) {
            OFINCCLCHECK(extNet->irecv((void *)ctx.rComms[rank][partner], 
                        (void *)((char*)output_buffer + recv_offset),
                        count, recv_handle, (void **)&recv_req));
        }
#endif
        
        // Wait for completion
        int done_send = 0, done_recv = 0;
        int received_size;
        
        while (!done_send || !done_recv) {
            if (!done_send) {
                OFINCCLCHECK(extNet->test((void *)send_req, &done_send, NULL));
            }
            if (!done_recv) {
                OFINCCLCHECK(extNet->test((void *)recv_req, &done_recv, &received_size));
            }
        }
        
        // For CUDA buffers, we might need to flush
        if (buffer_type == NCCL_PTR_CUDA) {
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 8, 0))
            nccl_ofi_req_t *flush_req = NULL;
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0))
            OFINCCLCHECK(extNet->iflush((void *)ctx.rComms[rank][partner], 
                        nrecv,
                        (void **)((char*)output_buffer + recv_offset),
                        counts, &recv_handle, (void **)&flush_req));
#else
            OFINCCLCHECK(extNet->iflush((void *)ctx.rComms[rank][partner],
                        (void *)((char*)output_buffer + recv_offset),
                        count, recv_handle, (void **)&flush_req));
#endif
            done_recv = 0;
            while (!done_recv && flush_req) {
                OFINCCLCHECK(extNet->test((void *)flush_req, &done_recv, NULL));
            }
#else
            OFINCCLCHECK(extNet->flush((void *)ctx.rComms[rank][partner],
                        (void *)((char*)output_buffer + recv_offset),
                        count, recv_handle));
#endif
        }
        
        // Deregister memory
        OFINCCLCHECK(extNet->deregMr((void *)ctx.sComms[rank][partner], send_handle));
        OFINCCLCHECK(extNet->deregMr((void *)ctx.rComms[rank][partner], recv_handle));
        
        seg_size *= 2;  // Double segment size for next iteration
        
        // Wait for all ranks to complete this phase
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    return 0;
}

int main(int argc, char* argv[])
{
    bool connect_all_peers = false; // Default to optimized connections
    
    // Check command line arguments for connection strategy
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--full-mesh") == 0) {
            connect_all_peers = true;
        }
    }

    MPI_Init(&argc, &argv);
    
    // Initialize the process group
    int init_result = init_process_group(connect_all_peers);
    if (init_result != 0) {
        NCCL_OFI_WARN("Failed to initialize process group");
        MPI_Finalize();
        return init_result;
    }
    
    // Get rank and num_ranks from the global context
    OFICommContext& ctx = get_comm_context();
    int rank = ctx.rank;
    int num_ranks = ctx.num_ranks;
    int buffer_type = ctx.buffer_type;
    
    // Calculate the block size
    int block_size = MESSAGE_SIZE;
    int total_elems = block_size * num_ranks;
    
    // Allocate ROCm/HIP memory for input data (local data), output buffer (result of all-gather)
    char *input_buffer = NULL;
    char *output_buffer = NULL;
    
    OFINCCLCHECK(allocate_buff((void **)&input_buffer, block_size, buffer_type));
    OFINCCLCHECK(allocate_buff((void **)&output_buffer, total_elems, buffer_type));

    // Initialize input buffer with rank-specific data for validation
    for (int i = 0; i < block_size; i++) {
        if (buffer_type == NCCL_PTR_HOST) {
            input_buffer[i] = (char)(rank + 1);  // Fill with rank+1 for easy verification
        } else {
            // For ROCm/CUDA buffers, we need to use hipMemcpy
            char value = (char)(rank + 1);
            CUDACHECK(hipMemcpy(input_buffer + i, &value, sizeof(char), hipMemcpyHostToDevice));
        }
    }
    
    // Call the all_gather function (now uses global context)
    int result = all_gather(input_buffer, output_buffer, block_size);
    if (result != 0) {
        NCCL_OFI_WARN("Rank %d: All-gather operation failed", rank);
        
        // Cleanup before exit
        OFINCCLCHECK(deallocate_buffer(input_buffer, buffer_type));
        OFINCCLCHECK(deallocate_buffer(output_buffer, buffer_type));
        destroy_process_group();
        MPI_Finalize();
        return result;
    }
    
    // Validate the result - each process should now have all data
    NCCL_OFI_INFO(NCCL_NET, "Rank %d: All-gather complete, validating results", rank);
    
    bool success = true;
    for (int r = 0; r < num_ranks; r++) {
        for (int i = 0; i < block_size; i++) {
            char expected = (char)(r + 1);
            char actual;
            
            if (buffer_type == NCCL_PTR_HOST) {
                actual = output_buffer[r * block_size + i];
            } else {
                CUDACHECK(hipMemcpy(&actual, output_buffer + r * block_size + i, sizeof(char), hipMemcpyDeviceToHost));
            }
            
            if (actual != expected) {
                NCCL_OFI_WARN("Rank %d: Validation failed at position %d (block %d): expected %d, got %d", 
                              rank, r * block_size + i, r, expected, actual);
                success = false;
                break;
            }
        }
        if (!success) break;
    }
    
    if (success) {
        NCCL_OFI_INFO(NCCL_NET, "Rank %d: All-gather validation successful", rank);
    }
    
    // Cleanup memory
    OFINCCLCHECK(deallocate_buffer(input_buffer, buffer_type));
    OFINCCLCHECK(deallocate_buffer(output_buffer, buffer_type));
    
    // Destroy the process group (cleans up all connections)
    destroy_process_group();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    return 0;
}

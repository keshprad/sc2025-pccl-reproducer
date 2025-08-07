/*
 * Implementation of recursive doubling all gather using OFI/libfabric
 * operations through the NCCL-Net OFI plugin
 */

// utilities from aws-ofi-rccl lib - use -I<path/to/aws-ofi-rccl>/include when compiling
#include "tests/test-common.h"

#include <vector>
#include <cstring>
#include <cassert>

#define GPUS_PER_NODE 8
#define NICS_PER_NODE 4
#define MESSAGE_SIZE 1024 * 1024  // 1 MB message size
#define TAG 1

// Structure to hold communication setup data
struct OFICommContext {
    ncclNet_t *extNet;
    int rank;
    int num_ranks;
    int dev;
    int buffer_type;
    std::vector<std::vector<sendComm_t*>> sComms;
    std::vector<std::vector<recvComm_t*>> rComms;
    
    OFICommContext(int r, int nr, int d, int bt, ncclNet_t *net) 
        : rank(r), num_ranks(nr), dev(d), buffer_type(bt), extNet(net),
          sComms(nr, std::vector<sendComm_t*>(nr, NULL)),
          rComms(nr, std::vector<recvComm_t*>(nr, NULL)) {}
};

// All-gather function using recursive doubling algorithm
int all_gather(void* input_buffer, void* output_buffer, int block_size, OFICommContext& ctx) {
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
    int rank, proc_name_len, num_ranks, local_rank = 0;
    int buffer_type = NCCL_PTR_HOST; // Default to host buffers
    
    /* Plugin defines */
    int ndev, dev, cuda_dev, i;
    ncclNet_t *extNet = NULL;

    ofi_log_function = logger;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    char all_proc_name[num_ranks][MPI_MAX_PROCESSOR_NAME];

    MPI_Get_processor_name(all_proc_name[rank], &proc_name_len);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name,
                MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

    /* Determine local rank */
    for (i = 0; i < num_ranks; i++) {
        if (!strcmp(all_proc_name[rank], all_proc_name[i])) {
            if (i < rank) {
                ++local_rank;
            }
        }
    }

    /* Set CUDA device for subsequent device memory allocation, in case GDR is used */
    cuda_dev = local_rank;
    NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", cuda_dev);
    CUDACHECK(hipSetDevice(cuda_dev));

    /* Get external Network from NCCL-OFI library */
    extNet = get_extNet();
    if (extNet == NULL)
        return -1;

    /* Init API */
    OFINCCLCHECK(extNet->init(&logger));
    NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.",
            rank, all_proc_name[rank], extNet->name);

    /* Devices API */
    OFINCCLCHECK(extNet->devices(&ndev));
    NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

    /* Indicates if NICs support GPUDirect */
    int support_gdr[ndev];

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4))
    /* Get Properties for the device */
    for (dev = 0; dev < ndev; dev++) {
        ncclNetProperties_t props = {0};
        OFINCCLCHECK(extNet->getProperties(dev, &props));
        print_dev_props(dev, &props);

        /* Set CUDA support */
        support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
    }
#else
    /* Get PCIe path and plugin memory pointer support */
    for (dev = 0; dev < ndev; dev++) {
        char *path = NULL;
        int supported_types = 0;
        extNet->pciPath(dev, &path);
        OFINCCLCHECK(extNet->ptrSupport(dev, &supported_types));
        NCCL_OFI_TRACE(NCCL_INIT, "Dev %d has path %s and supports pointers of type %d", dev, path, supported_types);

        /* Set CUDA support */
        support_gdr[dev] = is_gdr_supported_nic(supported_types);
    }
#endif

    /* Choose specific device per rank for communication */
    dev = local_rank / (GPUS_PER_NODE / ndev);
    NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Rank %d: local_rank=%d, cuda_dev=%d, net_dev=%d, hostname=%s", 
        rank, local_rank, cuda_dev, dev, all_proc_name[rank]);

    if (support_gdr[dev] == 1) {
        NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
                "Network supports communication using CUDA buffers. Dev: %d", dev);
        buffer_type = NCCL_PTR_CUDA;
    }

    // Initialize communication context
    OFICommContext comm_ctx(rank, num_ranks, dev, buffer_type, extNet);
    
    // Data structures to hold connection information
    std::vector<std::vector<listenComm_t*>> lComms(num_ranks, std::vector<listenComm_t*>(num_ranks, NULL));
    
    // Create a vector to hold all handles
    std::vector<char> handles(num_ranks * num_ranks * NCCL_NET_HANDLE_MAXSIZE, 0);
    
    // Create listening endpoints - one for each potential peer
    for (int peer = 0; peer < num_ranks; peer++) {
        if (peer == rank) continue;

        char* myHandle = handles.data() + (rank * num_ranks + peer) * NCCL_NET_HANDLE_MAXSIZE;

        NCCL_OFI_INFO(NCCL_NET, "Rank %d: Creating listener for rank %d on dev %d", rank, peer, dev);
        OFINCCLCHECK(extNet->listen(dev, (void *)myHandle, (void **)&lComms[rank][peer]));
    } 
    
    // Exchange handles among all ranks
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), 
                num_ranks * NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, MPI_COMM_WORLD);
    
    // Every rank connects to every other rank
    for (int peer = 0; peer < num_ranks; peer++) {
        if (peer == rank) continue;
        
        // Get handle of peer's listener
        char *peerHandle = handles.data() + (peer * num_ranks + rank) * NCCL_NET_HANDLE_MAXSIZE;
        
        NCCL_OFI_INFO(NCCL_NET, "Rank %d: Connecting to rank %d", rank, peer);
        while (comm_ctx.sComms[rank][peer] == NULL) {
            OFINCCLCHECK(extNet->connect(dev, (void *)peerHandle, (void **)&comm_ctx.sComms[rank][peer]));
        }
    }
    
    // Accept connections from all other ranks on the appropriate listeners
    for (int peer = 0; peer < num_ranks; peer++) {
        if (peer == rank) continue;
        
        NCCL_OFI_INFO(NCCL_NET, "Rank %d: Accepting connection from rank %d", rank, peer);
        while (comm_ctx.rComms[rank][peer] == NULL) {
            OFINCCLCHECK(extNet->accept((void *)lComms[rank][peer], (void **)&comm_ctx.rComms[rank][peer]));
        }
    }

    NCCL_OFI_INFO(NCCL_NET, "Rank %d: All connections established", rank);
    
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
    
    // Call the all_gather function
    int result = all_gather(input_buffer, output_buffer, block_size, comm_ctx);
    if (result != 0) {
        NCCL_OFI_WARN("Rank %d: All-gather operation failed", rank);
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
    
    // Close all connections
    for (int peer = 0; peer < num_ranks; peer++) {
        if (peer == rank) continue;
        if (comm_ctx.sComms[rank][peer]) {
            OFINCCLCHECK(extNet->closeSend((void *)comm_ctx.sComms[rank][peer]));
        }
        if (comm_ctx.rComms[rank][peer]) {
            OFINCCLCHECK(extNet->closeRecv((void *)comm_ctx.rComms[rank][peer]));
        }
        if (lComms[rank][peer]) {
            OFINCCLCHECK(extNet->closeListen(static_cast<void*>(lComms[rank][peer])));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    return 0;
}

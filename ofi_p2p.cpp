/*
 * This program establishes connections between all pairs of ranks
 * and then sends messages specifically between rank 0 and rank 1
 */

#include "tests/test-common.h"
#include <vector>

#define GPUS_PER_NODE 8
#define NICS_PER_NODE 4

int main(int argc, char* argv[])
{
    int rank, proc_name_len, num_ranks, local_rank = 0;
    int buffer_type = NCCL_PTR_HOST;

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
    // Use a consistent device across all ranks for more predictable behavior
    // dev = 0;
    NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Rank %d: local_rank=%d, cuda_dev=%d, net_dev=%d, hostname=%s", 
        rank, local_rank, cuda_dev, dev, all_proc_name[rank]);

    if (support_gdr[dev] == 1) {
        NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
                "Network supports communication using CUDA buffers. Dev: %d", dev);
        buffer_type = NCCL_PTR_CUDA;
    }

    // Data structures to hold connection information
    std::vector<std::vector<listenComm_t*>> lComms(num_ranks, std::vector<listenComm_t*>(num_ranks, NULL));
    std::vector<std::vector<sendComm_t*>> sComms(num_ranks, std::vector<sendComm_t*>(num_ranks, NULL));
    std::vector<std::vector<recvComm_t*>> rComms(num_ranks, std::vector<recvComm_t*>(num_ranks, NULL));
    
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
        while (sComms[rank][peer] == NULL) {
            OFINCCLCHECK(extNet->connect(dev, (void *)peerHandle, (void **)&sComms[rank][peer]));
        }
    }
    
    // Accept connections from all other ranks on the appropriate listeners
    for (int peer = 0; peer < num_ranks; peer++) {
        if (peer == rank) continue;
        
        NCCL_OFI_INFO(NCCL_NET, "Rank %d: Accepting connection from rank %d", rank, peer);
        while (rComms[rank][peer] == NULL) {
            OFINCCLCHECK(extNet->accept((void *)lComms[rank][peer], (void **)&rComms[rank][peer]));
        }
    }

    NCCL_OFI_INFO(NCCL_NET, "Rank %d: using dev %d (local_rank = %d, ndev = %d)\n", rank, dev, local_rank, ndev);
    
    // Now all connections are established, let's send messages between rank 0 and rank 1
    if (rank == 0 || rank == 8) {
        const int NUM_MSGS = 5;
        const int MSG_SIZE = 1024;
        int inflight_reqs = NUM_MSGS;
        void *mhandle[NUM_MSGS];
        nccl_ofi_req_t *req[NUM_MSGS] = {NULL};
        int req_completed[NUM_MSGS] = {0};
        char *send_buf[NUM_MSGS] = {NULL};
        char *recv_buf[NUM_MSGS] = {NULL};
        int done, received_size;
        
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0))
        /* For grouped recvs */
        int tag = 1;
        int nrecv = NCCL_OFI_MAX_RECVS;
        int *sizes = (int *)malloc(sizeof(int)*nrecv);
        int *tags = (int *)malloc(sizeof(int)*nrecv);
        for (int recv_n = 0; recv_n < nrecv; recv_n++) {
            sizes[recv_n] = MSG_SIZE;
            tags[recv_n] = tag;
        }
#endif
        // buffer_type = NCCL_PTR_HOST;
        NCCL_OFI_INFO(NCCL_NET, "Rank: %d\tbuffer_type: %d", rank, buffer_type);
        NCCL_OFI_INFO(NCCL_NET, "Rank: %d\tNCCL_PTR_HOST: %d", rank, NCCL_PTR_HOST);
        NCCL_OFI_INFO(NCCL_NET, "Rank: %d\tNCCL_PTR_CUDA: %d", rank, NCCL_PTR_CUDA);
        
        if (rank == 0) {
            // Get handle of peer's listener
            int peer = 8;

            // char *peerHandle = handles.data() + (peer * NCCL_NET_HANDLE_MAXSIZE);

            // NCCL_OFI_INFO(NCCL_NET, "Rank %d: Connecting to rank %d", rank, peer);
            // while (sComms[rank][peer] == NULL) {
            //     OFINCCLCHECK(extNet->connect(dev, (void *)peerHandle, (void **)&sComms[rank][peer]));
            // }

            // NCCL_OFI_INFO(NCCL_NET, "Rank %d: Accepting connection from rank %d", rank, peer);
            // while (rComms[rank][peer] == NULL) {
            //     OFINCCLCHECK(extNet->accept((void *)lComms[rank], (void **)&rComms[rank][peer]));
            // }

            // Rank 0 sends to rank 1
            NCCL_OFI_INFO(NCCL_NET, "Rank %d, sending %d messages to rank %d", rank, NUM_MSGS, peer);
            for (i = 0; i < NUM_MSGS; i++) {
                OFINCCLCHECK(allocate_buff((void **)&send_buf[i], MSG_SIZE, buffer_type));
                // Ensure buffer is properly initialized with non-zero data
                OFINCCLCHECK(initialize_buff((void *)send_buf[i], MSG_SIZE, buffer_type));
                
                // Ensure proper memory registration
                OFINCCLCHECK(extNet->regMr((void *)sComms[rank][peer], (void *)send_buf[i], MSG_SIZE,
                            buffer_type, &mhandle[i]));  // Force HOST registration for debugging
                NCCL_OFI_TRACE(NCCL_NET, "Rank %d, Successfully registered send memory for request %d", rank, i);

                NCCL_OFI_INFO(NCCL_NET, "Rank %d: About to send using net_dev=%d, cuda_dev=%d, buffer_type=%d", 
                    rank, dev, cuda_dev, buffer_type);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
                while (req[i] == NULL) {
                    OFINCCLCHECK(extNet->isend((void *)sComms[rank][peer], (void *)send_buf[i], MSG_SIZE, tag,
                                mhandle[i], (void **)&req[i]));
                }
#else
                while (req[i] == NULL) {
                    OFINCCLCHECK(extNet->isend((void *)sComms[rank][peer], (void *)send_buf[i], MSG_SIZE,
                                mhandle[i], (void **)&req[i]));
                }
#endif
            }
            NCCL_OFI_INFO(NCCL_NET, "Rank %d, sent %d messages to rank %d", rank, NUM_MSGS, peer);
        }
        else if (rank == 8) {
            // Get handle of peer's listener
            int peer = 0;
            
            // char *peerHandle = handles.data() + (peer * NCCL_NET_HANDLE_MAXSIZE);
            
            // NCCL_OFI_INFO(NCCL_NET, "Rank %d: Connecting to rank %d", rank, peer);
            // while (sComms[rank][peer] == NULL) {
            //     OFINCCLCHECK(extNet->connect(dev, (void *)peerHandle, (void **)&sComms[rank][peer]));
            // }

            // NCCL_OFI_INFO(NCCL_NET, "Rank %d: Accepting connection from rank %d", rank, peer);
            // while (rComms[rank][peer] == NULL) {
            //     OFINCCLCHECK(extNet->accept((void *)lComms[rank], (void **)&rComms[rank][peer]));
            // }
            
            // Rank 1 receives from rank 0
            NCCL_OFI_INFO(NCCL_NET, "Rank %d receiving %d messages from rank %d", rank, NUM_MSGS, peer);
            for (i = 0; i < NUM_MSGS; i++) {
                OFINCCLCHECK(allocate_buff((void **)&recv_buf[i], MSG_SIZE, buffer_type));
                OFINCCLCHECK(extNet->regMr((void *)rComms[rank][peer], (void *)recv_buf[i], MSG_SIZE,
                            buffer_type, &mhandle[i]));
                NCCL_OFI_TRACE(NCCL_NET, "Rank %d, Successfully registered recv memory for request %d", rank, i);
                
                NCCL_OFI_INFO(NCCL_NET, "Rank %d: About to recv using net_dev=%d, cuda_dev=%d, buffer_type=%d", 
                    rank, dev, cuda_dev, buffer_type);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
                while (req[i] == NULL) {
                    OFINCCLCHECK(extNet->irecv((void *)rComms[rank][peer], nrecv, (void **)&recv_buf[i],
                                sizes, tags, &mhandle[i], (void **)&req[i]));
                }
#else
                while (req[i] == NULL) {
                    OFINCCLCHECK(extNet->irecv((void *)rComms[rank][peer], (void *)recv_buf[i],
                                MSG_SIZE, mhandle[i], (void **)&req[i]));
                }
#endif
            }
            NCCL_OFI_INFO(NCCL_NET, "Rank %d, received %d messages from rank %d", rank, NUM_MSGS, peer);
        }

        // Create expected buffer for validation
        char *expected_buf = NULL;
        OFINCCLCHECK(allocate_buff((void **)&expected_buf, MSG_SIZE, NCCL_PTR_HOST));
        OFINCCLCHECK(initialize_buff((void *)expected_buf, MSG_SIZE, NCCL_PTR_HOST));
        
        // Test for completions
        while (inflight_reqs > 0) {
            for (i = 0; i < NUM_MSGS; i++) {
                if (req_completed[i])
                    continue;
                
                OFINCCLCHECK(extNet->test((void *)req[i], &done, &received_size));
                if (done) {
                    inflight_reqs--;
                    req_completed[i] = 1;
                    
                    if ((rank == 8) && (buffer_type == NCCL_PTR_CUDA)) {
                        int peer = 0;
                        NCCL_OFI_TRACE(NCCL_NET,
                                "Issue flush for data consistency. Request idx: %d", i);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 8, 0)) /* Support NCCL v2.8 */
                        nccl_ofi_req_t *iflush_req = NULL;
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
                        OFINCCLCHECK(extNet->iflush((void *)rComms[rank][peer], nrecv,
                                    (void **)&recv_buf[i],
                                    sizes, &mhandle[i], (void **)&iflush_req));
#else
                        OFINCCLCHECK(extNet->iflush((void *)rComms[rank][peer],
                                    (void **)recv_buf[i],
                                    MSG_SIZE, mhandle[i], (void **)&iflush_req));
#endif
                        done = 0;
                        if (iflush_req) {
                            while (!done) {
                                OFINCCLCHECK(extNet->test((void *)iflush_req, &done, NULL));
                            }
                        }
#else
                        OFINCCLCHECK(extNet->flush((void *)rComms[rank][peer],
                                    (void *)recv_buf[i],
                                    MSG_SIZE, mhandle[i]));
#endif
                    }
                    
                    // Deregister memory handle
                    if (rank == 0) {
                        int peer = 8;
                        OFINCCLCHECK(extNet->deregMr((void *)sComms[rank][peer], mhandle[i]));
                    }
                    else if (rank == 8) {
                        int peer = 0;
                        if ((buffer_type == NCCL_PTR_CUDA) && !ofi_nccl_gdr_flush_disable()) {
                            // Data validation may fail if flush operations are disabled
                        } else {
                            OFINCCLCHECK(validate_data(recv_buf[i], expected_buf, MSG_SIZE, buffer_type));
                        }
                        OFINCCLCHECK(extNet->deregMr((void *)rComms[rank][peer], mhandle[i]));
                    }
                }
            }
        }

        // Deallocate buffers
        OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
        for (i = 0; i < NUM_MSGS; i++) {
            if (send_buf[i])
                OFINCCLCHECK(deallocate_buffer(send_buf[i], buffer_type));
            if (recv_buf[i])
                OFINCCLCHECK(deallocate_buffer(recv_buf[i], buffer_type));
        }

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0))
        free(sizes);
        free(tags);
#endif
    }
    
    // Close all connections
    for (int peer = 0; peer < num_ranks; peer++) {
        if (peer == rank) continue;
        if (sComms[rank][peer]) {
            OFINCCLCHECK(extNet->closeSend((void *)sComms[rank][peer]));
        }
        if (rComms[rank][peer]) {
            OFINCCLCHECK(extNet->closeRecv((void *)rComms[rank][peer]));
        }
        if (lComms[rank][peer]) {
            OFINCCLCHECK(extNet->closeListen(static_cast<void*>(lComms[rank][peer])));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}

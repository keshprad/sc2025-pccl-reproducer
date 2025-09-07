/*
 * Implementation of OFI distributed communication functions
 * Similar to torch.distributed functionality but using OFI/libfabric directly
 */

#include "ofi_distributed.h"
#include <cstring>

// Global communication context
OFICommContext g_comm_ctx;

// Initialize the OFI process group - similar to torch.distributed.init_process_group
int init_process_group(bool connect_all_peers) {
    int rank, proc_name_len, num_ranks, local_rank;
    int buffer_type = NCCL_PTR_HOST; // Default to host buffers
    
    /* Plugin defines */
    int ndev, dev, cuda_dev, i;
    ncclNet_t *extNet = NULL;

    ofi_log_function = logger;

    // Get MPI rank and size information
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    char all_proc_name[num_ranks][MPI_MAX_PROCESSOR_NAME];

    MPI_Get_processor_name(all_proc_name[rank], &proc_name_len);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name,
                MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

    /* Determine local rank */
    local_rank = 0;
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

    // Initialize global communication context using parameterized constructor
    g_comm_ctx = OFICommContext(extNet, rank, num_ranks, local_rank, cuda_dev, dev, buffer_type);
    
    // Determine which peers to connect to
    std::set<int> needed_peers;
    if (connect_all_peers) {
        // Connect to all peers (full mesh)
        for (int peer = 0; peer < num_ranks; peer++) {
            if (peer != rank) {
                needed_peers.insert(peer);
            }
        }
        NCCL_OFI_TRACE(NCCL_NET, "Rank %d: Creating full mesh connections to %lu peers", 
                       rank, needed_peers.size());
    } else {
        // Connect only to peers needed for recursive doubling all-gather
        for (int step_size = 1; step_size < num_ranks; step_size *= 2) {
            int partner = rank ^ step_size;
            if (partner < num_ranks) {
                needed_peers.insert(partner);
            }
        }
        NCCL_OFI_TRACE(NCCL_NET, "Rank %d: Creating optimized connections to %lu peers for recursive doubling", 
                       rank, needed_peers.size());
    }
    
    // Create a vector to hold all handles (keep full size for MPI_Allgather)
    std::vector<char> handles(num_ranks * num_ranks * NCCL_NET_HANDLE_MAXSIZE, 0);
    
    // Create listening endpoints only for ranks that will connect to us
    for (int peer : needed_peers) {
        char* myHandle = handles.data() + (rank * num_ranks + peer) * NCCL_NET_HANDLE_MAXSIZE;

        NCCL_OFI_TRACE(NCCL_NET, "Rank %d: Creating listener for rank %d on dev %d", rank, peer, dev);
        OFINCCLCHECK(extNet->listen(dev, (void *)myHandle, (void **)&g_comm_ctx.lComms[rank][peer]));
    } 
    
    // Exchange handles among all ranks
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), 
                num_ranks * NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, MPI_COMM_WORLD);
    
    // Connect only to needed peers
    for (int peer : needed_peers) {
        // Get handle of peer's listener
        char *peerHandle = handles.data() + (peer * num_ranks + rank) * NCCL_NET_HANDLE_MAXSIZE;
        
        NCCL_OFI_TRACE(NCCL_NET, "Rank %d: Connecting to rank %d", rank, peer);
        while (g_comm_ctx.sComms[rank][peer] == NULL) {
            OFINCCLCHECK(extNet->connect(dev, (void *)peerHandle, (void **)&g_comm_ctx.sComms[rank][peer]));
        }
    }
    
    // Accept connections only from needed peers
    for (int peer : needed_peers) {
        NCCL_OFI_TRACE(NCCL_NET, "Rank %d: Accepting connection from rank %d", rank, peer);
        while (g_comm_ctx.rComms[rank][peer] == NULL) {
            OFINCCLCHECK(extNet->accept((void *)g_comm_ctx.lComms[rank][peer], (void **)&g_comm_ctx.rComms[rank][peer]));
        }
    }

    // Store connected peers for cleanup
    g_comm_ctx.connected_peers = needed_peers;

    NCCL_OFI_INFO(NCCL_NET, "Rank %d: Process group initialization complete (%lu connections established)", 
                  rank, needed_peers.size());
    
    return 0;
}

// Cleanup and destroy the process group - similar to torch.distributed.destroy_process_group
int destroy_process_group() {
    if (!g_comm_ctx.is_initialized()) {
        NCCL_OFI_WARN("Process group not initialized or already destroyed");
        return -1;
    }
    
    NCCL_OFI_INFO(NCCL_NET, "Rank %d: Destroying process group", g_comm_ctx.rank);
    
    // Close all connections - only close the ones we actually created
    for (int peer : g_comm_ctx.connected_peers) {
        if (g_comm_ctx.sComms[g_comm_ctx.rank][peer]) {
            OFINCCLCHECK(g_comm_ctx.extNet->closeSend((void *)g_comm_ctx.sComms[g_comm_ctx.rank][peer]));
            g_comm_ctx.sComms[g_comm_ctx.rank][peer] = nullptr;
        }
        if (g_comm_ctx.rComms[g_comm_ctx.rank][peer]) {
            OFINCCLCHECK(g_comm_ctx.extNet->closeRecv((void *)g_comm_ctx.rComms[g_comm_ctx.rank][peer]));
            g_comm_ctx.rComms[g_comm_ctx.rank][peer] = nullptr;
        }
        if (g_comm_ctx.lComms[g_comm_ctx.rank][peer]) {
            OFINCCLCHECK(g_comm_ctx.extNet->closeListen(static_cast<void*>(g_comm_ctx.lComms[g_comm_ctx.rank][peer])));
            g_comm_ctx.lComms[g_comm_ctx.rank][peer] = nullptr;
        }
    }
    
    g_comm_ctx.connected_peers.clear();
    
    NCCL_OFI_INFO(NCCL_NET, "Rank %d: Process group destroyed", g_comm_ctx.rank);
    
    // Reset the context
    g_comm_ctx = OFICommContext();
    
    return 0;
}

// Helper function to get the communication context
OFICommContext& get_comm_context() {
    return g_comm_ctx;
}

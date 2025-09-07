/*
 * Header file for OFI distributed communication functions
 * Similar to torch.distributed functionality but using OFI/libfabric directly
 */

#ifndef OFI_DISTRIBUTED_H_
#define OFI_DISTRIBUTED_H_

#include "ofi_distributed_utils.h"
#include <vector>
#include <set>
#include <mpi.h>

#define GPUS_PER_NODE 8
#define NICS_PER_NODE 4

// Structure to hold communication setup data
struct OFICommContext {
    ncclNet_t *extNet;
    int rank;
    int num_ranks;
    int dev;
    int buffer_type;
    int local_rank;
    int cuda_dev;
    std::vector<std::vector<sendComm_t*>> sComms;
    std::vector<std::vector<recvComm_t*>> rComms;
    std::vector<std::vector<listenComm_t*>> lComms;
    std::set<int> connected_peers;
    
    // Default constructor
    OFICommContext() 
        : extNet(nullptr), rank(-1), num_ranks(0), dev(-1), buffer_type(NCCL_PTR_HOST),
          local_rank(-1), cuda_dev(-1) {}
    
    // Parameterized constructor for initialization
    OFICommContext(ncclNet_t* ext_net, int r, int nr, int local_r, int cuda_d, int d, int buf_type) 
        : extNet(ext_net), rank(r), num_ranks(nr), dev(d), buffer_type(buf_type),
          local_rank(local_r), cuda_dev(cuda_d) {
        resize_for_ranks(nr);
    }
    
    void resize_for_ranks(int nr) {
        num_ranks = nr;
        sComms.assign(nr, std::vector<sendComm_t*>(nr, nullptr));
        rComms.assign(nr, std::vector<recvComm_t*>(nr, nullptr));
        lComms.assign(nr, std::vector<listenComm_t*>(nr, nullptr));
    }
    
    // Helper method to check if context is initialized
    bool is_initialized() const {
        return extNet != nullptr && rank >= 0;
    }
};

// Global communication context
extern OFICommContext g_comm_ctx;

// Function declarations
int init_process_group(bool connect_all_peers = false);
int destroy_process_group();
OFICommContext& get_comm_context();

#endif // OFI_DISTRIBUTED_H_

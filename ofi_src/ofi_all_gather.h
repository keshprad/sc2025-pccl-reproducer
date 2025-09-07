/*
 * Header file for recursive doubling all gather using OFI/libfabric
 * operations through the NCCL-Net OFI plugin
 */

#ifndef OFI_ALL_GATHER_H_
#define OFI_ALL_GATHER_H_

#include "ofi_distributed.h"
#include <mpi.h>

// Constants for all-gather operation
#define MESSAGE_SIZE 1024 * 1024  // 1 MB message size
#define TAG 1  // TODO: properly define TAG per collective / per "channel"

/**
 * Performs all-gather operation using recursive doubling algorithm
 * 
 * @param input_buffer  Pointer to input data buffer (local data)
 * @param output_buffer Pointer to output buffer (result of all-gather)
 * @param block_size    Size of each process's data block in bytes
 * @return 0 on success, -1 on failure
 */
int all_gather(void* input_buffer, void* output_buffer, int block_size);

#endif // OFI_ALL_GATHER_H_

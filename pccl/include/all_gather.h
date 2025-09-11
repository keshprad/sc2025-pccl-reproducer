#ifndef ALL_GATHER_H
#define ALL_GATHER_H

#include <mpi.h>

void recursiveDoublingAllGatherGPU(void* output, 
                                  const void* input, 
                                  int64_t total_elems, 
                                  void* recv_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);

void ringAllGatherGPU(void* output,
                      const void* input,
                      int64_t total_elems,
                      MPI_Comm comm);

#endif // ALL_GATHER_H

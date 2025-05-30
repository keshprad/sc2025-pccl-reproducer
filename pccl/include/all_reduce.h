#ifndef ALL_REDUCE_H
#define ALL_REDUCE_H

#include <mpi.h>

void recursiveHalvingDoublingAllReduceGPU(float* output, 
                                  const float* input, 
                                  int total_elems, 
                                  float* buf,  
                                  float* recv_buf,  
                                  float* intermediate_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);

#endif // ALL_REDUCE_H

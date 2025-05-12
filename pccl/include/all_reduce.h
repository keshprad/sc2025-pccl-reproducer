#ifndef ALL_REDUCE_H
#define ALL_REDUCE_H

#include <mpi.h>

void recursiveDoublingAllReduceGPU(void* output, 
                                  const void* input, 
                                  int total_elems, 
                                  void* buf,  
                                  void* recv_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);

#endif // ALL_REDUCE_H

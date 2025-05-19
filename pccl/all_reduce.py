import torch
import torch.distributed as dist
from mpi4py import MPI
from typing import List, Optional, Union
from .request import Request
from .process_groups import ProcessGroups
import numpy as np
from .all_gather import all_gather_2D, recursive_doubling_allgather_mpi
from .reduce_scatter import reduce_scatter_2D, recursive_halving_reduce_scatter_mpi


import torch
import math
from mpi4py import MPI
from typing import Optional

def recursive_halving_doubling_allreduce_mpi(output_tensor: torch.Tensor,
                                                input_tensor: torch.Tensor,
                                                group: Optional[MPI.Comm] = None,
                                                async_op: bool = False,):
    """
    Performs a recursive halving and doubling based all-reduce on CUDA tensors using MPI point-to-point
    Sendrecv operations.

    Each process starts with a 1D input_tensor (block_size) and the final output_tensor 
    is a 1D tensor of size (block_size). The goal is to reduce (using op, e.g. torch.add)
    the block over all processes so that all processes end with the fully reduced block.
    """
    assert not async_op, "Non-blocking operations not supported"

    world_size = group.Get_size()
    
    output_intermediate = torch.empty(input_tensor.size(0) // world_size,
                                      device=input_tensor.device,
                                      dtype=input_tensor.dtype)
    recursive_halving_reduce_scatter_mpi(output_intermediate, input_tensor, group, async_op)
    recursive_doubling_allgather_mpi(output_tensor, output_intermediate, group, async_op)

def _all_reduce(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    directly_call_mpi: bool = False,
    use_rh_and_rd: bool = False,
    use_pccl_cpp_backend: bool = False,
) -> Optional[Request]:
    
    # Case 1: torch.distributed.ProcessGroup
    if group is None or isinstance(group, dist.ProcessGroup):
        # Delegate to torch.distributed.all_reduce
        
        # all_reduce_into_tensor doesn't exist...
        # Copy input tensor to output tensor
        output_tensor.copy_(input_tensor)
        print("nccl thru torch")
        request = dist.all_reduce(output_tensor,
                                  group=group,
                                  async_op=async_op)
    
    # Case 2: mpi4py.MPI.Comm
    elif isinstance(group, MPI.Comm):
        if use_pccl_cpp_backend:
            print("pccl directly")
            print("group size", group.Get_size())
            import pccl as pccl_cpp
            request = pccl_cpp.all_reduce_mpi(output_tensor,
                                              input_tensor,
                                              group,
                                              "recursive")
        else:
            if not directly_call_mpi:
                if use_rh_and_rd:
                    print("recursive_halving_doubling")
                    request = recursive_halving_doubling_allreduce_mpi(output_tensor, input_tensor, group, async_op)
                else:
                    print("allreduce ring")
                    # TODO: allreduce ring? (no current allgather ring implementation)
                    # request = ring_allreduce_mpi(output_tensor, input_tensor, group, async_op)
                    raise Exception("ring allreduce currently not implemented")
            else:
                print("MPI directly")
                torch.cuda.current_stream().synchronize()
                if async_op:
                    request = group.Iallreduce(input_tensor, output_tensor)
                else:
                    request = group.Allreduce(input_tensor, output_tensor)

    return request

def all_reduce_2D(output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[ProcessGroups] = None,
    async_op: bool = False,
    use_rh_and_rd: bool = False,
    use_pccl_cpp_backend: bool = False):
    
    assert not async_op, "Non blocking version not implemented"
    
    assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "all_gather_2D only admits 1D tensors"

    # TESTING cpp allreduce
    output_intermediate = torch.empty(input_tensor.size(0), device=input_tensor.device, dtype=input_tensor.dtype)
    # Step-1 inter-node all-gather 
    _all_reduce(output_intermediate, input_tensor, group.get_outer_group(), async_op=False, use_rh_and_rd=True, use_pccl_cpp_backend=True, directly_call_mpi=True)
    # Step-2 intra-node all-gather
    _all_reduce(output_tensor, output_intermediate, group.get_inner_group(), async_op=False, use_rh_and_rd=True, use_pccl_cpp_backend=True, directly_call_mpi=True)

    # intra_node_group_size, inter_node_group_size = group.get_world_size()
    # world_size = intra_node_group_size * inter_node_group_size
    # output_intermediate = torch.empty(input_tensor.size(0) // world_size,
    #                                   device=input_tensor.device,
    #                                   dtype=input_tensor.dtype)

    # # Step-1 2-dim reduce-scatter
    # reduce_scatter_2D(output_intermediate, input_tensor, group, async_op, use_rh_and_rd, use_pccl_cpp_backend)

    # # Step-2 2-dim all-gather
    # all_gather_2D(output_tensor, output_intermediate, group, async_op, use_rh_and_rd, use_pccl_cpp_backend)

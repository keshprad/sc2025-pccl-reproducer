import torch
import torch.distributed as dist
from mpi4py import MPI
import numpy as np
import os
from argparse import ArgumentParser
import csv
from pccl import ProcessGroups, all_reduce_2D, _all_reduce
from pccl.build_kernels import build as build_pccl
from benchmark_raw_collectives.utils import time_something, init, allclose

def get_gpu_counts_and_job_id():
    gpu_count = int(os.getenv("SLURM_NTASKS", 1))  # Default to 1 if not found
    slurm_job_id = os.getenv("SLURM_JOB_ID", "unknown")
    return gpu_count, slurm_job_id

if __name__ == "__main__":
    init()
    parser = ArgumentParser()
    parser.add_argument("--num-gpus-per-node", 
                        type=int, 
                        required=True, 
                        help="specify number of GPUs/GCDs per node")
    parser.add_argument("--machine",
                        type=str,
                        required=True,
                        help="specify the machine you are running on. Will be used to create folders")
    parser.add_argument("--library", type=str, choices=["pccl", "mpi", "xccl"])
    parser.add_argument("--pccl-disable-cpp-backend", 
                    dest="use_pccl_cpp_backend",
                    action="store_false",
                    help="Disable the C++ backend in PCCL. Will fall back to the inefficient Python backend.")
    parser.add_argument("--test", 
                        action="store_true",
                        help="test for correctness")
    parser.add_argument("--pccl-recursive-alg", 
                        action="store_true",
                        help="Use the recursive doubling algorithm for PCCL.")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["fp32"], # TODO: support bf16 (currently reduce_scatter doesnt)
                        default="fp32")
    args = parser.parse_args()

    if args.use_pccl_cpp_backend:
        if dist.get_rank() == 0:
            build_pccl()
            MPI.COMM_WORLD.Barrier()
        else:
            MPI.COMM_WORLD.Barrier()
            build_pccl()
    
    gpu_count, slurm_job_id = get_gpu_counts_and_job_id()
    sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    unit = "MB"
    use_rd = args.pccl_recursive_alg 
    if args.library == "pccl":
        # creating 2D process groups for intra- and inter-node communication
        pg = ProcessGroups(args.num_gpus_per_node, 
                                   dist.get_world_size() // args.num_gpus_per_node)                           
        args.library += "_cpp" if args.use_pccl_cpp_backend else "_py"
        args.library += "_rec" if use_rd else "_ring"
        function = all_reduce_2D
    elif args.library == "mpi":
        pg = MPI.COMM_WORLD
        function = _all_reduce
    elif args.library == "xccl":
        pg = None # None is mapped to comm-world in torch.dist + xccl
        function = _all_reduce
    
    data_folder = f"./data/all_reduce/{args.machine}"
    os.makedirs(data_folder, exist_ok=True)

    csv_filename = os.path.join(data_folder,
                                f"gpus_{gpu_count}_slurm_{slurm_job_id}.csv")
    
    with open(csv_filename, "w") as f:
        if dist.get_rank() == 0:
            writer = csv.writer(f)
            # Write the header
            header = ["gpu_count", "slurm_job_id", "output_size", "unit", f"time_{args.library}"]
            writer.writerow(header)
            f.flush()

        for size in sizes:
            if dist.get_rank() == 0:
                print(f"output size = {size} {unit}")
            mult = 2**20 if unit == "MB" else 2**10
            dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
            device = "cuda"

            # calculate input and output numel 
            elem_size = 4
            numel = size * mult // elem_size
            input_buffer_numel = output_buffer_numel = numel

            # create input/output tensors
            output_tensor = torch.empty((output_buffer_numel,), dtype=dtype, device=device)
            input_tensor = torch.randn((input_buffer_numel,), dtype=dtype, device=device)

            kwargs = {}
            if args.library == "mpi":
                kwargs["directly_call_mpi"] = True
            kwargs["use_rh_and_rd"] = args.pccl_recursive_alg
            kwargs["use_pccl_cpp_backend"] = args.use_pccl_cpp_backend

            time = time_something(function, output_tensor, input_tensor, group=pg, **kwargs)

            # gold
            if args.test:
                output_tensor_gold = torch.empty((output_buffer_numel,), dtype=dtype, device=device)
                nccl_time = time_something(_all_reduce, output_tensor_gold, input_tensor)
                assert allclose(output_tensor, output_tensor_gold)
            
            if dist.get_rank() == 0:
                print(f"time_{args.library} = {time:.2f} ms")
                if args.test:
                    print(f"time_nccl = {nccl_time:.2f} ms")
            
            if dist.get_rank() == 0:
                print("===============================")
                writer.writerow([gpu_count, slurm_job_id, size, unit, time])
                f.flush()
        
        if args.test and dist.get_rank() == 0:
            print("All tests passed! PCCL outputs match NCCL outputs")
    dist.destroy_process_group()


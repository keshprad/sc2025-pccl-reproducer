#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC547
#SBATCH -t 00:10:00
#SBATCH --job-name=deepspeed_zero3_benchmark
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

# NOTE: IF ADDING ANY ENV VARIABLES HERE, ADD CORRESPONDING LINE TO setup_deepspeed_env.sh

# DeepSpeed ZeRO-3 Benchmark Runner
# Usage for single node: sbatch --export=MODEL_SIZE=7B run_deepspeed_benchmark.sh
# Usage for multi-node:  sbatch -N 2 --export=MODEL_SIZE=7B run_deepspeed_benchmark.sh
#                        sbatch -N 4 --export=MODEL_SIZE=13B run_deepspeed_benchmark.sh

PROJ_NAME="csc547"
export WRKSPC=/lustre/orion/$PROJ_NAME/scratch/$USER
VENV_NAME="pccl-venv"

module load cray-mpich/8.1.31
module load rocm/6.2.4
module load cpe/24.11
module load Core/24.00
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
module load craype-accel-amd-gfx90a
module load ninja
module list

# Load Cray OpenMP runtime libraries to fix CPU Adam extension
# export OMP_NUM_THREADS=8
# export CRAY_OMP_CHECK_AFFINITY=TRUE

# # Fix for CCE OpenMP fatal error: prevent PyTorch from changing thread settings
# export MKL_NUM_THREADS=8
# export OPENBLAS_NUM_THREADS=8
# export VECLIB_MAXIMUM_THREADS=8
# export NUMEXPR_NUM_THREADS=8
# # Tell PyTorch not to modify OpenMP thread settings at runtime
# export TORCH_SET_NUM_THREADS=8
# export KMP_DUPLICATE_LIB_OK=TRUE

# Add Cray compiler runtime libraries to LD_LIBRARY_PATH
# if [ -d "/opt/cray/pe/cce/18.0.1/cce/x86_64/lib" ]; then
#     export LD_LIBRARY_PATH="/opt/cray/pe/cce/18.0.1/cce/x86_64/lib:${LD_LIBRARY_PATH}"
#     echo "Added Cray CCE runtime library path"
# fi
# # Preload Cray OpenMP runtime if available
# CRAY_OMP_LIB="/opt/cray/pe/cce/18.0.1/cce/x86_64/lib/libcraymp.so"
# if [ -f "$CRAY_OMP_LIB" ]; then
#     export LD_PRELOAD="$CRAY_OMP_LIB:$LD_PRELOAD"
#     echo "Preloading Cray OpenMP library: $CRAY_OMP_LIB"
# fi

# Set proper compiler for CPU extensions (needs GCC 9+ compatibility)
export CXX=CC
export CC=cc
export CPP=cpp

source $WRKSPC/$VENV_NAME/bin/activate

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=8 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 

## pytorch dist variables
# For multi-node, use the first node in the allocation as master
export MASTER_ADDR=$(hostname -i)
echo "Master address is $MASTER_ADDR"
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0
export WORLD_SIZE=$GPUS

## some RCCL env variables
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL="PHB"
# Additional NCCL settings for multi-node
export NCCL_IB_DISABLE=0
# export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
## RCCL plugin
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WRKSPC/aws-ofi-rccl/lib"

# mpich gpu support
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_VERBOSE=1
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:0-1; 1:2-3; 2:4-5; 3:6-7"

# fi variables
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0 
export MPICH_OFI_CXI_COUNTER_VERBOSE=1
# collecting counter data
export MPICH_OFI_CXI_COUNTER_REPORT=5
export HSA_ENABLE_SDMA=0

# Prevent Python from modifying OpenMP settings at runtime
# export PYTHONPATH=$PYTHONPATH
# export PYTORCH_TENSORPIPE_INIT_METHOD_TIMEOUT=60000

MASK_0="0x00fe000000000000" # Cores 49-55
MASK_1="0xfe00000000000000" # Cores 57-64
MASK_2="0x0000000000fe0000" # Cores 17-23
MASK_3="0x00000000fe000000" # Cores 25-31
MASK_4="0x00000000000000fe" # Cores 1-7
MASK_5="0x000000000000fe00" # Cores 9-15
MASK_6="0x000000fe00000000" # Cores 33-39
MASK_7="0x0000fe0000000000" # Cores 41-47
CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"
export DEEPSPEED_SLURM_EXTRA_ARGS="-N $NNODES -n $GPUS --ntasks-per-node=8 --gpus-per-task=1 -c 7 ${CPU_MASK} --mem-bind=map_mem:3,3,1,1,0,0,2,2"

MODEL_SIZE="1B"
# Create output directory
OUTPUT_DIR="./benchmark_results"
mkdir -p $OUTPUT_DIR
HF_HOME=$WRKSPC/.cache/hf
# Set data directory (adjust path to your pre-tokenized data)
DATA_DIR="${HF_HOME}/datasets/openwebtext"

# Generate hostfile for multi-node training
HOSTFILE="benchmark_deepspeed_zero3/hostfile-$SLURM_JOB_ID"
if [ $NNODES -gt 1 ]; then
    echo "Generating hostfile for multi-node training..."
    scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE
    # Add slots (number of GPUs) for each host
    sed -i "s/$/ slots=$GPUS_PER_NODE/" $HOSTFILE
    echo "Generated hostfile:"
    cat $HOSTFILE
else
    echo "Single node training - hostfile not required"
fi
# Create .deepspeed_env for multi-node training
if [ $NNODES -gt 1 ]; then
    export DS_ENV_FILE=benchmark_deepspeed_zero3/.deepspeed_env
    source benchmark_deepspeed_zero3/setup_deepspeed_env.sh
fi

echo "Starting DeepSpeed ZeRO-3 Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Model Size: $MODEL_SIZE"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $GPUS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Output directory: $OUTPUT_DIR"
echo "Master address: $MASTER_ADDR"
echo "Using pre-tokenized data from: ${DATA_DIR}"
if [ ! -f "${DATA_DIR}/train.bin" ]; then
    echo "Error: Pre-tokenized data not found at ${DATA_DIR}/train.bin"
    echo "Please run the data preparation script first:"
    echo "  HF_HOME=\$SCRATCH/.cache/hf python dataset/openwebtext/prepare.py"
    exit 1
fi
# Run benchmark
echo "Starting benchmark for ${MODEL_SIZE} model"

# Configure DeepSpeed command based on number of nodes
if [ $NNODES -gt 1 ]; then
    echo "Running multi-node training with $NNODES nodes"
    # Launch DeepSpeed across all nodes
    # TODO: add cpu mask and mem binding
        deepspeed --launcher slurm \
                --hostfile=$HOSTFILE \
                --num_gpus=$GPUS \
                --num_nodes=$NNODES \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                benchmark_deepspeed_zero3/benchmark_deepspeed_zero3.py \
                --model_size $MODEL_SIZE \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR
else
    echo "Running single-node training"
    # Launch DeepSpeed across all nodes
    # TODO: add cpu mask and mem binding
    scripts/get_rank.sh \
        deepspeed --launcher slurm \
                --num_gpus=$GPUS \
                --num_nodes=$NNODES \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                benchmark_deepspeed_zero3/benchmark_deepspeed_zero3.py \
                --model_size $MODEL_SIZE \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR
fi

echo "Benchmark completed for ${MODEL_SIZE} model!"
echo "Results saved in: $OUTPUT_DIR"

# Cleanup setup files for multi-node training
if [ $NNODES -gt 1 ]; then
    rm $HOSTFILE
    rm $DS_ENV_FILE
fi


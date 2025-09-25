#!/bin/bash

## REPLACE WITH YOUR OLCF PROJECT NAME 
PROJ_NAME="lrn089"
rocm_version="6.4.1"
export ROCM_PATH="/opt/rocm-${rocm_version}/"

module load PrgEnv-cray
module load rocm/${rocm_version}
module load cray-mpich/8.1.32
module load cpe/25.03
module load craype-accel-amd-gfx90a
module load cray-python/3.11.7


export WRKSPC=/lustre/orion/$PROJ_NAME/scratch/$USER
mkdir -p $WRKSPC
cd $WRKSPC
ENV_NAME="pccl-venv"
ENV_LOC="$WRKSPC/$ENV_NAME"

# Setup Virtual Environment
echo "Setting up Virtual Environment"
# python -m venv ${ENV_LOC} --system-site-packages
python -m venv ${ENV_LOC}
. ${ENV_LOC}/bin/activate

pip install --upgrade pip

# PyTorch
echo "Installing PyTorch"
if [ "${rocm_version}" == 5.6.0  ]; then
	pip install --force-reinstall /lustre/orion/world-shared/stf007/msandov1/wheels/TorchROCm5.6/torch-2.1.2-cp310-cp310-linux_x86_64.whl
elif [ "${rocm_version}" == 6.0.0  ]; then
	pip3 install torch  --index-url https://download.pytorch.org/whl/rocm6.0
elif [ "${rocm_version}" == 5.7.0  ]; then
	pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
elif [ "${rocm_version}" == 6.2.4  ]; then
	pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.2.4
	pip install --upgrade numpy
elif [ "${rocm_version}" == 6.4.1  ]; then
	pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
	pip install --upgrade numpy
fi

# other pip dependencies
pip install --no-cache-dir  deepspeed
pip install --no-cache-dir  transformers datasets tiktoken wandb tqdm


# mpi4py
export MPICH_GPU_SUPPORT_ENABLED=1
INC="-I${ROCM_PATH}/include"
LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
#CC=cc
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
MPICC="cc -shared" INC=$INC LDFLAGS=$LDFLAGS pip install --upgrade --no-cache-dir --no-binary=mpi4py mpi4py


# AWS-OFI RCCL plugin
echo "Installing RCCL Plugin"
git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl 
cd aws-ofi-rccl
libfabric_path=/opt/cray/libfabric/1.22.0
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
PLUG_PREFIX=$PWD
CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
	--with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
	--prefix=$PLUG_PREFIX --with-hip=/opt/rocm-$rocm_version/ --with-mpi=$MPICH_DIR
make
make install
cd ..
tar -cvzf aws-ofi-rccl.tar.gz aws-ofi-rccl/

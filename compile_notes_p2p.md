# Compiling ofi_all_gather.cpp

We'll setup the repo like this:
```
sc2025-pccl-reproducer
├── ...
├── external
│   └── aws-ofi-rccl
├── ofi_all_gather
├── ofi_all_gather.cpp
├── ...
└── run_ofi_all_gather.sh
```

Load modules
---
```bash
module load cray-mpich/8.1.32
module load rocm/6.4.1
module load cpe/25.03
module load craype-accel-amd-gfx90a
```

aws-ofi-rccl setup
---
Clone aws-ofi-rccl to `./external/aws-ofi-rccl`:
```bash
mkdir external
cd external
git clone https://github.com/ROCm/aws-ofi-rccl
cd aws-ofi-rccl
```

**Small change to let it work with c++ compiler:**  
- in `tests/test-common.h`, change L180 from `return -1;` to `return ncclInvalidArgument;`. This is because I'm compiling with cpp instead of c.

Compile aws-ofi-rccl:

```bash
./autogen.sh
libfabric_path=/opt/cray/libfabric/1.22.0
rocm_version=6.2.4
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
--with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
--prefix=$PWD --with-hip=/opt/rocm-$rocm_version --with-mpi=$MPICH_DIR
make
make install
```

ofi_p2p.cpp
---
```bash
hipcc -fpermissive --offload-arch=gfx90a ofi_p2p.cpp -o ofi_p2p -Iexternal/aws-ofi-rccl/include -Iexternal/aws-ofi-rccl -I$ROCM_PATH/include -I$MPICH_DIR/include -I/opt/cray/libfabric/1.22.0/include -Lexternal/aws-ofi-rccl/lib -L/opt/cray/libfabric/1.22.0/lib64 -lfabric -L$MPICH_DIR/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}
```
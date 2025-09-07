# Makefile for OFI All-Gather implementation
# Make sure required modules are loaded prior to using Makefile

CXX = hipcc
CXXFLAGS = -fpermissive --offload-arch=gfx90a
INCLUDES = -I./external/aws-ofi-rccl/include -I./external/aws-ofi-rccl \
		   -I/opt/cray/libfabric/1.22.0/include \
           -I$(ROCM_PATH)/include -I$(MPICH_DIR)/include \
           -I./ofi_src
LIBS = -L./external/aws-ofi-rccl/lib \
	 -L/opt/cray/libfabric/1.22.0/lib64 -lfabric \
	 -L$(MPICH_DIR)/lib -lmpi \
	 -L$(ROCM_PATH)/lib -lrccl \
	 $(PE_MPICH_GTL_DIR_amd_gfx90a) $(PE_MPICH_GTL_LIBS_amd_gfx90a)
LOG_FLAGS = -DOFI_NCCL_WARN=1 -DOFI_NCCL_INFO=1 -DOFI_NCCL_TRACE=1

# Directories
SRCDIR = ofi_src
BUILDDIR = ofi_build

# Source files
SOURCES = ofi_all_gather.cpp ofi_distributed.cpp ofi_distributed_utils.cpp
OBJECTS = $(addprefix $(BUILDDIR)/, $(SOURCES:.cpp=.o))
TARGET = $(BUILDDIR)/ofi_all_gather

all: $(BUILDDIR) $(TARGET)

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(LOG_FLAGS) $(INCLUDES) -c $< -o $@

# Dependencies
$(BUILDDIR)/ofi_all_gather.o: $(SRCDIR)/ofi_all_gather.cpp $(SRCDIR)/ofi_all_gather.h $(SRCDIR)/ofi_distributed.h $(SRCDIR)/ofi_distributed_utils.h
$(BUILDDIR)/ofi_distributed.o: $(SRCDIR)/ofi_distributed.cpp $(SRCDIR)/ofi_distributed.h $(SRCDIR)/ofi_distributed_utils.h
$(BUILDDIR)/ofi_distributed_utils.o: $(SRCDIR)/ofi_distributed_utils.cpp $(SRCDIR)/ofi_distributed_utils.h

clean:
	rm -rf $(BUILDDIR)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the OFI all-gather program (default)"
	@echo "  clean   - Remove build directory and all build artifacts"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Build artifacts will be placed in: $(BUILDDIR)/"
	@echo "Source files are located in: $(SRCDIR)/"

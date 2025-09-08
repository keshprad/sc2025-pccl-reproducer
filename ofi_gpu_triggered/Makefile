# Makefile for GPU-triggered OFI queue implementation

CXX = hipcc
CXXFLAGS = -fpermissive --offload-arch=gfx90a -std=c++14
INCLUDES = -I$(ROCM_PATH)/include -I$(MPICH_DIR)/include
LIBS = -L$(MPICH_DIR)/lib -lmpi -L$(ROCM_PATH)/lib -lhip
LOG_FLAGS = -DDEBUG=1

# Directories
SRCDIR = .
BUILDDIR = build

# Source files
SOURCES = main.cpp
OBJECTS = $(addprefix $(BUILDDIR)/, $(SOURCES:.cpp=.o))
TARGET = $(BUILDDIR)/gpu_triggered_queue

all: $(BUILDDIR) $(TARGET)

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(LOG_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILDDIR)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the GPU-triggered queue program (default)"
	@echo "  clean   - Remove build directory and all build artifacts"
	@echo "  help    - Show this help message"

.PHONY: all clean help

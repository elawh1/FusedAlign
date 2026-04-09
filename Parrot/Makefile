CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = parrot
SRC = parrot.cpp
HDR = helper.h helper2.h fused_spmm.h accuracy.h

NVCC ?= nvcc
NVCCFLAGS = -O3 -std=c++17

CUDA_TARGET = parrot_cuda
CUDA_SRC = parrot_cuda.cu
CUDA_HDR = helper.h helper2.h fused_spmm.h accuracy.h randomwalk.cu

# ---- BLAS / MKL ----
# Default: OpenBLAS
LIBS = -lopenblas

# Optional: Intel MKL
# Usage:
#   export MKLROOT=/path/to/mkl
#   make
ifdef MKLROOT
CXXFLAGS += -march=native -fopenmp -DMKL_AVAILABLE -DMKL_ILP64
NVCCFLAGS += -Xcompiler -fopenmp -DMKL_AVAILABLE
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
NVLIBS = -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib \
		-lcusparse -lcublas
endif


all: $(TARGET) $(CUDA_TARGET)

$(TARGET): $(SRC) $(HDR) 
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

$(CUDA_TARGET): $(CUDA_SRC) $(CUDA_HDR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRC) -o $(CUDA_TARGET) $(LIBS) $(NVLIBS)

clean:
	rm -f $(TARGET) $(CUDA_TARGET)

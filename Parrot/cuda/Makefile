HDR = ../include/helper.h ../include/helper2.h ../include/fused_spmm.h ../include/accuracy.h

NVCC ?= nvcc
CUSPARSELT_DIR = $(HOME)/opt/cusparselt
NVCCFLAGS = -O3 -std=c++17 -rdc=true -Iinclude  -I../include \
    -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include \
    -I$(CUSPARSELT_DIR)/include

CUDA_TARGET = parrot_cuda
CUDA_SRC = src/parrot_cuda.cu src/randomwalk.cu src/sddmm.cu \
           src/helper.cu src/helper2.cu src/fused_spmm.cu src/accuracy.cu
CUDA_OBJS = $(CUDA_SRC:.cu=.o)
CUDA_HDR = include/cuda_helper.cuh

ifdef MKLROOT
NVCCFLAGS += -I$(MKLROOT)/include -Xcompiler -fopenmp -DMKL_AVAILABLE -DMKL_ILP64
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
endif

CUSPARSELT_LIBDIR = $(CUSPARSELT_DIR)/lib

NVLIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib \
         -L$(CUSPARSELT_LIBDIR) \
         -Xlinker -rpath -Xlinker $(CUSPARSELT_LIBDIR) \
         -lcusparseLt -lcusparse -lcublas

all: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_OBJS) -o $(CUDA_TARGET) $(LIBS) $(NVLIBS)

%.o: %.cu $(CUDA_HDR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CUDA_TARGET) $(CUDA_OBJS)

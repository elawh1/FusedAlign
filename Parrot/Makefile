CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = parrot
SRC = parrot.cpp
HDR = helper.h helper2.h fused_spmm.h accuracy.h

CPP_SRCS = helper.cpp helper2.cpp fused_spmm.cpp
CPP_OBJS = $(CPP_SRCS:.cpp=.o)

NVCC ?= nvcc
CUSPARSELT_DIR = $(HOME)/opt/cusparselt
NVCCFLAGS = -O3 -std=c++17 -rdc=true \
    -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include \
    -I$(CUSPARSELT_DIR)/include

CUDA_TARGET = parrot_cuda
CUDA_SRC = parrot_cuda.cu randomwalk.cu sddmm.cu
CUDA_OBJS = $(CUDA_SRC:.cu=.o)
CUDA_HDR = helper.h helper2.h fused_spmm.h accuracy.h cuda_helper.cuh

LIBS = -lopenblas

ifdef MKLROOT
CXXFLAGS += -march=native -fopenmp -DMKL_AVAILABLE -DMKL_ILP64
NVCCFLAGS += -Xcompiler -fopenmp -DMKL_AVAILABLE -DMKL_ILP64
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
endif

CUSPARSELT_LIBDIR = $(CUSPARSELT_DIR)/lib

NVLIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib \
         -L$(CUSPARSELT_LIBDIR) \
         -Xlinker -rpath -Xlinker $(CUSPARSELT_LIBDIR) \
         -lcusparseLt -lcusparse -lcublas

all: $(TARGET) $(CUDA_TARGET)

$(TARGET): $(SRC) $(CPP_OBJS) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) $(CPP_OBJS) -o $(TARGET) $(LIBS)

$(CUDA_TARGET): $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_OBJS) $(CPP_OBJS) -o $(CUDA_TARGET) $(LIBS) $(NVLIBS)

%.o: %.cpp $(HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu $(CUDA_HDR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(CUDA_TARGET) $(CUDA_OBJS) $(CPP_OBJS)
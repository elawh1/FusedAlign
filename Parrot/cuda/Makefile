NVCC ?= nvcc
ELEMENT_TYPE ?= double
NVCCFLAGS = -O3 -std=c++17 -rdc=true -Iinclude \
    -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include

ifeq ($(ELEMENT_TYPE),float)
NVCCFLAGS += -DCUDA_ELEMENT_FLOAT
else ifeq ($(ELEMENT_TYPE),double)
NVCCFLAGS += -DCUDA_ELEMENT_DOUBLE
else
$(error ELEMENT_TYPE must be 'float' or 'double')
endif

ifeq ($(DEBUG),1)
NVCCFLAGS += -DDEBUG
endif

CUDA_TARGET = cuda
CUDA_SRC = src/main.cu src/randomwalk.cu src/sddmm.cu \
           src/helper.cu src/helper2.cu src/optimization.cu src/accuracy.cu \
           src/memory_tracker.cu
CUDA_OBJS = $(CUDA_SRC:.cu=.o)
CUDA_HDR = include/cuda_helper.cuh
BENCH_TARGET = spmm_ell_bench
ARXIV_SPMM_BENCH_TARGET = arxiv_spmm_format_bench

NVLIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib \
         -lcusparse -lcublas

BENCH_FLAGS = -O3 -std=c++17 \
    -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include

.PHONY: all bench arxiv-spmm-bench clean clean-bench distclean

all: $(CUDA_TARGET)

bench: $(BENCH_TARGET)

arxiv-spmm-bench: $(ARXIV_SPMM_BENCH_TARGET)

$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_OBJS) -o $(CUDA_TARGET) $(NVLIBS)

%.o: %.cu $(CUDA_HDR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BENCH_TARGET): src/spmm_ell_benchmark.cu
	$(NVCC) $(BENCH_FLAGS) src/spmm_ell_benchmark.cu -o $@ $(NVLIBS)

$(ARXIV_SPMM_BENCH_TARGET): src/arxiv_spmm_format_benchmark.cu src/helper.cu src/helper2.cu src/memory_tracker.cu $(CUDA_HDR)
	$(NVCC) $(NVCCFLAGS) src/arxiv_spmm_format_benchmark.cu src/helper.cu src/helper2.cu src/memory_tracker.cu -o $@ $(NVLIBS)

clean:
	rm -f $(CUDA_TARGET) $(CUDA_OBJS)

clean-bench:
	rm -f $(BENCH_TARGET) $(ARXIV_SPMM_BENCH_TARGET)

distclean: clean clean-bench

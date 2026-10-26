CXX ?= g++
ELEMENT_TYPE ?= float

MKLROOT ?= /opt/intel/oneapi/mkl/latest

CXXFLAGS = -O3 -std=c++17 -Iinclude -fopenmp -I$(MKLROOT)/include

ifeq ($(ELEMENT_TYPE),float)
CXXFLAGS += -DCUDA_ELEMENT_FLOAT
else ifeq ($(ELEMENT_TYPE),double)
CXXFLAGS += -DCUDA_ELEMENT_DOUBLE
else
$(error ELEMENT_TYPE must be 'float' or 'double')
endif

ifeq ($(DEBUG),1)
CXXFLAGS += -DDEBUG
endif

TARGET = mkl_sparse_alignment
SRC = src/main.cpp src/randomwalk.cpp src/dense_intra.cpp \
      src/helper.cpp src/helper2.cpp src/optimization.cpp src/accuracy.cpp
OBJS = $(SRC:.cpp=.o)
HDR = include/cuda_helper.cuh include/accuracy.h

MKL_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
           -lgomp -lpthread -lm -ldl

.PHONY: all clean distclean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -fopenmp $(OBJS) -o $(TARGET) $(MKL_LIBS)

%.o: %.cpp $(HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) cuda_dense_alignment cuda_sparse_alignment $(OBJS) src/memory_tracker.o src/fused_spmm.o

distclean: clean

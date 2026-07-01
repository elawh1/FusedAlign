CXX ?= g++
ELEMENT_TYPE ?= float
CXXFLAGS = -O3 -std=c++17 -Wall -Iinclude

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

TARGET = mkl_dense_alignment
SRC = src/main.cu src/randomwalk.cu src/dense_intra.cu \
      src/helper.cu src/helper2.cu src/optimization.cu src/accuracy.cu
OBJ = $(SRC:.cu=.o)
HDR = include/cuda_helper.cuh include/accuracy.h

ifndef MKLROOT
$(error MKLROOT must be set for the MKL implementation)
endif

CXXFLAGS += -march=native -fopenmp -DMKL_AVAILABLE -DMKL_ILP64 -I$(MKLROOT)/include
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl

.PHONY: all clean distclean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(LIBS)

%.o: %.cu $(HDR)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

clean:
	rm -f $(TARGET) cuda_dense_alignment $(OBJ) src/memory_tracker.o

distclean: clean

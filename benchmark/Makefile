CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = spmm_col
SRC =  spmm_col.cpp
HDR = helper2.h spmm.h mtx.h
# ---- BLAS / MKL ----
# Default: OpenBLAS
LIBS = -lopenblas

# Optional: Intel MKL
# Usage:
#   export MKLROOT=/path/to/mkl
#   make
ifdef MKLROOT
CXXFLAGS += -march=native -fopenmp -DMKL_AVAILABLE -DMKL_ILP64
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
endif


all: $(TARGET)

$(TARGET): $(SRC) $(HDR) 
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)


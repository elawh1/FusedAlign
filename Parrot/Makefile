CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = parrot
SRC = parrot.cpp
HDR = helper.h helper2.h fused_spmm.h accuracy.h

# ---- BLAS / MKL ----
# Default: OpenBLAS
LIBS = -lopenblas

# Optional: Intel MKL
# Usage:
#   export MKLROOT=/path/to/mkl
#   make
ifdef MKLROOT
CXXFLAGS += -fopenmp -DMKL_AVAILABLE
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_lp64 -lmkl_core -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
endif


all: $(TARGET)

$(TARGET): $(SRC) $(HDR) 
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

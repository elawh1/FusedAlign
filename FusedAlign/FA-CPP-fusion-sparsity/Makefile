CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall -Iinclude

ifeq ($(DEBUG),1)
CXXFLAGS += -DDEBUG
endif

TARGET = fusedalign
SRC = src/fusedalign.cpp src/helper.cpp src/helper2.cpp src/fused_spmm.cpp src/accuracy.cpp
HDR = include/helper.h include/helper2.h include/fused_spmm.h include/accuracy.h include/debug_log.h

# ---- MKL ----
ifndef MKLROOT
$(error MKLROOT must be set for the MKL implementation)
endif

CXXFLAGS += -march=native -fopenmp -DMKL_AVAILABLE -DMKL_ILP64 -I$(MKLROOT)/include
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall -Iinclude

ifeq ($(DEBUG),1)
CXXFLAGS += -DDEBUG
endif

TARGET = fusedalign
SRC = src/fusedalign.cpp
HDR =

# ---- MKL ----
ifndef MKLROOT
$(error MKLROOT must be set for the MKL dense implementation)
endif

CXXFLAGS += -march=native -fopenmp -DMKL_AVAILABLE -DMKL_ILP64 -I$(MKLROOT)/include
LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

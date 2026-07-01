CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall -Iinclude -I$(MKLROOT)/include \
           -march=native -fopenmp -DMKL_AVAILABLE -DMKL_ILP64

ifeq ($(DEBUG),1)
CXXFLAGS += -DDEBUG
endif

TARGET = fastalign
SRC = src/fastalign.cpp src/helper.cpp src/helper2.cpp src/accuracy.cpp
HDR = include/helper.h include/helper2.h include/accuracy.h include/debug_log.h

ifndef MKLROOT
$(error MKLROOT is not set; source the Intel oneAPI/MKL environment before building)
endif

LIBS = -L$(MKLROOT)/lib/intel64 \
       -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

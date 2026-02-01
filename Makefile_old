CXX = icpc
# CXX = mpic++
CXXFLAGS = -O2 -qmkl
# MATLAB_INC = $(MATLABROOT)/extern/include
# MATLAB_LIB = $(MATLABROOT)/bin/glnxa64

parrot: parrot.cpp helper.h
	$(CXX) $(CXXFLAGS) parrot.cpp -o parrot \
		# -I$(MATLAB_INC) \
		# -L$(MATLAB_LIB) -Wl,-rpath,$(MATLAB_LIB) \
		# -lmat
		# -lmx

load:
	module load intel/2022a
	module load Matlab

clean:
	rm -f parrot
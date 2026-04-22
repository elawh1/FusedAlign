// #include <iostream>
// #include <vector>
// #include <mkl.h>
// #ifdef MKL_AVAILABLE
// #include <mkl.h>
// #include <mkl_vml.h>
// #include <omp.h>
// #include <set>
// #include <map>
// #endif
// #include <string>
// #include <sstream>
// #include <algorithm>
// #include <fstream>
// #include <ctime> 
// #include <chrono>
// #include <cstdio>
// #include <cinttypes>  
// #include <random>
// #include <cmath> 
// #include <cstddef>
// #include <cstring> // for memset
// #include <cassert>
// #include <unordered_set>
// #include <iostream>
// #include <stdexcept>
// #include <thread>
#include "fused_spmm.h"

// Given: A (n1 x n1), B (n1 x n2), C (n2 x n2), D (n1 x n2), block_size
// All dense matrices are column-major.
// A and C are sparse_matrix_t (MKL CSR).
// B and D are std::vector<double>.
// For each block of columns in n2, generate E_col (n1 x 1) and D_col (n1 x 1)
std::vector<double> blockwise_colsum_matmul( sparse_matrix_t A, sparse_matrix_t C, const std::vector<double>& B, size_t n1, size_t n2, int n_blocks, double scalar) ;

// Given: A (n1 x n1), B (n1 x n2), C (n2 x n2), D (n1 x n2), block_size
// All dense matrices are column-major.
// A and C are sparse_matrix_t (MKL CSR).
// B and D are std::vector<double>.
// For each block of columns in n2, generate E_col (n1 x 1) and D_col (n1 x 1)
std::vector<double> blockwise_colsum_matmul_2( sparse_matrix_t A, sparse_matrix_t C, const std::vector<double>& B, size_t n1, size_t n2, int n_blocks) ;

std::vector<double> blockwise_outputcols_matmul( sparse_matrix_t A,  sparse_matrix_t C,  const std::vector<double>& B, 
    size_t n1, size_t n2, int n_blocks, double scalar) ;



#pragma once

#ifdef MKL_AVAILABLE
#include <mkl.h>
#include <omp.h>
#include <mkl_vml.h>
#endif
#include <cstdio>
#include <cinttypes>   
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <numeric>
#include <set>

#include <cstdlib>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <cuda_fp16.h>
#include <cusparseLt.h>

#include <stdexcept>
#include <cstdint>


#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t status_ = (call);                                            \
        if (status_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(status_));            \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUSPARSE(call)                                                     \
    do {                                                                         \
        cusparseStatus_t status_ = (call);                                       \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, static_cast<int>(status_));              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
do {                                                                           \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__         \
                  << " -> status " << static_cast<int>(st__) << std::endl;     \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

#define CHECK_CUSPARSELT(func) {                                               \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "cuSPARSELt Error: " << status << " at line " << __LINE__ \
                  << std::endl;                                                \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// randomwalk.cu
void get_batch_rwr_scores_cusparse(sparse_matrix_t graph, std::vector<MKL_INT> anchors, 
                                    MKL_INT n, MKL_INT avg_nnz_per_n, bool directed, 
                                    float restart_prob, int max_iters, bool connect_isolated, 
                                    std::vector<double>& Xnext_vals);

std::vector<double> get_product_rwr_mat_cusparse( sparse_matrix_t graph1_normalized, 
                                                    sparse_matrix_t graph2_normalized, 
                                                    const std::vector<double>& in_cross_dist, 
                                                    int n1, int n2, double rwr_restart_prob, 
                                                    double gamma, int max_iters = 50, 
                                                    double tol = 1e-2);

//sddmm.cu
void masked_sddmm_intra_cost_cusparse( std::vector<double>& emb, sparse_matrix_t mask, 
                                        int n, int L, sparse_matrix_t& result);

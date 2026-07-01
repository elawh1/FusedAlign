#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <type_traits>

#include <mkl.h>
#include <omp.h>

#ifdef DEBUG
#define DEBUG_ONLY(stmt) do { stmt; } while (0)
#else
#define DEBUG_ONLY(stmt) do { } while (0)
#endif

#if defined(CUDA_ELEMENT_FLOAT) && defined(CUDA_ELEMENT_DOUBLE)
#error "Define only one of CUDA_ELEMENT_FLOAT or CUDA_ELEMENT_DOUBLE"
#endif

#if defined(CUDA_ELEMENT_DOUBLE)
using element_type = double;
constexpr const char* CUDA_ELEMENT_TYPE_NAME = "double";
#define CUDA_USE_FLOAT 0
#else
using element_type = float;
constexpr const char* CUDA_ELEMENT_TYPE_NAME = "float";
#define CUDA_USE_FLOAT 1
#endif

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind {
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyHostToHost
};

inline const char* cudaGetErrorString(cudaError_t) { return "host MKL compatibility error"; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    return (*ptr || bytes == 0) ? cudaSuccess : 1;
}
inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind) {
    if (bytes != 0) std::memmove(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* dst, int value, size_t bytes) {
    std::memset(dst, value, bytes);
    return cudaSuccess;
}

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t status_ = (call);                                            \
        if (status_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "host memory error at %s:%d\n", __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

using cublasStatus_t = int;
using cublasHandle_t = void*;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };

inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    *handle = reinterpret_cast<void*>(1);
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t st__ = (call);                                            \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::cerr << "MKL BLAS compatibility error at " << __FILE__ << ":"  \
                      << __LINE__ << " -> status " << st__ << std::endl;         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

inline CBLAS_TRANSPOSE to_cblas_trans(cublasOperation_t op) {
    return op == CUBLAS_OP_T ? CblasTrans : CblasNoTrans;
}

inline cublasStatus_t cuda_cublas_axpy(cublasHandle_t, int n, const element_type* alpha,
                                       const element_type* x, int incx, element_type* y, int incy) {
#if CUDA_USE_FLOAT
    cblas_saxpy(n, *alpha, x, incx, y, incy);
#else
    cblas_daxpy(n, *alpha, x, incx, y, incy);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cuda_cublas_scal(cublasHandle_t, int n, const element_type* alpha,
                                       element_type* x, int incx) {
#if CUDA_USE_FLOAT
    cblas_sscal(n, *alpha, x, incx);
#else
    cblas_dscal(n, *alpha, x, incx);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cuda_cublas_gemm(cublasHandle_t,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const element_type* alpha,
                                       const element_type* A, int lda,
                                       const element_type* B, int ldb,
                                       const element_type* beta,
                                       element_type* C, int ldc) {
#if CUDA_USE_FLOAT
    cblas_sgemm(CblasColMajor, to_cblas_trans(transa), to_cblas_trans(transb),
                m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
#else
    cblas_dgemm(CblasColMajor, to_cblas_trans(transa), to_cblas_trans(transb),
                m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cuda_cublas_geam(cublasHandle_t,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n,
                                       const element_type* alpha,
                                       const element_type* A, int lda,
                                       const element_type* beta,
                                       const element_type* B, int ldb,
                                       element_type* C, int ldc) {
    const bool use_a = *alpha != static_cast<element_type>(0);
    const bool use_b = *beta != static_cast<element_type>(0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            element_type val = static_cast<element_type>(0);
            if (use_a) {
                const element_type av = (transa == CUBLAS_OP_T)
                    ? A[static_cast<size_t>(row) * lda + col]
                    : A[static_cast<size_t>(col) * lda + row];
                val += (*alpha) * av;
            }
            if (use_b) {
                const element_type bv = (transb == CUBLAS_OP_T)
                    ? B[static_cast<size_t>(row) * ldb + col]
                    : B[static_cast<size_t>(col) * ldb + row];
                val += (*beta) * bv;
            }
            C[static_cast<size_t>(col) * ldc + row] = val;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cuda_cublas_dot(cublasHandle_t, int n,
                                      const element_type* x, int incx,
                                      const element_type* y, int incy,
                                      element_type* result) {
#if CUDA_USE_FLOAT
    *result = cblas_sdot(n, x, incx, y, incy);
#else
    *result = cblas_ddot(n, x, incx, y, incy);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cuda_cublas_asum(cublasHandle_t, int n,
                                       const element_type* x, int incx,
                                       element_type* result) {
#if CUDA_USE_FLOAT
    *result = cblas_sasum(n, x, incx);
#else
    *result = cblas_dasum(n, x, incx);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

struct CudaDenseMatrix {
    int rows = 0;
    int cols = 0;
    element_type* d_vals = nullptr;
};

struct BatchRwrContext {
    element_type* d_x = nullptr;
    element_type* d_xnext = nullptr;
    int* d_anchors = nullptr;
    size_t dense_capacity = 0;
    size_t anchor_capacity = 0;
    int current_n = 0;
    int current_l = 0;
};

inline void cuda_memtracker_reset() {}
inline void cuda_memtracker_report_peak(const char* label = nullptr) { (void)label; }
inline cudaError_t cuda_malloc_tracked(void** ptr, size_t bytes, const char*, int) {
    return cudaMalloc(ptr, bytes);
}
inline cudaError_t cuda_free_tracked(void* ptr, const char*, int) {
    return cudaFree(ptr);
}

#define CUDA_MALLOC(ptr, bytes) \
    cuda_malloc_tracked(reinterpret_cast<void**>(ptr), static_cast<size_t>(bytes), __FILE__, __LINE__)

#define CUDA_FREE(ptr) \
    cuda_free_tracked(reinterpret_cast<void*>(ptr), __FILE__, __LINE__)

template <typename T>
inline void cuda_free_and_null(T*& ptr) {
    if (ptr != nullptr) {
        CHECK_CUDA(CUDA_FREE(ptr));
        ptr = nullptr;
    }
}

void destroy_cuda_dense(CudaDenseMatrix& matrix);
CudaDenseMatrix make_cuda_dense_from_edgelist(const std::vector<element_type>& edgelist, int m, int n);
CudaDenseMatrix transpose_cuda_dense(const CudaDenseMatrix& matrix);
CudaDenseMatrix row_normalize_graph_cuda(const CudaDenseMatrix& graph, int avg_edges_per_node);
void destroy_batch_rwr_context(BatchRwrContext& context);

void load_config(const std::string& data_path, int& g1_num_nodes, int& g2_num_nodes,
                 bool& use_attr, bool& directed, int& inner_iters, int& outer_iters);
std::vector<element_type> readFileToVector(const std::string& filename, bool directed);
std::pair<std::vector<int>, std::vector<int>> get_anchor_pairs(const std::vector<element_type>& H_vec);
void get_batch_rwr_scores_dense_device(const CudaDenseMatrix& graph, std::vector<int> anchors,
                                       int n, element_type restart_prob, int max_iters,
                                       element_type* d_X_out, BatchRwrContext& context);
void get_product_rwr_mat_dense_device(const CudaDenseMatrix& graph1_normalized,
                                      const CudaDenseMatrix& graph2_normalized,
                                      const element_type* d_in_cross_dist,
                                      int n1, int n2, element_type rwr_restart_prob,
                                      element_type gamma, element_type* d_out_cross_dist,
                                      int max_iters = 50, element_type tol = static_cast<element_type>(1e-2));

CudaDenseMatrix dense_intra_cost_device(const element_type* d_emb,
                                        const CudaDenseMatrix& mask,
                                        int n, int L);

void l2_norm_inplace_device(element_type* d_rwr, size_t n, size_t d);
void get_normalized_neg_exp_dist_device(const element_type* d_rwr1, const element_type* d_rwr2,
                                        int n1, int n2, int l, element_type* d_cross_rwr_dist);
void anchor_cost_zero_device(element_type* d_A, int nrows, int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             element_type val);
std::vector<element_type> con_prox_pt_opt_device(const CudaDenseMatrix& row_norm_adj1,
                                                 const CudaDenseMatrix& row_norm_adj2,
                                                 int n1, int n2, int avg_edges1_per_n1, int avg_edges2_per_n2,
                                                 const element_type* d_cross_dist,
                                                 const CudaDenseMatrix& intra_dist1_dense,
                                                 const CudaDenseMatrix& intra_dist2_dense,
                                                 int inner_iters, int outer_iters, element_type lambda_n, element_type lambda_a,
                                                 element_type lambda_p, element_type lambda_e_in,
                                                 const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs,
                                                 bool sparsify);

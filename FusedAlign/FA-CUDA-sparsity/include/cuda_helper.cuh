#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <type_traits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#ifdef DEBUG
#define DEBUG_ONLY(stmt) \
    do {                 \
        stmt;            \
    } while (0)
#else
#define DEBUG_ONLY(stmt) \
    do {                 \
    } while (0)
#endif

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t status_ = (call);                                            \
        if (status_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(status_));            \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
do {                                                                             \
    cublasStatus_t st__ = (call);                                                \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                         \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__          \
                  << " -> status " << static_cast<int>(st__) << std::endl;      \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

#define CHECK_CUSPARSE(call)                                                     \
do {                                                                             \
    cusparseStatus_t st__ = (call);                                              \
    if (st__ != CUSPARSE_STATUS_SUCCESS) {                                       \
        std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__        \
                  << " -> status " << static_cast<int>(st__) << std::endl;      \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

#if defined(CUDA_ELEMENT_FLOAT) && defined(CUDA_ELEMENT_DOUBLE)
#error "Define only one of CUDA_ELEMENT_FLOAT or CUDA_ELEMENT_DOUBLE"
#endif

#if defined(CUDA_ELEMENT_FLOAT)
using element_type = float;
constexpr cudaDataType CUDA_R_TYPE = CUDA_R_32F;
constexpr const char* CUDA_ELEMENT_TYPE_NAME = "float";
#define CUDA_USE_FLOAT 1
#else
using element_type = double;
constexpr cudaDataType CUDA_R_TYPE = CUDA_R_64F;
constexpr const char* CUDA_ELEMENT_TYPE_NAME = "double";
#define CUDA_USE_FLOAT 0
#endif

inline cublasStatus_t cuda_cublas_axpy(cublasHandle_t handle, int n, const element_type* alpha,
                                         const element_type* x, int incx, element_type* y, int incy) {
#if CUDA_USE_FLOAT
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
#else
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
#endif
}

inline cublasStatus_t cuda_cublas_scal(cublasHandle_t handle, int n, const element_type* alpha,
                                         element_type* x, int incx) {
#if CUDA_USE_FLOAT
        return cublasSscal(handle, n, alpha, x, incx);
#else
        return cublasDscal(handle, n, alpha, x, incx);
#endif
}

inline cublasStatus_t cuda_cublas_gemm(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const element_type* alpha,
                                         const element_type* A, int lda,
                                         const element_type* B, int ldb,
                                         const element_type* beta,
                                         element_type* C, int ldc) {
#if CUDA_USE_FLOAT
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline cublasStatus_t cuda_cublas_geam(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n,
                                         const element_type* alpha,
                                         const element_type* A, int lda,
                                         const element_type* beta,
                                         const element_type* B, int ldb,
                                         element_type* C, int ldc) {
#if CUDA_USE_FLOAT
        return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
        return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
}

inline cublasStatus_t cuda_cublas_dot(cublasHandle_t handle, int n,
                                        const element_type* x, int incx,
                                        const element_type* y, int incy,
                                        element_type* result) {
#if CUDA_USE_FLOAT
        return cublasSdot(handle, n, x, incx, y, incy, result);
#else
        return cublasDdot(handle, n, x, incx, y, incy, result);
#endif
}

inline cublasStatus_t cuda_cublas_asum(cublasHandle_t handle, int n,
                                         const element_type* x, int incx,
                                         element_type* result) {
#if CUDA_USE_FLOAT
        return cublasSasum(handle, n, x, incx, result);
#else
        return cublasDasum(handle, n, x, incx, result);
#endif
}

struct CudaDenseMatrix {
    int rows = 0;
    int cols = 0;
    element_type* d_vals = nullptr;
};

struct CudaSparseMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    element_type* d_vals = nullptr;
};

struct BatchRwrContext {
    cublasHandle_t cublas = nullptr;
    cusparseHandle_t cusparse = nullptr;
    element_type* d_x = nullptr;
    element_type* d_xnext = nullptr;
    int* d_anchors = nullptr;
    size_t dense_capacity = 0;
    size_t anchor_capacity = 0;
    int current_n = 0;
    int current_l = 0;
};

#ifdef DEBUG
void cuda_memtracker_reset();
void cuda_memtracker_report_peak(const char* label = nullptr);
cudaError_t cuda_malloc_tracked(void** ptr, size_t bytes, const char* file, int line);
cudaError_t cuda_free_tracked(void* ptr, const char* file, int line);
#else
inline void cuda_memtracker_reset() {}
inline void cuda_memtracker_report_peak(const char* label = nullptr) {
    (void)label;
}
inline cudaError_t cuda_malloc_tracked(void** ptr, size_t bytes, const char* file, int line) {
    (void)file;
    (void)line;
    return cudaMalloc(ptr, bytes);
}
inline cudaError_t cuda_free_tracked(void* ptr, const char* file, int line) {
    (void)file;
    (void)line;
    return cudaFree(ptr);
}
#endif

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
CudaSparseMatrix make_cuda_sparse_from_edgelist(const std::vector<element_type>& edgelist, int m, int n);
CudaDenseMatrix make_cuda_dense_from_edgelist(const std::vector<element_type>& edgelist, int m, int n);
CudaDenseMatrix transpose_cuda_dense(const CudaDenseMatrix& matrix);
void destroy_cuda_sparse(CudaSparseMatrix& matrix);
CudaSparseMatrix transpose_cuda_sparse(const CudaSparseMatrix& matrix);
CudaSparseMatrix row_normalize_graph_cuda(const CudaSparseMatrix& graph, int avg_edges_per_node);
void cuda_sparse_spmm(cusparseHandle_t handle,
                      const CudaSparseMatrix& sparse,
                      cusparseOperation_t op_sparse,
                      const element_type* d_dense_b,
                      int b_rows,
                      int b_cols,
                      int ldb,
                      cusparseOrder_t b_order,
                      element_type beta,
                      element_type* d_dense_c,
                      int c_rows,
                      int c_cols,
                      int ldc,
                      cusparseOrder_t c_order,
                      element_type alpha = static_cast<element_type>(1.0));
void destroy_batch_rwr_context(BatchRwrContext& context);

void load_config(const std::string& data_path, int& g1_num_nodes, int& g2_num_nodes,
                 bool& use_attr, bool& directed, int& inner_iters, int& outer_iters);
std::vector<element_type> readFileToVector(const std::string& filename, bool directed);
std::pair<std::vector<int>, std::vector<int>> get_anchor_pairs(const std::vector<element_type>& H_vec);
void get_batch_rwr_scores_sparse_device(const CudaSparseMatrix& graph, std::vector<int> anchors,
                                       int n, element_type restart_prob, int max_iters,
                                       element_type* d_X_out, BatchRwrContext& context);
void get_product_rwr_mat_sparse_device(const CudaSparseMatrix& graph1_normalized,
                                      const CudaSparseMatrix& graph2_normalized,
                                      const element_type* d_in_cross_dist,
                                      int n1, int n2, element_type rwr_restart_prob,
                                      element_type gamma, element_type* d_out_cross_dist,
                                      int max_iters = 50, element_type tol = static_cast<element_type>(1e-2));

CudaDenseMatrix dense_intra_cost_device(const element_type* d_emb,
                                        const CudaSparseMatrix& mask,
                                        int n, int L);

void l2_norm_inplace_device(element_type* d_rwr, size_t n, size_t d);
void get_normalized_neg_exp_dist_device(const element_type* d_rwr1, const element_type* d_rwr2,
                                        int n1, int n2, int l, element_type* d_cross_rwr_dist);
void anchor_cost_zero_device(element_type* d_A, int nrows, int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             element_type val);
std::vector<element_type> con_prox_pt_opt_device(const CudaSparseMatrix& row_norm_adj1,
                                                 const CudaSparseMatrix& row_norm_adj2,
                                                 int n1, int n2, int avg_edges1_per_n1, int avg_edges2_per_n2,
                                                 const element_type* d_cross_dist,
                                                 const CudaDenseMatrix& intra_dist1_dense,
                                                 const CudaDenseMatrix& intra_dist2_dense,
                                                 int inner_iters, int outer_iters, element_type lambda_n, element_type lambda_a,
                                                 element_type lambda_p, element_type lambda_e_in,
                                                 const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs,
                                                 bool sparsify);

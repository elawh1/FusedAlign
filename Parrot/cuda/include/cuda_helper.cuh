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
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusparseLt.h>

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

#define CHECK_CUSPARSE(call)                                                     \
    do {                                                                         \
        cusparseStatus_t status_ = (call);                                       \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, static_cast<int>(status_));              \
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

#define CHECK_CUSPARSELT(func) {                                                 \
    cusparseStatus_t status = (func);                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                     \
        std::cerr << "cuSPARSELt Error: " << status << " at line " << __LINE__  \
                  << std::endl;                                                  \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

#if defined(PARROT_ELEMENT_FLOAT) && defined(PARROT_ELEMENT_DOUBLE)
#error "Define only one of PARROT_ELEMENT_FLOAT or PARROT_ELEMENT_DOUBLE"
#endif

#if defined(PARROT_ELEMENT_FLOAT)
using element_type = float;
constexpr cudaDataType PARROT_CUDA_R_TYPE = CUDA_R_32F;
constexpr const char* PARROT_ELEMENT_TYPE_NAME = "float";
#define PARROT_USE_FLOAT 1
#else
using element_type = double;
constexpr cudaDataType PARROT_CUDA_R_TYPE = CUDA_R_64F;
constexpr const char* PARROT_ELEMENT_TYPE_NAME = "double";
#define PARROT_USE_FLOAT 0
#endif

inline cublasStatus_t parrot_cublas_axpy(cublasHandle_t handle, int n, const element_type* alpha,
                                         const element_type* x, int incx, element_type* y, int incy) {
#if PARROT_USE_FLOAT
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
#else
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
#endif
}

inline cublasStatus_t parrot_cublas_scal(cublasHandle_t handle, int n, const element_type* alpha,
                                         element_type* x, int incx) {
#if PARROT_USE_FLOAT
        return cublasSscal(handle, n, alpha, x, incx);
#else
        return cublasDscal(handle, n, alpha, x, incx);
#endif
}

inline cublasStatus_t parrot_cublas_gemm(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const element_type* alpha,
                                         const element_type* A, int lda,
                                         const element_type* B, int ldb,
                                         const element_type* beta,
                                         element_type* C, int ldc) {
#if PARROT_USE_FLOAT
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline cublasStatus_t parrot_cublas_geam(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n,
                                         const element_type* alpha,
                                         const element_type* A, int lda,
                                         const element_type* beta,
                                         const element_type* B, int ldb,
                                         element_type* C, int ldc) {
#if PARROT_USE_FLOAT
        return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
        return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
}

inline cublasStatus_t parrot_cublas_dot(cublasHandle_t handle, int n,
                                        const element_type* x, int incx,
                                        const element_type* y, int incy,
                                        element_type* result) {
#if PARROT_USE_FLOAT
        return cublasSdot(handle, n, x, incx, y, incy, result);
#else
        return cublasDdot(handle, n, x, incx, y, incy, result);
#endif
}

inline cublasStatus_t parrot_cublas_asum(cublasHandle_t handle, int n,
                                         const element_type* x, int incx,
                                         element_type* result) {
#if PARROT_USE_FLOAT
        return cublasSasum(handle, n, x, incx, result);
#else
        return cublasDasum(handle, n, x, incx, result);
#endif
}

inline cublasStatus_t parrot_cublas_iamax(cublasHandle_t handle, int n,
                                          const element_type* x, int incx,
                                          int* result) {
#if PARROT_USE_FLOAT
        return cublasIsamax(handle, n, x, incx, result);
#else
        return cublasIdamax(handle, n, x, incx, result);
#endif
}

struct ParrotSpmmAlgorithm {
    cusparseSpMMAlg_t value;
    const char* name;
};

inline ParrotSpmmAlgorithm select_parrot_spmm_algorithm(const char* env_name) {
    const char* env_value = std::getenv(env_name);
    if (env_value == nullptr || env_value[0] == '\0') {
        return {CUSPARSE_SPMM_ALG_DEFAULT, "default"};
    }

    const std::string alg(env_value);
    if (alg == "default") {
        return {CUSPARSE_SPMM_ALG_DEFAULT, "default"};
    }
    if (alg == "csr_alg1" || alg == "alg1" || alg == "1") {
        return {CUSPARSE_SPMM_CSR_ALG1, "csr_alg1"};
    }
    if (alg == "csr_alg2" || alg == "alg2" || alg == "2") {
        return {CUSPARSE_SPMM_CSR_ALG2, "csr_alg2"};
    }
    if (alg == "csr_alg3" || alg == "alg3" || alg == "3") {
        return {CUSPARSE_SPMM_CSR_ALG3, "csr_alg3"};
    }

    std::cerr << "Unknown " << env_name << "='" << alg
              << "', using default. Expected default, csr_alg1, csr_alg2, or csr_alg3."
              << std::endl;
    return {CUSPARSE_SPMM_ALG_DEFAULT, "default"};
}

struct CudaCsrMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int* d_row_offsets = nullptr;
    int* d_col_ind = nullptr;
    element_type* d_vals = nullptr;
    cusparseSpMatDescr_t descr = nullptr;
};

struct CudaSpMatView {
    cusparseSpMatDescr_t descr = nullptr;
};

struct BatchRwrContext {
    cusparseHandle_t cusparse = nullptr;
    cublasHandle_t cublas = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_x = nullptr;
    cusparseDnMatDescr_t mat_xnext = nullptr;
    element_type* d_x = nullptr;
    element_type* d_xnext = nullptr;
    int* d_anchors = nullptr;
    void* d_buffer = nullptr;
    size_t dense_capacity = 0;
    size_t anchor_capacity = 0;
    size_t buffer_capacity = 0;
    int current_n = 0;
    int current_l = 0;
};

#ifdef DEBUG
void parrot_cuda_memtracker_reset();
void parrot_cuda_memtracker_report_peak(const char* label = nullptr);
cudaError_t parrot_cuda_malloc_tracked(void** ptr, size_t bytes, const char* file, int line);
cudaError_t parrot_cuda_free_tracked(void* ptr, const char* file, int line);
#else
inline void parrot_cuda_memtracker_reset() {}
inline void parrot_cuda_memtracker_report_peak(const char* label = nullptr) {
    (void)label;
}
inline cudaError_t parrot_cuda_malloc_tracked(void** ptr, size_t bytes, const char* file, int line) {
    (void)file;
    (void)line;
    return cudaMalloc(ptr, bytes);
}
inline cudaError_t parrot_cuda_free_tracked(void* ptr, const char* file, int line) {
    (void)file;
    (void)line;
    return cudaFree(ptr);
}
#endif

#define PARROT_CUDA_MALLOC(ptr, bytes) \
    parrot_cuda_malloc_tracked(reinterpret_cast<void**>(ptr), static_cast<size_t>(bytes), __FILE__, __LINE__)

#define PARROT_CUDA_FREE(ptr) \
    parrot_cuda_free_tracked(reinterpret_cast<void*>(ptr), __FILE__, __LINE__)

void destroy_cuda_csr(CudaCsrMatrix& matrix);
CudaCsrMatrix make_cuda_csr_from_edgelist(const std::vector<element_type>& edgelist, int m, int n);
CudaCsrMatrix transpose_cuda_csr(const CudaCsrMatrix& matrix);
CudaCsrMatrix row_normalize_graph_cuda(const CudaCsrMatrix& graph, int avg_nnz_per_n);
CudaSpMatView create_spmm_matrix_view(const CudaCsrMatrix& matrix);
void destroy_spmm_matrix_view(CudaSpMatView& view);
void destroy_batch_rwr_context(BatchRwrContext& context);

void load_config(const std::string& data_path, int& g1_num_nodes, int& g2_num_nodes,
                 bool& use_attr, bool& directed, int& inner_iters, int& outer_iters);
std::vector<element_type> readFileToVector(const std::string& filename, bool directed);
std::pair<std::vector<int>, std::vector<int>> get_anchor_pairs(std::vector<element_type>& H_vec);
void get_batch_rwr_scores_cusparse_device(const CudaCsrMatrix& graph, std::vector<int> anchors,
                                          int n, float restart_prob, int max_iters,
                                          element_type* d_X_out, BatchRwrContext& context);
void get_product_rwr_mat_cusparse_device(const CudaCsrMatrix& graph1_normalized,
                                         const CudaCsrMatrix& graph2_normalized,
                                         const element_type* d_in_cross_dist,
                                         int n1, int n2, element_type rwr_restart_prob,
                                         element_type gamma, element_type* d_out_cross_dist,
                                         int max_iters = 50, element_type tol = static_cast<element_type>(1e-2));

CudaCsrMatrix masked_sddmm_intra_cost_cusparse_device(const element_type* d_emb,
                                                      const CudaCsrMatrix& mask,
                                                      int n, int L);

void l2_norm_inplace_device(element_type* d_rwr, size_t n, size_t d);
void get_normalized_neg_exp_dist_device(const element_type* d_rwr1, const element_type* d_rwr2,
                                        int n1, int n2, int l, element_type* d_cross_rwr_dist);
void anchor_cost_zero_device(element_type* d_A, int nrows, int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             element_type val);
std::vector<element_type> con_prox_pt_opt_device(const CudaCsrMatrix& row_norm_adj1,
                                                 const CudaCsrMatrix& row_norm_adj2,
                                                 int n1, int n2, int avg_nnz1_per_n1, int avg_nnz2_per_n2,
                                                 const element_type* d_cross_dist,
                                                 const CudaCsrMatrix& intra_dist1_sparse,
                                                 const CudaCsrMatrix& intra_dist2_sparse,
                                                 int inner_iters, int outer_iters, element_type lambda_n, element_type lambda_a,
                                                 element_type lambda_p, element_type lambda_e_in,
                                                 const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs,
                                                 bool sparsify);

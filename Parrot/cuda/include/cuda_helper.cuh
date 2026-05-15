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
    double* d_vals = nullptr;
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
    double* d_x = nullptr;
    double* d_xnext = nullptr;
    int* d_anchors = nullptr;
    void* d_buffer = nullptr;
    size_t dense_capacity = 0;
    size_t anchor_capacity = 0;
    size_t buffer_capacity = 0;
    int current_n = 0;
    int current_l = 0;
};

void destroy_cuda_csr(CudaCsrMatrix& matrix);
CudaCsrMatrix make_cuda_csr_from_edgelist(const std::vector<double>& edgelist, int m, int n);
CudaCsrMatrix transpose_cuda_csr(const CudaCsrMatrix& matrix);
CudaCsrMatrix row_normalize_graph_cuda(const CudaCsrMatrix& graph, int avg_nnz_per_n);
CudaSpMatView create_spmm_matrix_view(const CudaCsrMatrix& matrix);
void destroy_spmm_matrix_view(CudaSpMatView& view);
void destroy_batch_rwr_context(BatchRwrContext& context);

void load_config(const std::string& data_path, int& g1_num_nodes, int& g2_num_nodes,
                 bool& use_attr, bool& directed, int& inner_iters, int& outer_iters);
std::vector<double> readFileToVector(const std::string& filename, bool directed);
std::pair<std::vector<int>, std::vector<int>> get_anchor_pairs(std::vector<double>& H_vec);
void get_batch_rwr_scores_cusparse_device(const CudaCsrMatrix& graph, std::vector<int> anchors,
                                          int n, float restart_prob, int max_iters,
                                          double* d_X_out, BatchRwrContext& context);
void get_product_rwr_mat_cusparse_device(const CudaCsrMatrix& graph1_normalized,
                                         const CudaCsrMatrix& graph2_normalized,
                                         const double* d_in_cross_dist,
                                         int n1, int n2, double rwr_restart_prob,
                                         double gamma, double* d_out_cross_dist,
                                         int max_iters = 50, double tol = 1e-2);

CudaCsrMatrix masked_sddmm_intra_cost_cusparse_device(const double* d_emb,
                                                      const CudaCsrMatrix& mask,
                                                      int n, int L);

void l2_norm_inplace_device(double* d_rwr, size_t n, size_t d);
void get_normalized_neg_exp_dist_device(const double* d_rwr1, const double* d_rwr2,
                                        int n1, int n2, int l, double* d_cross_rwr_dist);
void anchor_cost_zero_device(double* d_A, int nrows, int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             double val);
std::vector<double> con_prox_pt_opt_device(const CudaCsrMatrix& row_norm_adj1,
                                           const CudaCsrMatrix& row_norm_adj2,
                                           int n1, int n2, int avg_nnz1_per_n1, int avg_nnz2_per_n2,
                                           const double* d_cross_dist,
                                           const CudaCsrMatrix& intra_dist1_sparse,
                                           const CudaCsrMatrix& intra_dist2_sparse,
                                           int inner_iters, int outer_iters, double lambda_n, double lambda_a,
                                           double lambda_p, double lambda_e_in,
                                           const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs,
                                           bool sparsify);

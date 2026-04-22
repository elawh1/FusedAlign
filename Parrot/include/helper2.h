#pragma once

#include <mkl.h>
#include <cmath>
#include <cinttypes>   
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <omp.h>
#include <sstream>
#include <cstdio>
#include <mkl_spblas.h>
// #include <metis.h>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

bool has_nan(const std::vector<double>& v);

bool has_nan_or_inf(const std::vector<double>& v) ;

bool sparse_has_nan_or_inf(sparse_matrix_t A) ;

double sparse_max_abs(sparse_matrix_t A) ;

void sparse_print_large_values(sparse_matrix_t A, double threshold) ;

double getMax(const std::vector<double>& v) ;

double getMin(const std::vector<double>& v) ;

void load_config(const std::string& data_path, MKL_INT& g1_num_nodes, MKL_INT& g2_num_nodes, bool& use_attr, bool& directed, int& inner_iters, int& outer_iters) ;

std::vector<double> readFileToVector(const std::string &filename, bool directed) ;

bool compare_sparse_exact( sparse_matrix_t A, sparse_matrix_t B, double eps = 1e-9) ;

sparse_matrix_t make_mat_from_edgelist(const std::vector<double>& edgelist, int m, int n) ;

void print_sparse_matrix_info(sparse_matrix_t A, const char* name = "matrix") ;

void print_num_zeros(std::vector<double>  M, const char* name ) ;

void compare( const std::vector<double>& cpp_vec, const std::string& filename, const std::string& label) ;

void print_Cost_vals (const std::vector<double>& Cost) ;

double percentile(std::vector<double> S, double p) ;

void print_stats(const std::vector<double>& S) ;

void sparsify_vector(std::vector<double>& S, double threshold) ;

void sparsify_vector_reverse(std::vector<double>& S, double threshold) ;

void graph_connect_isolated(sparse_matrix_t graph, std::vector<double>& graph_row_sum, sparse_matrix_t& graph_connected) ;


// namespace {

// inline void check_sparse(sparse_status_t st, const char* msg) {
//     if (st != SPARSE_STATUS_SUCCESS) {
//         throw std::runtime_error(msg);
//     }
// }

// // Deep-copy an MKL sparse matrix handle.
// sparse_matrix_t deep_copy_sparse(sparse_matrix_t A) {
//     sparse_matrix_t B = nullptr;
//     matrix_descr descr{};
//     descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//     check_sparse(mkl_sparse_copy(A, descr, &B), "mkl_sparse_copy failed");
//     return B;
// }

// // Build a unit-valued CSR matrix from an edge list and return an independent MKL sparse handle.
// sparse_matrix_t build_unit_csr_from_edges(
//     const std::vector<std::pair<MKL_INT, MKL_INT>>& edges_in,
//     MKL_INT nrows,
//     MKL_INT ncols)
// {
//     if (edges_in.empty()) {
//         return nullptr;
//     }

//     std::vector<std::pair<MKL_INT, MKL_INT>> edges = edges_in;
//     std::sort(edges.begin(), edges.end());
//     edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

//     std::vector<MKL_INT> row_counts(nrows, 0);
//     for (const auto& e : edges) {
//         if (e.first < 0 || e.first >= nrows || e.second < 0 || e.second >= ncols) {
//             throw std::runtime_error("edge index out of bounds in build_unit_csr_from_edges");
//         }
//         row_counts[e.first]++;
//     }

//     std::vector<MKL_INT> row_start(nrows), row_end(nrows);
//     std::vector<MKL_INT> col_indx(edges.size());
//     std::vector<double> values(edges.size(), 1.0);

//     MKL_INT nnz_ptr = 0;
//     for (MKL_INT i = 0; i < nrows; ++i) {
//         row_start[i] = nnz_ptr;
//         nnz_ptr += row_counts[i];
//         row_end[i] = nnz_ptr;
//     }

//     std::vector<MKL_INT> next(row_start.begin(), row_start.end());
//     for (const auto& e : edges) {
//         col_indx[next[e.first]++] = e.second;
//     }

//     sparse_matrix_t tmp = nullptr;
//     check_sparse(
//         mkl_sparse_d_create_csr(&tmp,
//                                 SPARSE_INDEX_BASE_ZERO,
//                                 nrows,
//                                 ncols,
//                                 row_start.data(),
//                                 row_end.data(),
//                                 col_indx.data(),
//                                 values.data()),
//         "mkl_sparse_d_create_csr failed in build_unit_csr_from_edges");

//     sparse_matrix_t out = deep_copy_sparse(tmp);
//     mkl_sparse_destroy(tmp);
//     return out;
// }

// } // namespace


// Assumes graph is an adjacency matrix.
// Row-normalization here is degree-based for an unweighted graph:
// every nonzero in row i becomes 1.0 / nnz(row i).
// sparse_matrix_t graph_row_normalize_connect_isolated(sparse_matrix_t graph, unsigned int seed = 42) ;

// Drop-in blocked replacement for mkl_sparse_d_mm
// Assumes dense matrices are column-major.
// Internally splits columns into blocks and calls mkl_sparse_d_mm on each block.
//
// Parameters match mkl_sparse_d_mm:
//
// operation        : sparse op flag
// alpha            : scalar
// A                : sparse matrix (CSR handle typically)
// descr            : matrix descriptor
// layout           : must be SPARSE_LAYOUT_COLUMN_MAJOR here
// B                : input dense matrix
// columns          : number of columns of B and C
// ldb              : leading dimension of B
// beta             : scalar
// C                : output dense matrix
// ldc              : leading dimension of C
//
// Optional block_size has a default. You can tune 32/64/128/256/512.
//
inline sparse_status_t blocked_mkl_sparse_d_mm( sparse_operation_t operation, double alpha, sparse_matrix_t A, matrix_descr descr, sparse_layout_t layout, const double* B, 
    MKL_INT columns, MKL_INT ldb, double beta, double* C, MKL_INT ldc, MKL_INT block_size = 128 ) ;

inline void prepare_sparse_mm_once( sparse_matrix_t A, matrix_descr descr, MKL_INT typical_block_cols, MKL_INT expected_calls  ) ;

void daxpy_chunked(size_t N, double alpha, const double* x, double* y) ;

void dscal_chunked(size_t N, double alpha, double* x) ;

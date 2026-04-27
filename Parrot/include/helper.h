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
#include "debug_log.h"
// #include <metis.h>
// #include "helper2.h"
#include "fused_spmm.h"
#include "mkl_spblas.h"

struct MKLCSC {
    sparse_matrix_t spMat = nullptr;
    std::vector<MKL_INT> col_start, col_end, row_ind;
    std::vector<double>  vals;
    ~MKLCSC() { if (spMat) mkl_sparse_destroy(spMat); }
};

std::vector<double> sparse_to_dense_colmajor(sparse_matrix_t A, MKL_INT nrows, MKL_INT ncols) ;

//Create digonal sparse matrix from a vector of diagonal elements
MKLCSC make_diag_csc(const std::vector<double>& diag, MKL_INT m, MKL_INT n, bool invert) ;

std::vector<double> l2_norm(std::vector<double> rwr, int n, int d) ;

void l2_norm_inplace(std::vector<double>& rwr, size_t n, size_t d) ;

void get_normalized_neg_exp_dist (std::vector<double> rwr1, std::vector<double>  rwr2, int n1, int n2, int l, std::vector<double>& cross_rwr_dist) ;

void masked_sddmm_intra_cost (std::vector<double>&  emb, sparse_matrix_t mask, int n, int L, sparse_matrix_t& result) ;

std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<double>& H_vec) ;

sparse_matrix_t row_normalize_graph( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) ;

sparse_matrix_t row_normalize_graph_new( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) ;

void get_batch_rwr_scores(sparse_matrix_t graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, 
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) ;

void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, 
                    std::vector<MKL_INT>>& anchors, double val) ;

// Then set H[row, col] = 1 for each anchor
void init_H_with_anchors(std::vector<double>& H, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors) ;
// Then set H[row, col] = log(1e-6) for each anchor
std::vector<double> init_Hlog_with_anchors(MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors) ;

bool out_cross_dist_converge( std::vector<double>& out_cross_dist, std::vector<double>& out_cross_dist_old, double tol) ;

std::vector<double> get_product_rwr_mat_fused( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, 
    int n1, int n2, int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    double rwr_restart_prob, double gamma, int max_iters = 50, double tol = 1e-2) ;

void get_product_rwr_mat( sparse_matrix_t graph1_normalized, sparse_matrix_t graph2_normalized, const std::vector<double>& in_cross_dist, 
    int n1, int n2, double rwr_restart_prob, double gamma, std::vector<double>& out_cross_dist, int max_iters = 50, double tol = 1e-2) ;

void get_product_rwr_mat_new( sparse_matrix_t graph1_normalized, sparse_matrix_t graph2_normalized, 
    const std::vector<double>& in_cross_dist, MKL_INT n1, MKL_INT n2, double rwr_restart_prob, double gamma, 
    std::vector<double>& out_cross_dist, int max_iters = 50, double tol = 1e-2) ;

void minaa_v3_dense( std::vector<double>& Cost, const std::vector<double>& b, int n_row, int n_col, double epsilon, std::vector<double>& out_vec ) ;

void euclidean_distance_mat_vec( const std::vector<double>& X, const std::vector<double>& y, int n, int d, std::vector<double>& out_vec ) ;

void row_avg_square_dense(const std::vector<double>& matrix, size_t n, size_t d, std::vector<double>& row_sum_sqr) ;

void row_avg_square ( sparse_matrix_t& intra_dist, int n, std::vector<double>& row_avg_sqr) ;

void compute_L_fixed( sparse_matrix_t& intra_dist1, sparse_matrix_t& intra_dist2, int n1, int n2, std::vector<double>& L_fixed) ;

void compute_Cost_fixed( sparse_matrix_t& intra_dist1, sparse_matrix_t& intra_dist2, std::vector<double>& cross_dist, const std::vector<double>& H_log, 
        int n1, int n2, double lambda_e, double lambda_a ) ;

void compute_RW_term ( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, std::vector<double>& RW_term) ;

void compute_A2T_term( sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& S, int n1, int n2, std::vector<double>& A2T_term) ;

void compute_C_L( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, 
        sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& L_fixed,
         const std::vector<double>& cross_dist, const std::vector<double>& H_log, double lambda_e, 
         double lambda_a, double lambda_n, std::vector<double>& C) ;
void compute_C_L_fused( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, 
        sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& L_fixed,
         const std::vector<double>& cross_dist, const std::vector<double>& H_log, double lambda_e, 
         double lambda_a, double lambda_n, std::vector<double>& C) ;


void compute_cost_fused( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2,  sparse_matrix_t intra_dist1, 
        sparse_matrix_t intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C) ;


void compute_cost_fused_2( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2,  sparse_matrix_t intra_dist1, 
        sparse_matrix_t intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C) ;


double update_S_mat (std::vector<double>& S, const std::vector<double>& S_old, const std::vector<double>& Cost, const std::vector<double>& a_row, const std::vector<double>& b_col, int n1, int n2, double epsilon) ;

std::vector<double> con_prox_pt_opt( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2, 
    std::vector<double>& cross_dist, sparse_matrix_t intra_dist1_sparse, sparse_matrix_t intra_dist2_sparse, int inner_iters, int outer_iters, double lambda_n, double lambda_a, 
    double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs, bool sparsify) ;

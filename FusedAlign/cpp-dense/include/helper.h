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
#include <utility>
#include "debug_log.h"

std::vector<double> l2_norm(std::vector<double> rwr, int n, int d) ;

template <typename Real = double>
void l2_norm_inplace(std::vector<Real>& rwr, size_t n, size_t d) ;

template <typename Real = double>
void get_normalized_neg_exp_dist (const std::vector<Real>& rwr1, const std::vector<Real>&  rwr2, int n1, int n2, int l, std::vector<Real>& cross_rwr_dist) ;

template <typename Real = double>
std::vector<Real> make_mat_from_edgelist(const std::vector<Real>& edgelist, int m, int n) ;

template <typename Real = double>
void masked_dense_intra_cost(std::vector<Real>& emb, const std::vector<Real>& mask, int n, int L, std::vector<Real>& result) ;

template <typename Real = double>
std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<Real>& H_vec) ;

template <typename Real = double>
std::vector<Real> row_normalize_graph(const std::vector<Real>& graph, int n, MKL_INT avg_nnz_per_n) ;

void get_batch_rwr_scores(const std::vector<double>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, 
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) ;
template <typename Real = double>
void get_batch_rwr_scores_rowmjr(const std::vector<Real>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
        bool directed, Real restart_prob, int max_iters, bool connect_isolated, std::vector<Real>& Xnext_vals) ;

template <typename Real = double>
void anchor_cost_zero(std::vector<Real>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>,
                    std::vector<MKL_INT>>& anchors, Real val) ;

// Then set H[row, col] = 1 for each anchor
void init_H_with_anchors(std::vector<double>& H, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors) ;
// Then set H[row, col] = log(1e-6) for each anchor
template <typename Real = double>
std::vector<Real> init_Hlog_with_anchors(MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors) ;

bool out_cross_dist_converge( std::vector<double>& out_cross_dist, std::vector<double>& out_cross_dist_old, double tol) ;

template <typename Real = double>
void get_product_rwr_mat_new(const std::vector<Real>& graph1_normalized, const std::vector<Real>& graph2_normalized,
    const std::vector<Real>& in_cross_dist, MKL_INT n1, MKL_INT n2, Real rwr_restart_prob, Real gamma,
    std::vector<Real>& out_cross_dist, int max_iters = 50, Real tol = Real(1e-2)) ;

template <typename Real = double>
void minaa_v3_dense( std::vector<Real>& Cost, const std::vector<Real>& b, int n_row, int n_col, Real epsilon, std::vector<Real>& out_vec ) ;

void minaa_v3_dense_old( std::vector<double>& Cost, const std::vector<double>& b, int n_row, int n_col, double epsilon, std::vector<double>& out_vec ) ;

void euclidean_distance_mat_vec( const std::vector<double>& X, const std::vector<double>& y, int n, int d, std::vector<double>& out_vec ) ;

void row_avg_square_dense(const std::vector<double>& matrix, size_t n, size_t d, std::vector<double>& row_sum_sqr) ;

template <typename Real = double>
void row_avg_square(const std::vector<Real>& intra_dist, int n, std::vector<Real>& row_avg_sqr) ;

template <typename Real = double>
void compute_Cost_fixed(const std::vector<Real>& intra_dist1, const std::vector<Real>& intra_dist2, std::vector<Real>& cross_dist,
        int n1, int n2, Real lambda_e, Real lambda_a, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs) ;

template <typename Real = double>
void compute_cost_fused_2(const std::vector<Real>& row_norm_adj1, const std::vector<Real>& row_norm_adj2, const std::vector<Real>& S, int n1, int n2,
        const std::vector<Real>& intra_dist1, const std::vector<Real>& intra_dist2, const std::vector<Real>& C_fixed, Real lambda_e, Real lambda_n,
        std::vector<Real>& C, std::vector<Real>& buffer, std::vector<Real>& buffer2) ;

template <typename Real = double>
Real update_S_mat (std::vector<Real>& S, const std::vector<Real>& S_old, const std::vector<Real>& Cost, const std::vector<Real>& a_row, const std::vector<Real>& b_col, int n1, int n2, Real epsilon) ;

template <typename Real = double>
std::vector<Real> con_prox_pt_opt(const std::vector<Real>& row_norm_adj1, const std::vector<Real>& row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    std::vector<Real>& cross_dist, const std::vector<Real>& intra_dist1, const std::vector<Real>& intra_dist2, int inner_iters, int outer_iters, Real lambda_n, Real lambda_a,
    Real lambda_p, Real lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs) ;

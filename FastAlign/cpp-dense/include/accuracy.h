#pragma once

#include <mkl.h>
#include <cstdio>
#include <cinttypes>   
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> 
#include <unordered_map>


// S: column-major n1 x n2 matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
template <typename Real = double>
std::unordered_map<int, double> hits_ks_ltr_scores( const std::vector<Real>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) ;
// S: column-major n1 x n2 similarity matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
template <typename Real = double>
std::unordered_map<int, double> hits_ks_rtl_scores( const std::vector<Real>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) ;

template <typename Real = double>
std::unordered_map<int, double> hits_ks_mean_scores( const std::vector<Real>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) ;


template <typename Real = double>
double mrr_ltr_score_cpp( const std::vector<Real>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) ;

template <typename Real = double>
double mrr_rtl_score_cpp( const std::vector<Real>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) ;

template <typename Real = double>
double mrr_mean_score_cpp( const std::vector<Real>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) ;

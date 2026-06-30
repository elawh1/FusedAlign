// #include <cstdio>
// #include <cinttypes>   
// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <numeric> 
// #include <unordered_map>
#include "accuracy.h"

// S: column-major n1 x n2 matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
template <typename Real>
std::unordered_map<int, double> hits_ks_ltr_scores( const std::vector<Real>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {
    const int num_tests = anchor_pairs_test.first.size();
    const int kcount = ks.size();
    std::vector<int> total_hit_counts(kcount, 0);
    std::unordered_map<int, double> hits_ks;

#pragma omp parallel
    {
        std::vector<int> idx(n2);
        std::vector<int> local_hit_counts(kcount, 0);

#pragma omp for schedule(static)
        for (int t = 0; t < num_tests; ++t) {
            int i = anchor_pairs_test.first[t];
            int j_true = anchor_pairs_test.second[t];
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                [&](int a, int b) {
                    size_t idx1 = (size_t) i + (size_t) a * (size_t) n1;
                    size_t idx2 = (size_t) i + (size_t) b * (size_t) n1;
                    return S[idx1] > S[idx2];
                });

            for (int r = 0; r < n2; ++r) {
                if (idx[r] == j_true) {
                    for (int q = 0; q < kcount; ++q)
                        if (r < ks[q]) local_hit_counts[q]++;
                    break;
                }
            }
        }

#pragma omp critical
        for (int q = 0; q < kcount; ++q)
            total_hit_counts[q] += local_hit_counts[q];
    }

    for (int q = 0; q < kcount; ++q) {
        hits_ks[ks[q]] = static_cast<double>(total_hit_counts[q]) / num_tests;
    }

    return hits_ks;
}

// S: column-major n1 x n2 similarity matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
template <typename Real>
std::unordered_map<int, double> hits_ks_rtl_scores( const std::vector<Real>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {

    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();
    const int kcount = ks.size();
    std::vector<int> total_hit_counts(kcount, 0);
    std::unordered_map<int, double> hits_ks;

#pragma omp parallel
    {
        std::vector<int> idx(n1);
        std::vector<int> local_hit_counts(kcount, 0);

#pragma omp for schedule(static)
        for (int t = 0; t < num_tests; ++t) {
            int j = right_ids[t];
            int i_true = left_ids[t];
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                [&](int a, int b) {
                    size_t idx1 = (size_t) a + (size_t) j * (size_t) n1;
                    size_t idx2 = (size_t) b + (size_t) j * (size_t) n1;
                    return S[idx1] > S[idx2];
                });
            for (int r = 0; r < n1; ++r) {
                if (idx[r] == i_true) {
                    for (int q = 0; q < kcount; ++q)
                        if (r < ks[q]) local_hit_counts[q]++;
                    break;
                }
            }
        }

#pragma omp critical
        for (int q = 0; q < kcount; ++q)
            total_hit_counts[q] += local_hit_counts[q];
    }

    for (int q = 0; q < kcount; ++q)
        hits_ks[ks[q]] = double(total_hit_counts[q]) / num_tests;

    return hits_ks;
}

template <typename Real>
std::unordered_map<int, double> hits_ks_mean_scores( const std::vector<Real>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {
    std::unordered_map<int, double> hits_ks;
    auto hits_ltr = hits_ks_ltr_scores<Real>( S, n1, n2, anchor_pairs_test, ks);
    auto hits_rtl = hits_ks_rtl_scores<Real>( S, n1, n2, anchor_pairs_test, ks);
    for (int k : ks) {
        hits_ks[k] = 0.5 * (hits_ltr[k] + hits_rtl[k]);
    }
    return hits_ks;
}


template <typename Real>
double mrr_ltr_score_cpp( const std::vector<Real>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();
    double mrr_sum = 0.0;

#pragma omp parallel
    {
        std::vector<int> idx(n2);

#pragma omp for schedule(static) reduction(+:mrr_sum)
        for (int t = 0; t < num_tests; ++t) {
            int i = left_ids[t];        // left node
            int j_true = right_ids[t];  // correct right node
            std::iota(idx.begin(), idx.end(), 0);
            // sort similarity row i descending
            std::sort(idx.begin(), idx.end(),
                [&](int a, int b) {
                    size_t idx1 = (size_t) i + (size_t) a * (size_t) n1;
                    size_t idx2 = (size_t) i + (size_t) b * (size_t) n1;
                    return S[idx1] > S[idx2];
                });
            // find rank
            for (int r = 0; r < n2; ++r) {
                if (idx[r] == j_true) {
                    mrr_sum += 1.0 / double(r + 1);  // 1-based rank
                    break;
                }
            }
        }
    }

    return mrr_sum / num_tests;
}

template <typename Real>
double mrr_rtl_score_cpp( const std::vector<Real>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();
    double mrr_sum = 0.0;

#pragma omp parallel
    {
        std::vector<int> idx(n1);

#pragma omp for schedule(static) reduction(+:mrr_sum)
        for (int t = 0; t < num_tests; ++t) {
            int j = right_ids[t];       // right node
            int i_true = left_ids[t];   // correct left node
            std::iota(idx.begin(), idx.end(), 0);
            // sort similarity column j descending
            std::sort(idx.begin(), idx.end(),
                [&](int a, int b) {
                    size_t idx1 = (size_t) a + (size_t) j * (size_t) n1;
                    size_t idx2 = (size_t) b + (size_t) j * (size_t) n1;
                    return S[idx1] > S[idx2];
                });
            // find rank
            for (int r = 0; r < n1; ++r) {
                if (idx[r] == i_true) {
                    mrr_sum += 1.0 / double(r + 1);  // 1-based rank
                    break;
                }
            }
        }
    }

    return mrr_sum / num_tests;
}

template <typename Real>
double mrr_mean_score_cpp( const std::vector<Real>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    double mrr_ltr = mrr_ltr_score_cpp<Real>(S, n1, n2, anchor_pairs_test);
    double mrr_rtl = mrr_rtl_score_cpp<Real>(S, n1, n2, anchor_pairs_test);
    return 0.5 * (mrr_ltr + mrr_rtl);
}

template std::unordered_map<int, double> hits_ks_ltr_scores<double>(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks);
template std::unordered_map<int, double> hits_ks_ltr_scores<float>(const std::vector<float>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks);

template std::unordered_map<int, double> hits_ks_rtl_scores<double>(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks);
template std::unordered_map<int, double> hits_ks_rtl_scores<float>(const std::vector<float>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks);

template std::unordered_map<int, double> hits_ks_mean_scores<double>(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks);
template std::unordered_map<int, double> hits_ks_mean_scores<float>(const std::vector<float>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks);

template double mrr_ltr_score_cpp<double>(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test);
template double mrr_ltr_score_cpp<float>(const std::vector<float>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test);

template double mrr_rtl_score_cpp<double>(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test);
template double mrr_rtl_score_cpp<float>(const std::vector<float>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test);

template double mrr_mean_score_cpp<double>(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test);
template double mrr_mean_score_cpp<float>(const std::vector<float>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test);

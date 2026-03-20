#include <cstdio>
#include <cinttypes>   
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> 
#include <unordered_map>

// S: column-major n1 x n2 matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
std::unordered_map<int, double> hits_ks_ltr_scores( const std::vector<double>& S, int n1, int n2, 
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {
    const int num_tests = anchor_pairs_test.first.size();
    std::unordered_map<int, int> hit_counts;
    std::unordered_map<int, double> hits_ks;
    for (int k : ks) hit_counts[k] = 0;
    std::vector<int> idx(n2);

    for (int t = 0; t < num_tests; ++t) {
        int i = anchor_pairs_test.first[t];
        int j_true = anchor_pairs_test.second[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[i + a * n1] > S[i + b * n1];
            });

        for (int r = 0; r < n2; ++r) {
            if (idx[r] == j_true) {
                for (int k : ks)
                    if (r < k) hit_counts[k]++;
                break;
            }
        }
    }
    // normalize
    for (int k : ks) {
        hits_ks[k] = static_cast<double>(hit_counts[k]) / num_tests;
    }

    return hits_ks;
}

// S: column-major n1 x n2 similarity matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
std::unordered_map<int, double> hits_ks_rtl_scores( const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {

    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();

    std::unordered_map<int, int> hit_counts;
    std::unordered_map<int, double> hits_ks;
    for (int k : ks) hit_counts[k] = 0;

    std::vector<int> idx(n1);

    for (int t = 0; t < num_tests; ++t) {
        int j = right_ids[t];
        int i_true = left_ids[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[a + j * n1] > S[b + j * n1];
            });
        for (int r = 0; r < n1; ++r) {
            if (idx[r] == i_true) {
                for (int k : ks)
                    if (r < k) hit_counts[k]++;
                break;
            }
        }
    }
    for (int k : ks)
        hits_ks[k] = double(hit_counts[k]) / num_tests;

    return hits_ks;
}

std::unordered_map<int, double> hits_ks_mean_scores( const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {
    std::unordered_map<int, double> hits_ks;
    auto hits_ltr = hits_ks_ltr_scores( S, n1, n2, anchor_pairs_test, ks);
    auto hits_rtl = hits_ks_rtl_scores( S, n1, n2, anchor_pairs_test, ks);
    for (int k : ks) {
        hits_ks[k] = 0.5 * (hits_ltr[k] + hits_rtl[k]);
    }
    return hits_ks;
}


double mrr_ltr_score_cpp( const std::vector<double>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();
    std::vector<int> idx(n2);
    double mrr_sum = 0.0;

    for (int t = 0; t < num_tests; ++t) {
        int i = left_ids[t];        // left node
        int j_true = right_ids[t];  // correct right node
        std::iota(idx.begin(), idx.end(), 0);
        // sort similarity row i descending
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[i + a * n1] > S[i + b * n1];
            });
        // find rank
        for (int r = 0; r < n2; ++r) {
            if (idx[r] == j_true) {
                mrr_sum += 1.0 / double(r + 1);  // 1-based rank
                break;
            }
        }
    }

    return mrr_sum / num_tests;
}

double mrr_rtl_score_cpp( const std::vector<double>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();
    std::vector<int> idx(n1);
    double mrr_sum = 0.0;

    for (int t = 0; t < num_tests; ++t) {
        int j = right_ids[t];       // right node
        int i_true = left_ids[t];   // correct left node
        std::iota(idx.begin(), idx.end(), 0);
        // sort similarity column j descending
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[a + j * n1] > S[b + j * n1];
            });
        // find rank
        for (int r = 0; r < n1; ++r) {
            if (idx[r] == i_true) {
                mrr_sum += 1.0 / double(r + 1);  // 1-based rank
                break;
            }
        }
    }

    return mrr_sum / num_tests;
}

double mrr_mean_score_cpp( const std::vector<double>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    double mrr_ltr = mrr_ltr_score_cpp(S, n1, n2, anchor_pairs_test);
    double mrr_rtl = mrr_rtl_score_cpp(S, n1, n2, anchor_pairs_test);
    return 0.5 * (mrr_ltr + mrr_rtl);
}
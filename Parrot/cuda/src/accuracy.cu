#include "accuracy.h"

#include <algorithm>

std::unordered_map<int, double> hits_ks_ltr_scores(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test, const std::vector<int>& ks) {
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    std::unordered_map<int, int> hit_counts;
    std::unordered_map<int, double> hits_ks;
    for (int k : ks) hit_counts[k] = 0;
    std::vector<int> idx(n2);

    for (int t = 0; t < num_tests; ++t) {
        const int i = anchor_pairs_test.first[t];
        const int j_true = anchor_pairs_test.second[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const size_t idx1 = static_cast<size_t>(i) + static_cast<size_t>(a) * static_cast<size_t>(n1);
            const size_t idx2 = static_cast<size_t>(i) + static_cast<size_t>(b) * static_cast<size_t>(n1);
            return S[idx1] > S[idx2];
        });
        for (int r = 0; r < n2; ++r) {
            if (idx[r] == j_true) {
                for (int k : ks) {
                    if (r < k) hit_counts[k]++;
                }
                break;
            }
        }
    }

    for (int k : ks) {
        hits_ks[k] = static_cast<double>(hit_counts[k]) / num_tests;
    }
    return hits_ks;
}

std::unordered_map<int, double> hits_ks_rtl_scores(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test, const std::vector<int>& ks) {
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    std::unordered_map<int, int> hit_counts;
    std::unordered_map<int, double> hits_ks;
    for (int k : ks) hit_counts[k] = 0;
    std::vector<int> idx(n1);

    for (int t = 0; t < num_tests; ++t) {
        const int j = anchor_pairs_test.second[t];
        const int i_true = anchor_pairs_test.first[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const size_t idx1 = static_cast<size_t>(a) + static_cast<size_t>(j) * static_cast<size_t>(n1);
            const size_t idx2 = static_cast<size_t>(b) + static_cast<size_t>(j) * static_cast<size_t>(n1);
            return S[idx1] > S[idx2];
        });
        for (int r = 0; r < n1; ++r) {
            if (idx[r] == i_true) {
                for (int k : ks) {
                    if (r < k) hit_counts[k]++;
                }
                break;
            }
        }
    }

    for (int k : ks) {
        hits_ks[k] = static_cast<double>(hit_counts[k]) / num_tests;
    }
    return hits_ks;
}

std::unordered_map<int, double> hits_ks_mean_scores(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test, const std::vector<int>& ks) {
    auto hits_ltr = hits_ks_ltr_scores(S, n1, n2, anchor_pairs_test, ks);
    auto hits_rtl = hits_ks_rtl_scores(S, n1, n2, anchor_pairs_test, ks);
    std::unordered_map<int, double> hits_ks;
    for (int k : ks) {
        hits_ks[k] = 0.5 * (hits_ltr[k] + hits_rtl[k]);
    }
    return hits_ks;
}

double mrr_ltr_score_cpp(const std::vector<double>& S, int n1, int n2,
                         const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test) {
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    std::vector<int> idx(n2);
    double mrr_sum = 0.0;
    for (int t = 0; t < num_tests; ++t) {
        const int i = anchor_pairs_test.first[t];
        const int j_true = anchor_pairs_test.second[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const size_t idx1 = static_cast<size_t>(i) + static_cast<size_t>(a) * static_cast<size_t>(n1);
            const size_t idx2 = static_cast<size_t>(i) + static_cast<size_t>(b) * static_cast<size_t>(n1);
            return S[idx1] > S[idx2];
        });
        for (int r = 0; r < n2; ++r) {
            if (idx[r] == j_true) {
                mrr_sum += 1.0 / static_cast<double>(r + 1);
                break;
            }
        }
    }
    return mrr_sum / num_tests;
}

double mrr_rtl_score_cpp(const std::vector<double>& S, int n1, int n2,
                         const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test) {
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    std::vector<int> idx(n1);
    double mrr_sum = 0.0;
    for (int t = 0; t < num_tests; ++t) {
        const int j = anchor_pairs_test.second[t];
        const int i_true = anchor_pairs_test.first[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const size_t idx1 = static_cast<size_t>(a) + static_cast<size_t>(j) * static_cast<size_t>(n1);
            const size_t idx2 = static_cast<size_t>(b) + static_cast<size_t>(j) * static_cast<size_t>(n1);
            return S[idx1] > S[idx2];
        });
        for (int r = 0; r < n1; ++r) {
            if (idx[r] == i_true) {
                mrr_sum += 1.0 / static_cast<double>(r + 1);
                break;
            }
        }
    }
    return mrr_sum / num_tests;
}

double mrr_mean_score_cpp(const std::vector<double>& S, int n1, int n2,
                          const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test) {
    return 0.5 * (mrr_ltr_score_cpp(S, n1, n2, anchor_pairs_test) +
                  mrr_rtl_score_cpp(S, n1, n2, anchor_pairs_test));
}

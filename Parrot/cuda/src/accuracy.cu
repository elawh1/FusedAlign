#include "accuracy.h"

#include <stdexcept>

namespace {

void validate_inputs(const std::vector<double>& S, int n1, int n2,
                     const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test) {
    if (anchor_pairs_test.first.size() != anchor_pairs_test.second.size()) {
        throw std::invalid_argument("anchor_pairs_test size mismatch");
    }
    if (S.size() != static_cast<size_t>(n1) * static_cast<size_t>(n2)) {
        throw std::invalid_argument("S size mismatch in accuracy computation");
    }
}

int ltr_rank(const std::vector<double>& S, int n1, int n2, int i, int j_true) {
    const double true_score = S[static_cast<size_t>(i) + static_cast<size_t>(j_true) * static_cast<size_t>(n1)];
    int rank = 1;
    for (int j = 0; j < n2; ++j) {
        if (j == j_true) {
            continue;
        }
        const double candidate = S[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(n1)];
        if (candidate > true_score) {
            ++rank;
        }
    }
    return rank;
}

int rtl_rank(const std::vector<double>& S, int n1, int j, int i_true) {
    const double true_score = S[static_cast<size_t>(i_true) + static_cast<size_t>(j) * static_cast<size_t>(n1)];
    int rank = 1;
    const size_t col_offset = static_cast<size_t>(j) * static_cast<size_t>(n1);
    for (int i = 0; i < n1; ++i) {
        if (i == i_true) {
            continue;
        }
        const double candidate = S[static_cast<size_t>(i) + col_offset];
        if (candidate > true_score) {
            ++rank;
        }
    }
    return rank;
}

std::unordered_map<int, double> make_hits_from_counts(const std::vector<int>& ks,
                                                       const std::vector<int>& hit_counts,
                                                       int num_tests) {
    std::unordered_map<int, double> hits;
    hits.reserve(ks.size());
    for (size_t idx = 0; idx < ks.size(); ++idx) {
        hits[ks[idx]] = num_tests > 0 ? static_cast<double>(hit_counts[idx]) / static_cast<double>(num_tests) : 0.0;
    }
    return hits;
}

}  // namespace

std::unordered_map<int, double> hits_ks_ltr_scores(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test, const std::vector<int>& ks) {
    validate_inputs(S, n1, n2, anchor_pairs_test);
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    std::vector<int> hit_counts(ks.size(), 0);

    for (int t = 0; t < num_tests; ++t) {
        const int rank = ltr_rank(S, n1, n2, anchor_pairs_test.first[t], anchor_pairs_test.second[t]);
        for (size_t idx = 0; idx < ks.size(); ++idx) {
            if (rank <= ks[idx]) {
                ++hit_counts[idx];
            }
        }
    }

    return make_hits_from_counts(ks, hit_counts, num_tests);
}

std::unordered_map<int, double> hits_ks_rtl_scores(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test, const std::vector<int>& ks) {
    validate_inputs(S, n1, n2, anchor_pairs_test);
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    std::vector<int> hit_counts(ks.size(), 0);

    for (int t = 0; t < num_tests; ++t) {
        const int rank = rtl_rank(S, n1, anchor_pairs_test.second[t], anchor_pairs_test.first[t]);
        for (size_t idx = 0; idx < ks.size(); ++idx) {
            if (rank <= ks[idx]) {
                ++hit_counts[idx];
            }
        }
    }

    return make_hits_from_counts(ks, hit_counts, num_tests);
}

std::unordered_map<int, double> hits_ks_mean_scores(const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test, const std::vector<int>& ks) {
    return accuracy_mean_scores_cpp(S, n1, n2, anchor_pairs_test, ks).hits;
}

double mrr_ltr_score_cpp(const std::vector<double>& S, int n1, int n2,
                         const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test) {
    validate_inputs(S, n1, n2, anchor_pairs_test);
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    double mrr_sum = 0.0;
    for (int t = 0; t < num_tests; ++t) {
        const int rank = ltr_rank(S, n1, n2, anchor_pairs_test.first[t], anchor_pairs_test.second[t]);
        mrr_sum += 1.0 / static_cast<double>(rank);
    }
    return num_tests > 0 ? mrr_sum / static_cast<double>(num_tests) : 0.0;
}

double mrr_rtl_score_cpp(const std::vector<double>& S, int n1, int n2,
                         const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test) {
    validate_inputs(S, n1, n2, anchor_pairs_test);
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());
    double mrr_sum = 0.0;
    for (int t = 0; t < num_tests; ++t) {
        const int rank = rtl_rank(S, n1, anchor_pairs_test.second[t], anchor_pairs_test.first[t]);
        mrr_sum += 1.0 / static_cast<double>(rank);
    }
    return num_tests > 0 ? mrr_sum / static_cast<double>(num_tests) : 0.0;
}

double mrr_mean_score_cpp(const std::vector<double>& S, int n1, int n2,
                          const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test) {
    return 0.5 * (mrr_ltr_score_cpp(S, n1, n2, anchor_pairs_test) +
                  mrr_rtl_score_cpp(S, n1, n2, anchor_pairs_test));
}

AccuracyScores accuracy_mean_scores_cpp(
    const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test,
    const std::vector<int>& ks) {
    validate_inputs(S, n1, n2, anchor_pairs_test);
    const int num_tests = static_cast<int>(anchor_pairs_test.first.size());

    std::vector<int> ltr_hit_counts(ks.size(), 0);
    std::vector<int> rtl_hit_counts(ks.size(), 0);
    double ltr_mrr_sum = 0.0;
    double rtl_mrr_sum = 0.0;

    for (int t = 0; t < num_tests; ++t) {
        const int i = anchor_pairs_test.first[t];
        const int j = anchor_pairs_test.second[t];
        const int rank_ltr = ltr_rank(S, n1, n2, i, j);
        const int rank_rtl = rtl_rank(S, n1, j, i);

        ltr_mrr_sum += 1.0 / static_cast<double>(rank_ltr);
        rtl_mrr_sum += 1.0 / static_cast<double>(rank_rtl);

        for (size_t idx = 0; idx < ks.size(); ++idx) {
            if (rank_ltr <= ks[idx]) {
                ++ltr_hit_counts[idx];
            }
            if (rank_rtl <= ks[idx]) {
                ++rtl_hit_counts[idx];
            }
        }
    }

    AccuracyScores scores;
    scores.hits.reserve(ks.size());
    if (num_tests == 0) {
        for (int k : ks) {
            scores.hits[k] = 0.0;
        }
        scores.mrr = 0.0;
        return scores;
    }

    const double inv_tests = 1.0 / static_cast<double>(num_tests);
    for (size_t idx = 0; idx < ks.size(); ++idx) {
        const double ltr_hit = static_cast<double>(ltr_hit_counts[idx]) * inv_tests;
        const double rtl_hit = static_cast<double>(rtl_hit_counts[idx]) * inv_tests;
        scores.hits[ks[idx]] = 0.5 * (ltr_hit + rtl_hit);
    }
    scores.mrr = 0.5 * (ltr_mrr_sum + rtl_mrr_sum) * inv_tests;
    return scores;
}

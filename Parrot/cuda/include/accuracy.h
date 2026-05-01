#pragma once

#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

struct AccuracyScores {
    std::unordered_map<int, double> hits;
    double mrr = 0.0;
};

std::unordered_map<int, double> hits_ks_ltr_scores(
    const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test,
    const std::vector<int>& ks);

std::unordered_map<int, double> hits_ks_rtl_scores(
    const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test,
    const std::vector<int>& ks);

std::unordered_map<int, double> hits_ks_mean_scores(
    const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test,
    const std::vector<int>& ks);

double mrr_ltr_score_cpp(const std::vector<double>& S, int n1, int n2,
                         const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test);
double mrr_rtl_score_cpp(const std::vector<double>& S, int n1, int n2,
                         const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test);
double mrr_mean_score_cpp(const std::vector<double>& S, int n1, int n2,
                          const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test);

AccuracyScores accuracy_mean_scores_cpp(
    const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test,
    const std::vector<int>& ks);

#pragma once

#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

struct AccuracyScores {
    std::unordered_map<int, double> hits;
    double mrr = 0.0;
};

AccuracyScores accuracy_mean_scores_cpp(
    const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs_test,
    const std::vector<int>& ks);

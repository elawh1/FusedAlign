#include <mkl.h>
#include <mkl_vml.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Real = float;
using Clock = std::chrono::high_resolution_clock;

struct Config {
    MKL_INT n1 = 0;
    MKL_INT n2 = 0;
    bool use_attr = false;
    bool directed = false;
    int inner_iters = 5;
    int outer_iters = 10;
};

struct Anchors {
    std::vector<MKL_INT> left;
    std::vector<MKL_INT> right;
};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

long long ms_since(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

Config load_config(const std::string& dir) {
    std::ifstream in(dir + "/config.txt");
    require(static_cast<bool>(in), "could not open config.txt");

    Config cfg;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        if (key == "g1_num_nodes") cfg.n1 = static_cast<MKL_INT>(std::stoll(value));
        else if (key == "g2_num_nodes") cfg.n2 = static_cast<MKL_INT>(std::stoll(value));
        else if (key == "use_attr") cfg.use_attr = (std::stoi(value) != 0);
        else if (key == "directed") cfg.directed = (std::stoi(value) != 0);
        else if (key == "inner_iter") cfg.inner_iters = std::stoi(value);
        else if (key == "outer_iter") cfg.outer_iters = std::stoi(value);
    }
    require(cfg.n1 > 0 && cfg.n2 > 0, "config.txt must define positive graph sizes");
    return cfg;
}

std::vector<Real> read_values(const std::string& path) {
    std::ifstream in(path);
    require(static_cast<bool>(in), "could not open " + path);

    std::vector<Real> values;
    double value = 0.0;
    while (in >> value) values.push_back(static_cast<Real>(value));
    return values;
}

Anchors read_pairs(const std::string& path) {
    const std::vector<Real> raw = read_values(path);
    require(raw.size() % 2 == 0, path + " must contain pairs");

    Anchors pairs;
    pairs.left.reserve(raw.size() / 2);
    pairs.right.reserve(raw.size() / 2);
    for (std::size_t k = 0; k < raw.size(); k += 2) {
        pairs.left.push_back(static_cast<MKL_INT>(raw[k]));
        pairs.right.push_back(static_cast<MKL_INT>(raw[k + 1]));
    }
    return pairs;
}

std::vector<Real> read_graph_dense(const std::string& path, MKL_INT n, bool directed, MKL_INT& avg_edges_per_node) {
    std::ifstream in(path);
    require(static_cast<bool>(in), "could not open " + path);

    std::vector<Real> graph(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), Real(0));
    long long edges = 0;
    long long u = 0;
    long long v = 0;
    while (in >> u >> v) {
        require(u >= 0 && u < n && v >= 0 && v < n, "edge index out of bounds in " + path);
        graph[static_cast<std::size_t>(u) * n + static_cast<std::size_t>(v)] = Real(1);
        ++edges;
        if (!directed && u != v) {
            graph[static_cast<std::size_t>(v) * n + static_cast<std::size_t>(u)] = Real(1);
            ++edges;
        }
    }
    avg_edges_per_node = static_cast<MKL_INT>(edges / std::max<long long>(1, n));
    return graph;
}

std::vector<Real> row_normalize_graph(std::vector<Real> graph, MKL_INT n) {
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < n; ++i) {
        Real sum = Real(0);
        for (MKL_INT j = 0; j < n; ++j) sum += graph[static_cast<std::size_t>(i) * n + j];
        if (sum <= Real(1e-10)) {
            for (MKL_INT j = 0; j < n; ++j) graph[static_cast<std::size_t>(i) * n + j] = Real(1);
            sum = static_cast<Real>(n);
        }
        const Real inv_sum = Real(1) / sum;
        for (MKL_INT j = 0; j < n; ++j) graph[static_cast<std::size_t>(i) * n + j] *= inv_sum;
    }
    return graph;
}

void gemm_row(CBLAS_TRANSPOSE trans_a,
              CBLAS_TRANSPOSE trans_b,
              MKL_INT m,
              MKL_INT n,
              MKL_INT k,
              Real alpha,
              const std::vector<Real>& a,
              MKL_INT lda,
              const std::vector<Real>& b,
              MKL_INT ldb,
              Real beta,
              std::vector<Real>& c,
              MKL_INT ldc) {
    cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha,
                a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
}

void l2_normalize_rows(std::vector<Real>& matrix, MKL_INT rows, MKL_INT cols) {
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows; ++i) {
        Real sq_sum = Real(0);
        for (MKL_INT j = 0; j < cols; ++j) {
            const Real x = matrix[static_cast<std::size_t>(i) * cols + j];
            sq_sum += x * x;
        }
        if (sq_sum > Real(0)) {
            const Real inv = Real(1) / std::sqrt(sq_sum);
            for (MKL_INT j = 0; j < cols; ++j) matrix[static_cast<std::size_t>(i) * cols + j] *= inv;
        }
    }
}

std::vector<Real> batch_rwr(const std::vector<Real>& row_norm_graph,
                            const std::vector<MKL_INT>& anchors,
                            MKL_INT n,
                            Real restart_prob,
                            int max_iters) {
    const MKL_INT l = static_cast<MKL_INT>(anchors.size());
    std::vector<Real> x(static_cast<std::size_t>(n) * l, Real(0));
    std::vector<Real> next(x.size(), Real(0));

    for (MKL_INT j = 0; j < l; ++j) {
        const MKL_INT anchor = anchors[j];
        require(anchor >= 0 && anchor < n, "anchor out of bounds in RWR");
        x[static_cast<std::size_t>(anchor) * l + j] = restart_prob;
    }

    for (int iter = 1; iter < max_iters; ++iter) {
        gemm_row(CblasNoTrans, CblasNoTrans, n, l, n, Real(1), row_norm_graph, n, x, l, Real(0), next, l);
        for (MKL_INT j = 0; j < l; ++j) next[static_cast<std::size_t>(anchors[j]) * l + j] += restart_prob;
        x.swap(next);
    }
    return x;
}

std::vector<Real> neg_exp_dot_dist(const std::vector<Real>& left,
                                   const std::vector<Real>& right,
                                   MKL_INT n_left,
                                   MKL_INT n_right,
                                   MKL_INT dim) {
    std::vector<Real> out(static_cast<std::size_t>(n_left) * n_right, Real(0));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_left, n_right, dim,
                Real(-1), left.data(), dim, right.data(), dim, Real(0), out.data(), n_right);
    vsExp(static_cast<MKL_INT>(out.size()), out.data(), out.data());
    return out;
}

void zero_anchor_costs(std::vector<Real>& cost, MKL_INT n1, MKL_INT n2, const Anchors& anchors) {
    require(anchors.left.size() == anchors.right.size(), "anchor size mismatch");
    for (std::size_t t = 0; t < anchors.left.size(); ++t) {
        const MKL_INT i = anchors.left[t];
        const MKL_INT j = anchors.right[t];
        require(i >= 0 && i < n1 && j >= 0 && j < n2, "anchor out of bounds");
        cost[static_cast<std::size_t>(i) * n2 + j] = Real(0);
    }
}

std::vector<Real> masked_intra_dist(const std::vector<Real>& emb,
                                    const std::vector<Real>& adjacency,
                                    MKL_INT n,
                                    MKL_INT dim) {
    std::vector<Real> dist = neg_exp_dot_dist(emb, emb, n, n, dim);
    #pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < dist.size(); ++idx) dist[idx] *= adjacency[idx];
    return dist;
}

std::vector<Real> product_rwr(const std::vector<Real>& adj1,
                              const std::vector<Real>& adj2,
                              const std::vector<Real>& input,
                              MKL_INT n1,
                              MKL_INT n2,
                              Real restart_prob,
                              Real gamma,
                              int max_iters,
                              Real tol) {
    const std::size_t total = static_cast<std::size_t>(n1) * n2;
    std::vector<Real> current(total, Real(0));
    std::vector<Real> next(total, Real(0));
    std::vector<Real> temp(total, Real(0));
    const Real input_scale = Real(1) + restart_prob;
    const Real walk_scale = (Real(1) - restart_prob) * gamma;

    for (int iter = 0; iter < max_iters; ++iter) {
        gemm_row(CblasNoTrans, CblasNoTrans, n1, n2, n1, Real(1), adj1, n1, current, n2, Real(0), temp, n2);
        gemm_row(CblasNoTrans, CblasTrans, n1, n2, n2, walk_scale, temp, n2, adj2, n2, Real(0), next, n2);

        Real max_delta = Real(0);
        #pragma omp parallel for reduction(max:max_delta) schedule(static)
        for (std::size_t idx = 0; idx < total; ++idx) {
            next[idx] += input_scale * input[idx];
            max_delta = std::max(max_delta, std::abs(next[idx] - current[idx]));
        }
        current.swap(next);
        if (max_delta < tol) break;
    }

    cblas_sscal(static_cast<MKL_INT>(current.size()), Real(1) - gamma, current.data(), 1);
    return current;
}

void row_square_means(const std::vector<Real>& matrix, MKL_INT n, std::vector<Real>& means) {
    means.assign(n, Real(0));
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < n; ++i) {
        Real sum = Real(0);
        for (MKL_INT j = 0; j < n; ++j) {
            const Real x = matrix[static_cast<std::size_t>(i) * n + j];
            sum += x * x;
        }
        means[i] = sum / static_cast<Real>(n);
    }
}

void build_fixed_cost(std::vector<Real>& cross_dist,
                      const std::vector<Real>& intra1,
                      const std::vector<Real>& intra2,
                      MKL_INT n1,
                      MKL_INT n2,
                      Real lambda_e,
                      Real lambda_a,
                      const Anchors& anchors) {
    std::vector<Real> left_means;
    std::vector<Real> right_means;
    row_square_means(intra1, n1, left_means);
    row_square_means(intra2, n2, right_means);

    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < n1; ++i) {
        for (MKL_INT j = 0; j < n2; ++j) {
            cross_dist[static_cast<std::size_t>(i) * n2 + j] += lambda_e * (left_means[i] + right_means[j]);
        }
    }

    const Real anchor_adjust = lambda_a * static_cast<Real>(std::log(1e-6));
    for (std::size_t t = 0; t < anchors.left.size(); ++t) {
        cross_dist[static_cast<std::size_t>(anchors.left[t]) * n2 + anchors.right[t]] -= anchor_adjust;
    }
}

void stable_min_columns(const std::vector<Real>& cost,
                        const std::vector<Real>& row_offsets,
                        MKL_INT rows,
                        MKL_INT cols,
                        Real epsilon,
                        std::vector<Real>& out) {
    out.assign(cols, Real(0));
    const Real inv_eps = Real(1) / epsilon;
    const Real inv_rows = Real(1) / static_cast<Real>(rows);

    #pragma omp parallel for schedule(dynamic, 4)
    for (MKL_INT j = 0; j < cols; ++j) {
        Real min_value = std::numeric_limits<Real>::infinity();
        for (MKL_INT i = 0; i < rows; ++i) {
            min_value = std::min(min_value, cost[static_cast<std::size_t>(i) * cols + j] - row_offsets[i]);
        }
        Real sum = Real(0);
        for (MKL_INT i = 0; i < rows; ++i) {
            const Real shifted = cost[static_cast<std::size_t>(i) * cols + j] - row_offsets[i] - min_value;
            sum += std::exp(-shifted * inv_eps);
        }
        out[j] = min_value - epsilon * std::log(sum * inv_rows);
    }
}

void stable_min_rows(const std::vector<Real>& cost,
                     const std::vector<Real>& col_offsets,
                     MKL_INT rows,
                     MKL_INT cols,
                     Real epsilon,
                     std::vector<Real>& out) {
    out.assign(rows, Real(0));
    const Real inv_eps = Real(1) / epsilon;
    const Real inv_cols = Real(1) / static_cast<Real>(cols);

    #pragma omp parallel for schedule(dynamic, 4)
    for (MKL_INT i = 0; i < rows; ++i) {
        Real min_value = std::numeric_limits<Real>::infinity();
        for (MKL_INT j = 0; j < cols; ++j) {
            min_value = std::min(min_value, cost[static_cast<std::size_t>(i) * cols + j] - col_offsets[j]);
        }
        Real sum = Real(0);
        for (MKL_INT j = 0; j < cols; ++j) {
            const Real shifted = cost[static_cast<std::size_t>(i) * cols + j] - col_offsets[j] - min_value;
            sum += std::exp(-shifted * inv_eps);
        }
        out[i] = min_value - epsilon * std::log(sum * inv_cols);
    }
}

std::vector<Real> optimize_alignment(const std::vector<Real>& adj1,
                                     const std::vector<Real>& adj2,
                                     std::vector<Real> cross_dist,
                                     const std::vector<Real>& intra1,
                                     const std::vector<Real>& intra2,
                                     MKL_INT n1,
                                     MKL_INT n2,
                                     int inner_iters,
                                     int outer_iters,
                                     const Anchors& anchors) {
    const std::size_t total = static_cast<std::size_t>(n1) * n2;
    const Real lambda_n = Real(5e-3);
    const Real lambda_a = Real(5e-4);
    const Real lambda_p = Real(5e-4);
    const Real lambda_e = Real(1e-5) * static_cast<Real>(total);
    const Real epsilon = lambda_n + lambda_a + lambda_p;

    build_fixed_cost(cross_dist, intra1, intra2, n1, n2, lambda_e, lambda_a, anchors);

    std::vector<Real> s(total, Real(1) / static_cast<Real>(total));
    std::vector<Real> old_s(total);
    std::vector<Real> cost(total);
    std::vector<Real> old_cost(total);
    std::vector<Real> temp(total);
    std::vector<Real> edge_term(total);
    std::vector<Real> neigh_term(total);
    std::vector<Real> a(n2, Real(0));
    std::vector<Real> b(n1, Real(1) / static_cast<Real>(n2));
    bool have_old_cost = false;

    for (int outer = 0; outer < outer_iters; ++outer) {
        old_s = s;

        gemm_row(CblasNoTrans, CblasTrans, n1, n2, n2, Real(1), s, n2, intra2, n2, Real(0), temp, n2);
        gemm_row(CblasNoTrans, CblasNoTrans, n1, n2, n1, Real(-2) * lambda_e, intra1, n1, temp, n2, Real(0), edge_term, n2);

        gemm_row(CblasNoTrans, CblasTrans, n1, n2, n2, Real(1), s, n2, adj2, n2, Real(0), temp, n2);
        gemm_row(CblasTrans, CblasNoTrans, n1, n2, n1, Real(1), adj1, n1, temp, n2, Real(0), neigh_term, n2);

        #pragma omp parallel for schedule(static)
        for (std::size_t idx = 0; idx < total; ++idx) {
            const Real neighborhood = std::max(neigh_term[idx], Real(1e-30));
            cost[idx] = cross_dist[idx] + edge_term[idx] - lambda_n * std::log(neighborhood);
        }

        if (!have_old_cost) {
            old_cost = cost;
            have_old_cost = true;
        } else {
            const Real old_score = cblas_sdot(static_cast<MKL_INT>(total), s.data(), 1, old_cost.data(), 1);
            const Real new_score = cblas_sdot(static_cast<MKL_INT>(total), s.data(), 1, cost.data(), 1);
            if (new_score <= old_score) old_cost = cost;
            else cost = old_cost;
        }

        #pragma omp parallel for schedule(static)
        for (std::size_t idx = 0; idx < total; ++idx) {
            cost[idx] -= lambda_p * std::log(std::max(s[idx], Real(1e-30)));
        }

        for (int inner = 0; inner < inner_iters; ++inner) {
            stable_min_columns(cost, b, n1, n2, epsilon, a);
            stable_min_rows(cost, a, n1, n2, epsilon, b);
        }

        #pragma omp parallel for schedule(static)
        for (MKL_INT i = 0; i < n1; ++i) {
            for (MKL_INT j = 0; j < n2; ++j) {
                const std::size_t idx = static_cast<std::size_t>(i) * n2 + j;
                const Real kernel = std::exp((a[j] + b[i] - cost[idx]) / epsilon) / static_cast<Real>(total);
                s[idx] = Real(0.95) * kernel + Real(0.05) * old_s[idx];
            }
        }
    }
    return s;
}

std::vector<Real> blend_cross_dist(std::vector<Real> rwr_dist,
                                   std::vector<Real> attr_dist,
                                   Real alpha) {
    require(rwr_dist.size() == attr_dist.size(), "cross distance size mismatch");
    #pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < rwr_dist.size(); ++idx) attr_dist[idx] += alpha * rwr_dist[idx];
    return attr_dist;
}

double hit_at_k_left_to_right(const std::vector<Real>& score, MKL_INT n1, MKL_INT n2, const Anchors& tests, int k) {
    int hits = 0;
    std::vector<MKL_INT> order(n2);
    for (std::size_t t = 0; t < tests.left.size(); ++t) {
        const MKL_INT i = tests.left[t];
        const MKL_INT target = tests.right[t];
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + std::min<int>(k, n2), order.end(),
                          [&](MKL_INT a, MKL_INT b) {
                              return score[static_cast<std::size_t>(i) * n2 + a] > score[static_cast<std::size_t>(i) * n2 + b];
                          });
        for (int r = 0; r < std::min<int>(k, n2); ++r) {
            if (order[r] == target) {
                ++hits;
                break;
            }
        }
    }
    return tests.left.empty() ? 0.0 : static_cast<double>(hits) / static_cast<double>(tests.left.size());
}

double hit_at_k_right_to_left(const std::vector<Real>& score, MKL_INT n1, MKL_INT n2, const Anchors& tests, int k) {
    int hits = 0;
    std::vector<MKL_INT> order(n1);
    for (std::size_t t = 0; t < tests.left.size(); ++t) {
        const MKL_INT target = tests.left[t];
        const MKL_INT j = tests.right[t];
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + std::min<int>(k, n1), order.end(),
                          [&](MKL_INT a, MKL_INT b) {
                              return score[static_cast<std::size_t>(a) * n2 + j] > score[static_cast<std::size_t>(b) * n2 + j];
                          });
        for (int r = 0; r < std::min<int>(k, n1); ++r) {
            if (order[r] == target) {
                ++hits;
                break;
            }
        }
    }
    return tests.left.empty() ? 0.0 : static_cast<double>(hits) / static_cast<double>(tests.left.size());
}

double mean_reciprocal_rank(const std::vector<Real>& score, MKL_INT n1, MKL_INT n2, const Anchors& tests) {
    double ltr = 0.0;
    double rtl = 0.0;
    std::vector<MKL_INT> right_order(n2);
    std::vector<MKL_INT> left_order(n1);

    for (std::size_t t = 0; t < tests.left.size(); ++t) {
        const MKL_INT i = tests.left[t];
        const MKL_INT j = tests.right[t];

        std::iota(right_order.begin(), right_order.end(), 0);
        std::sort(right_order.begin(), right_order.end(), [&](MKL_INT a, MKL_INT b) {
            return score[static_cast<std::size_t>(i) * n2 + a] > score[static_cast<std::size_t>(i) * n2 + b];
        });
        for (MKL_INT r = 0; r < n2; ++r) {
            if (right_order[r] == j) {
                ltr += 1.0 / static_cast<double>(r + 1);
                break;
            }
        }

        std::iota(left_order.begin(), left_order.end(), 0);
        std::sort(left_order.begin(), left_order.end(), [&](MKL_INT a, MKL_INT b) {
            return score[static_cast<std::size_t>(a) * n2 + j] > score[static_cast<std::size_t>(b) * n2 + j];
        });
        for (MKL_INT r = 0; r < n1; ++r) {
            if (left_order[r] == i) {
                rtl += 1.0 / static_cast<double>(r + 1);
                break;
            }
        }
    }

    if (tests.left.empty()) return 0.0;
    const double denom = static_cast<double>(tests.left.size());
    return 0.5 * (ltr / denom + rtl / denom);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "provide input directory\n";
            return 1;
        }

        const std::string data_dir = argv[1];
        const Config cfg = load_config(data_dir);
        std::cout << "----- Start -----\n";
        std::cout << "dataset: " << data_dir << "\n";
        std::cout << "MKL threads: " << mkl_get_max_threads() << "\n";
        std::cout << "graph1 #nodes " << cfg.n1 << " -- graph2 #nodes " << cfg.n2
                  << " -- directed " << cfg.directed << " -- use node attributes " << cfg.use_attr << "\n\n";

        std::vector<std::pair<std::string, long long>> timings;
        MKL_INT avg1 = 0;
        MKL_INT avg2 = 0;
        std::vector<Real> graph1 = read_graph_dense(data_dir + "/graph1.txt", cfg.n1, cfg.directed, avg1);
        std::vector<Real> graph2 = read_graph_dense(data_dir + "/graph2.txt", cfg.n2, cfg.directed, avg2);
        const Anchors train = read_pairs(data_dir + "/anchor_links.txt");
        const Anchors test = read_pairs(data_dir + "/test_pairs.txt");
        const MKL_INT anchor_count = static_cast<MKL_INT>(train.left.size());

        std::vector<Real> feat1;
        std::vector<Real> feat2;
        MKL_INT feat_dim = 0;
        if (cfg.use_attr) {
            feat1 = read_values(data_dir + "/graph1_feat.txt");
            feat2 = read_values(data_dir + "/graph2_feat.txt");
            require(feat1.size() % static_cast<std::size_t>(cfg.n1) == 0, "graph1 feature size mismatch");
            require(feat2.size() % static_cast<std::size_t>(cfg.n2) == 0, "graph2 feature size mismatch");
            feat_dim = static_cast<MKL_INT>(feat1.size() / static_cast<std::size_t>(cfg.n1));
            require(feat_dim == static_cast<MKL_INT>(feat2.size() / static_cast<std::size_t>(cfg.n2)), "feature dimension mismatch");
        }

        auto start = Clock::now();
        std::vector<Real> norm1 = row_normalize_graph(std::move(graph1), cfg.n1);
        std::vector<Real> norm2 = row_normalize_graph(std::move(graph2), cfg.n2);
        timings.push_back({"row normalizing graphs", ms_since(start)});

        start = Clock::now();
        std::vector<Real> rwr1 = batch_rwr(norm1, train.left, cfg.n1, Real(0.15), 100);
        timings.push_back({"graph1 RWR", ms_since(start)});

        start = Clock::now();
        std::vector<Real> rwr2 = batch_rwr(norm2, train.right, cfg.n2, Real(0.15), 100);
        timings.push_back({"graph2 RWR", ms_since(start)});

        l2_normalize_rows(rwr1, cfg.n1, anchor_count);
        l2_normalize_rows(rwr2, cfg.n2, anchor_count);

        start = Clock::now();
        std::vector<Real> cross_rwr = neg_exp_dot_dist(rwr1, rwr2, cfg.n1, cfg.n2, anchor_count);
        std::vector<Real> cross_node;
        if (cfg.use_attr) {
            l2_normalize_rows(feat1, cfg.n1, feat_dim);
            l2_normalize_rows(feat2, cfg.n2, feat_dim);
            cross_node = blend_cross_dist(std::move(cross_rwr),
                                          neg_exp_dot_dist(feat1, feat2, cfg.n1, cfg.n2, feat_dim),
                                          Real(0.5));
        } else {
            cblas_sscal(static_cast<MKL_INT>(cross_rwr.size()), Real(1.5), cross_rwr.data(), 1);
            cross_node = std::move(cross_rwr);
        }
        zero_anchor_costs(cross_node, cfg.n1, cfg.n2, train);
        timings.push_back({"cross RWR distance", ms_since(start)});

        start = Clock::now();
        std::vector<Real> intra1 = cfg.use_attr
            ? masked_intra_dist(feat1, norm1, cfg.n1, feat_dim)
            : masked_intra_dist(rwr1, norm1, cfg.n1, anchor_count);
        timings.push_back({"graph1 intra distance", ms_since(start)});

        start = Clock::now();
        std::vector<Real> intra2 = cfg.use_attr
            ? masked_intra_dist(feat2, norm2, cfg.n2, feat_dim)
            : masked_intra_dist(rwr2, norm2, cfg.n2, anchor_count);
        timings.push_back({"graph2 intra distance", ms_since(start)});

        rwr1.clear();
        rwr2.clear();
        feat1.clear();
        feat2.clear();

        start = Clock::now();
        std::vector<Real> cross_dist = product_rwr(norm1, norm2, cross_node, cfg.n1, cfg.n2,
                                                   Real(0.15), Real(0.75), 50, Real(1e-2));
        timings.push_back({"cross distance propagation", ms_since(start)});
        cross_node.clear();

        start = Clock::now();
        std::vector<Real> score = optimize_alignment(norm1, norm2, std::move(cross_dist), intra1, intra2,
                                                     cfg.n1, cfg.n2, cfg.inner_iters, cfg.outer_iters, train);
        timings.push_back({"optimization", ms_since(start)});

        long long total = 0;
        for (const auto& phase : timings) total += phase.second;
        std::cout << "algorithm completed in " << total << " ms\n";
        std::cout << "phase timings:\n";
        for (const auto& phase : timings) {
            std::cout << "  " << phase.first << ": " << phase.second << " ms\n";
        }

        for (int k : {1, 10, 30, 50}) {
            const double ltr = hit_at_k_left_to_right(score, cfg.n1, cfg.n2, test, k);
            const double rtl = hit_at_k_right_to_left(score, cfg.n1, cfg.n2, test, k);
            std::cout << "hit@ " << k << " : " << 0.5 * (ltr + rtl) << "\n";
        }
        std::cout << "mrr : " << mean_reciprocal_rank(score, cfg.n1, cfg.n2, test) << "\n";

        for (std::size_t i = 0; i < timings.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << std::fixed << std::setprecision(2) << static_cast<double>(timings[i].second) / 1000.0;
        }
        std::cout << "\n----- End -----\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}

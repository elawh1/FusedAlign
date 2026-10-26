#include "helper.h"
#include "helper2.h"
#include <ctime>

namespace {

template <typename Real>
void dense_left_multiply_colmajor(const std::vector<Real>& A, const std::vector<Real>& B,
                                  MKL_INT rows, MKL_INT inner, MKL_INT cols, Real alpha,
                                  std::vector<Real>& C) {
    if (C.size() != (size_t)rows * (size_t)cols) C.resize((size_t)rows * (size_t)cols);
    if constexpr (std::is_same_v<Real, float>) {
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, rows, cols, inner, alpha,
                    A.data(), inner, B.data(), inner, Real(0), C.data(), rows);
    } else {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rows, cols, inner, alpha,
                    A.data(), inner, B.data(), inner, Real(0), C.data(), rows);
    }
}

template <typename Real>
void dense_left_transpose_multiply_colmajor(const std::vector<Real>& A, const std::vector<Real>& B,
                                            MKL_INT rows, MKL_INT cols, Real alpha,
                                            std::vector<Real>& C) {
    if (C.size() != (size_t)rows * (size_t)cols) C.resize((size_t)rows * (size_t)cols);
    if constexpr (std::is_same_v<Real, float>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, cols, rows, alpha,
                    A.data(), rows, B.data(), rows, Real(0), C.data(), rows);
    } else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, cols, rows, alpha,
                    A.data(), rows, B.data(), rows, Real(0), C.data(), rows);
    }
}

template <typename Real>
void dense_left_multiply_colmajor_matrix(const std::vector<Real>& A, const std::vector<Real>& B,
                                         MKL_INT rows, MKL_INT inner, MKL_INT cols, Real alpha,
                                         std::vector<Real>& C) {
    if (C.size() != (size_t)rows * (size_t)cols) C.resize((size_t)rows * (size_t)cols);
    if constexpr (std::is_same_v<Real, float>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, cols, inner, alpha,
                    A.data(), rows, B.data(), inner, Real(0), C.data(), rows);
    } else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, cols, inner, alpha,
                    A.data(), rows, B.data(), inner, Real(0), C.data(), rows);
    }
}

template <typename Real>
void dense_right_multiply_transpose_colmajor(const std::vector<Real>& B, const std::vector<Real>& A,
                                             MKL_INT rows, MKL_INT cols, Real alpha,
                                             std::vector<Real>& C) {
    if (C.size() != (size_t)rows * (size_t)cols) C.resize((size_t)rows * (size_t)cols);
    if constexpr (std::is_same_v<Real, float>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, cols, cols, alpha,
                    B.data(), rows, A.data(), cols, Real(0), C.data(), rows);
    } else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, cols, cols, alpha,
                    B.data(), rows, A.data(), cols, Real(0), C.data(), rows);
    }
}

template <typename Real>
void dense_right_multiply_transpose_colmajor_matrix(const std::vector<Real>& B, const std::vector<Real>& A,
                                                    MKL_INT rows, MKL_INT cols, Real alpha,
                                                    std::vector<Real>& C) {
    if (C.size() != (size_t)rows * (size_t)cols) C.resize((size_t)rows * (size_t)cols);
    if constexpr (std::is_same_v<Real, float>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, rows, cols, cols, alpha,
                    B.data(), rows, A.data(), cols, Real(0), C.data(), rows);
    } else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, rows, cols, cols, alpha,
                    B.data(), rows, A.data(), cols, Real(0), C.data(), rows);
    }
}

template <typename Real>
void mkl_imatcopy_real(char ordering, char trans, size_t rows, size_t cols, Real alpha, Real* ab, size_t lda, size_t ldb);

template <>
void mkl_imatcopy_real<double>(char ordering, char trans, size_t rows, size_t cols, double alpha, double* ab, size_t lda, size_t ldb) {
    mkl_dimatcopy(ordering, trans, rows, cols, alpha, ab, lda, ldb);
}

template <>
void mkl_imatcopy_real<float>(char ordering, char trans, size_t rows, size_t cols, float alpha, float* ab, size_t lda, size_t ldb) {
    mkl_simatcopy(ordering, trans, rows, cols, alpha, ab, lda, ldb);
}

template <typename Real>
void mkl_omatcopy_real(char ordering, char trans, size_t rows, size_t cols, Real alpha, const Real* a, size_t lda, Real* b, size_t ldb);

template <>
void mkl_omatcopy_real<double>(char ordering, char trans, size_t rows, size_t cols, double alpha, const double* a, size_t lda, double* b, size_t ldb) {
    mkl_domatcopy(ordering, trans, rows, cols, alpha, a, lda, b, ldb);
}

template <>
void mkl_omatcopy_real<float>(char ordering, char trans, size_t rows, size_t cols, float alpha, const float* a, size_t lda, float* b, size_t ldb) {
    mkl_somatcopy(ordering, trans, rows, cols, alpha, a, lda, b, ldb);
}

template <typename Real>
void cblas_copy_real(size_t n, const Real* x, Real* y) {
    if constexpr (std::is_same_v<Real, float>) {
        cblas_scopy((MKL_INT)n, x, 1, y, 1);
    } else {
        cblas_dcopy((MKL_INT)n, x, 1, y, 1);
    }
}

template <typename Real>
Real cblas_dot_real(size_t n, const Real* x, const Real* y) {
    if constexpr (std::is_same_v<Real, float>) {
        return cblas_sdot((MKL_INT)n, x, 1, y, 1);
    } else {
        return cblas_ddot((MKL_INT)n, x, 1, y, 1);
    }
}

template <typename Real>
void cblas_axpy_real(size_t n, Real alpha, const Real* x, Real* y) {
    if constexpr (std::is_same_v<Real, float>) {
        cblas_saxpy((MKL_INT)n, alpha, x, 1, y, 1);
    } else {
        cblas_daxpy((MKL_INT)n, alpha, x, 1, y, 1);
    }
}

template <typename Real>
void vln_real(size_t n, const Real* x, Real* y) {
    if constexpr (std::is_same_v<Real, float>) {
        vsLn((MKL_INT)n, x, y);
    } else {
        vdLn((MKL_INT)n, x, y);
    }
}

}  // namespace

template <typename Real>
void l2_norm_inplace(std::vector<Real>& rwr, size_t n, size_t d) {
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    std::vector<Real> norms(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        Real norm;
        if constexpr (std::is_same_v<Real, float>) {
            norm = cblas_snrm2((MKL_INT)d, &rwr[i], (MKL_INT)n);
        } else {
            norm = cblas_dnrm2((MKL_INT)d, &rwr[i], (MKL_INT)n);
        }
        norms[i] = (norm > Real(0)) ? Real(1) / norm : Real(0);
    }
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        if (norms[i] != Real(0)) {
            if constexpr (std::is_same_v<Real, float>) {
                cblas_sscal((MKL_INT)d, norms[i], &rwr[i], (MKL_INT)n);
            } else {
                cblas_dscal((MKL_INT)d, norms[i], &rwr[i], (MKL_INT)n);
            }
        }
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
}

template void l2_norm_inplace<double>(std::vector<double>& rwr, size_t n, size_t d);
template void l2_norm_inplace<float>(std::vector<float>& rwr, size_t n, size_t d);

template <typename Real>
void get_normalized_neg_exp_dist (const std::vector<Real>& rwr1, const std::vector<Real>&  rwr2, int n1, int n2, int l, std::vector<Real>& cross_rwr_dist) {
    if constexpr (std::is_same_v<Real, float>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, (MKL_INT)n1, (MKL_INT)n2, (MKL_INT)l, -1.0f, rwr1.data(), (MKL_INT)n1,
                rwr2.data(), (MKL_INT)n2, 0.0f, cross_rwr_dist.data(), (MKL_INT)n1);
    } else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, (MKL_INT)n1, (MKL_INT)n2, (MKL_INT)l, -1.0, rwr1.data(), (MKL_INT)n1,
                rwr2.data(), (MKL_INT)n2, 0.0, cross_rwr_dist.data(), (MKL_INT)n1);
    }
    size_t total = (size_t)n1 * (size_t)n2;
    if constexpr (std::is_same_v<Real, float>) {
        vsExp((MKL_INT)total, cross_rwr_dist.data(), cross_rwr_dist.data());
    } else {
        vdExp((MKL_INT)total, cross_rwr_dist.data(), cross_rwr_dist.data());
    }
}

template void get_normalized_neg_exp_dist<double>(const std::vector<double>& rwr1, const std::vector<double>& rwr2, int n1, int n2, int l, std::vector<double>& cross_rwr_dist);
template void get_normalized_neg_exp_dist<float>(const std::vector<float>& rwr1, const std::vector<float>& rwr2, int n1, int n2, int l, std::vector<float>& cross_rwr_dist);

template <typename Real>
std::vector<Real> make_mat_from_edgelist(const std::vector<Real>& edgelist, int m, int n) {
    std::vector<Real> A((size_t)m * (size_t)n, Real(0));
    for (size_t e = 0; e + 1 < edgelist.size(); e += 2) {
        MKL_INT row = static_cast<MKL_INT>(edgelist[e]);
        MKL_INT col = static_cast<MKL_INT>(edgelist[e + 1]);
        if (row >= 0 && row < m && col >= 0 && col < n) {
            A[(size_t)row * (size_t)n + (size_t)col] = Real(1);
        }
    }
    return A;
}

template std::vector<double> make_mat_from_edgelist<double>(const std::vector<double>& edgelist, int m, int n);
template std::vector<float> make_mat_from_edgelist<float>(const std::vector<float>& edgelist, int m, int n);

template <typename Real>
void masked_dense_intra_cost(std::vector<Real>& emb, const std::vector<Real>& mask, int n, int L, std::vector<Real>& result) {
    if (mask.size() != (size_t)n * (size_t)n) throw std::runtime_error("mask dimensions do not match n");
    if (emb.size() != (size_t)n * (size_t)L) throw std::runtime_error("emb size mismatch");
    result.assign((size_t)n * (size_t)n, Real(0));
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for schedule(static)
    for (MKL_INT j = 0; j < n; ++j) {
        for (MKL_INT i = 0; i < n; ++i) {
            if (mask[(size_t)i * (size_t)n + (size_t)j] == Real(0)) continue;
            Real product = Real(0);
            for (MKL_INT l = 0; l < L; ++l) {
                const Real* col = emb.data() + (size_t)l * (size_t)n;
                product += col[i] * col[j];
            }
            result[(size_t)j * (size_t)n + (size_t)i] = std::exp(-product);
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}

template void masked_dense_intra_cost<double>(std::vector<double>& emb, const std::vector<double>& mask, int n, int L, std::vector<double>& result);
template void masked_dense_intra_cost<float>(std::vector<float>& emb, const std::vector<float>& mask, int n, int L, std::vector<float>& result);

template <typename Real>
std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<Real>& H_vec) {
    std::vector<MKL_INT> anchor_g1;
    std::vector<MKL_INT> anchor_g2;
    for(size_t i=0; i< H_vec.size()/2; i++) {
        anchor_g1.push_back((MKL_INT) H_vec[i*2]);
        anchor_g2.push_back((MKL_INT) H_vec[i*2+1]);
    }
    return {anchor_g1, anchor_g2};
}

template std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs<double>(std::vector<double>& H_vec);
template std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs<float>(std::vector<float>& H_vec);

template <typename Real>
std::vector<Real> row_normalize_graph(const std::vector<Real>& graph, int n, MKL_INT avg_nnz_per_n) {
    if (graph.size() != (size_t)n * (size_t)n) throw std::runtime_error("graph size mismatch in row_normalize_graph");
    std::vector<Real> graph_connected = graph;
    std::vector<Real> graph_row_sum(n, Real(0));
    std::vector<MKL_INT> isolated;
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);  
    #pragma omp parallel for
    for (MKL_INT i = 0; i < n; ++i) {
        Real sum = Real(0);
        for (MKL_INT j = 0; j < n; ++j) {
            sum += graph_connected[(size_t)i * (size_t)n + (size_t)j];
        }
        graph_row_sum[i] = sum;
    }
    for (MKL_INT i = 0; i < n; ++i) {
        if (graph_row_sum[i] < Real(1e-10)) isolated.push_back(i);
    }
    if (!isolated.empty()) {
        std::mt19937 rng(std::time(0));
        std::uniform_int_distribution<MKL_INT> dist(0, n - 1);
        MKL_INT degree = std::max<MKL_INT>(avg_nnz_per_n, 1);
        for (MKL_INT i : isolated) {
            std::vector<MKL_INT> random_nodes;
            random_nodes.reserve(degree);
            while ((MKL_INT)random_nodes.size() < degree && (MKL_INT)random_nodes.size() < n - 1) {
                MKL_INT r = dist(rng);
                if (r != i && std::find(random_nodes.begin(), random_nodes.end(), r) == random_nodes.end()) {
                    random_nodes.push_back(r);
                    graph_connected[(size_t)i * (size_t)n + (size_t)r] = Real(1);
                    graph_connected[(size_t)r * (size_t)n + (size_t)i] = Real(1);
                }
            }
        }
    }
    #pragma omp parallel for
    for (MKL_INT i = 0; i < n; ++i) {
        Real sum = Real(0);
        for (MKL_INT j = 0; j < n; ++j) {
            sum += graph_connected[(size_t)i * (size_t)n + (size_t)j];
        }
        graph_row_sum[i] = sum;
        Real inv_sum = (graph_row_sum[i] > Real(1e-10)) ? Real(1) / graph_row_sum[i] : Real(0);
        for (MKL_INT j = 0; j < n; ++j) {
            graph_connected[(size_t)i * (size_t)n + (size_t)j] *= inv_sum;
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);  
    return graph_connected;
}

template std::vector<double> row_normalize_graph<double>(const std::vector<double>& graph, int n, MKL_INT avg_nnz_per_n);
template std::vector<float> row_normalize_graph<float>(const std::vector<float>& graph, int n, MKL_INT avg_nnz_per_n);

void get_batch_rwr_scores(const std::vector<double>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, 
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) {
    auto time_0 = std::chrono::high_resolution_clock::now();
    int l = anchors.size();
    const size_t N = (size_t)n * (size_t)l;
    std::vector<double> X_vals(N, 0.0);
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);     
    #pragma omp parallel for
    for (int j = 0; j < l; ++j) {
        X_vals[(size_t)anchors[j] + (size_t)j * (size_t)n] = restart_prob;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);   
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto add_after_mm_time = 0, mm_time = 0, swap_after_mm_time = 0;
    for (int i = 0; i < max_iters - 1; i++) {
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        dense_left_multiply_colmajor<double>(graph_norm_csr, X_vals, n, n, l, 1.0, Xnext_vals);
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        mkl_set_num_threads(1);
        omp_set_num_threads(numThreads);  
        #pragma omp parallel for
        for (int j = 0; j < l; ++j) {
            Xnext_vals[(size_t)anchors[j] + (size_t)j * (size_t)n] += restart_prob;
        }
        mkl_set_num_threads(numThreads);
        omp_set_num_threads(1);  
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        add_after_mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        std::swap(X_vals, Xnext_vals);
        auto time_1_3 = std::chrono::high_resolution_clock::now();
        swap_after_mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2).count();
    }

    std::swap(X_vals, Xnext_vals);
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1); 

    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_batch_rwr_scores generate E " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " dense mm time " << mm_time << " add after dense mm time " << add_after_mm_time <<
        " swap after dense mm time " << swap_after_mm_time << std::endl);
}

template <typename Real>
void get_batch_rwr_scores_rowmjr(const std::vector<Real>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
        bool directed, Real restart_prob, int max_iters, bool connect_isolated, std::vector<Real>& Xnext_vals) {
    (void)avg_nnz_per_n;
    (void)directed;
    (void)connect_isolated;
    auto time_0 = std::chrono::high_resolution_clock::now();
    int l = anchors.size();
    const size_t N = (size_t)n * (size_t)l;
    std::vector<Real> X_vals(N, Real(0));
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for
    for (int j = 0; j < l; ++j) {
        X_vals[(size_t)anchors[j] * (size_t)l + (size_t)j] = restart_prob;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto add_after_mm_time = 0, mm_time = 0, swap_after_mm_time = 0;
    for (int i = 0; i < max_iters - 1; i++) {
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        if constexpr (std::is_same_v<Real, float>) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, l, n, Real(1),
                        graph_norm_csr.data(), n, X_vals.data(), l, Real(0), Xnext_vals.data(), l);
        } else {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, l, n, Real(1),
                        graph_norm_csr.data(), n, X_vals.data(), l, Real(0), Xnext_vals.data(), l);
        }
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        mkl_set_num_threads(1);
        omp_set_num_threads(numThreads);
        #pragma omp parallel for
        for (int j = 0; j < l; ++j) {
            Xnext_vals[(size_t)anchors[j] * (size_t)l + (size_t)j] += restart_prob;
        }
        mkl_set_num_threads(numThreads);
        omp_set_num_threads(1);
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        add_after_mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        std::swap(X_vals, Xnext_vals);
        auto time_1_3 = std::chrono::high_resolution_clock::now();
        swap_after_mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2).count();
    }
    
    // std::swap(X_vals, Xnext_vals);
    auto time_2_0 = std::chrono::high_resolution_clock::now();
    // mkl_imatcopy_real<Real>('R', 'T', n, l, Real(1), Xnext_vals.data(), l, n);
    mkl_omatcopy_real<Real>('R', 'T', n, l, Real(1), X_vals.data(), l, Xnext_vals.data(), n); 

    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_batch_rwr_scores_rowmjr generate E " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop + copy " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " mkl_omatcopy_real time " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_2_0).count() <<
        " dense mm time " << mm_time << " add after dense mm time " << add_after_mm_time <<
        " swap after dense mm time " << swap_after_mm_time << std::endl);
}

template void get_batch_rwr_scores_rowmjr<double>(const std::vector<double>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
        bool directed, double restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals);
template void get_batch_rwr_scores_rowmjr<float>(const std::vector<float>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<float>& Xnext_vals);

template <typename Real>
void anchor_cost_zero(std::vector<Real>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>,
                    std::vector<MKL_INT>>& anchors, Real val) {
    const MKL_INT k = (MKL_INT)anchors.first.size();
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);               
    #pragma omp parallel for
    for (MKL_INT t = 0; t < k; ++t) {
        MKL_INT i = anchors.first[t];
        MKL_INT j = anchors.second[t];
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            printf("error anchor_cost_zero: out-of-range anchor (%d,%d)\n", i, j);
            continue;
        }
        size_t idx = (size_t)j * (size_t)nrows + (size_t)i;
        A[idx] = val;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);  
}

template void anchor_cost_zero<double>(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>,
                    std::vector<MKL_INT>>& anchors, double val);
template void anchor_cost_zero<float>(std::vector<float>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>,
                    std::vector<MKL_INT>>& anchors, float val);

// Then set H[row, col] = log(1e-6) for each anchor
template <typename Real>
std::vector<Real> init_Hlog_with_anchors(MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors) {
    size_t N = nrows * ncols;
    std::vector<Real> H(N, Real(0));
    const MKL_INT k = (MKL_INT) anchors.first.size();
    if (anchors.second.size() != (size_t)k) {
        throw std::invalid_argument("anchor size mismatch  inside init_Hlog_with_anchors\n");
    }
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (MKL_INT t = 0; t < k; ++t) {
        MKL_INT i = anchors.first[t];   // row index
        MKL_INT j = anchors.second[t];  // col index
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            printf("error init_H_with_anchors: out-of-range anchor (%d,%d)\n", i, j);
            continue;
        }
        size_t idx = (size_t) i + (size_t) j * (size_t) nrows;    // column-major: (row, col) -> index
        H[idx] = static_cast<Real>(std::log(1e-6));
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return H;
}

template std::vector<double> init_Hlog_with_anchors<double>(MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors);
template std::vector<float> init_Hlog_with_anchors<float>(MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors);

template <typename Real>
void get_product_rwr_mat_new(const std::vector<Real>& graph1_normalized, const std::vector<Real>& graph2_normalized,
    const std::vector<Real>& in_cross_dist, MKL_INT n1, MKL_INT n2, Real rwr_restart_prob, Real gamma,
    std::vector<Real>& out_cross_dist, int max_iters, Real tol) {
    size_t N = (size_t)n1 * (size_t)n2;
    if (out_cross_dist.size() != N || in_cross_dist.size() != N)    throw std::runtime_error("in cross dist computation size mismatch");
    // const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
    // size_t off = 0;
    auto time_0 = std::chrono::high_resolution_clock::now();
    std::vector<Real> out_cross_dist_old(N, Real(0));
    std::vector<Real> temp(N);
    Real coeff = (Real(1) - rwr_restart_prob) * gamma;
    Real alpha = Real(1) + rwr_restart_prob;
    unsigned int numThreads = mkl_get_max_threads();
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto time_mm1 = 0, time_mm2 = 0, time_convergence = 0;
    bool swapped_last = false;
    bool converged = true;
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for reduction(&:converged)
    for (size_t i = 0; i < N; ++i) {
        out_cross_dist_old[i] = alpha * in_cross_dist[i];
        converged &= (std::abs(out_cross_dist_old[i]) < tol);
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
    if (converged || max_iters <= 1) {
        cblas_copy_real<Real>(N, out_cross_dist_old.data(), out_cross_dist.data());
    }
    for (int i = 1; !converged && i < max_iters; ++i) {
        swapped_last = false;
        // graph1_normalized and graph2_normalized are row-major. Cross-distance buffers are column-major.
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        dense_right_multiply_transpose_colmajor<Real>(out_cross_dist_old, graph2_normalized, n1, n2, Real(1), temp);
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        time_mm1 += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();

        dense_left_multiply_colmajor<Real>(graph1_normalized, temp, n1, n1, n2, coeff, out_cross_dist);
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        time_mm2 += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        converged = true;
        omp_set_num_threads(numThreads);
        mkl_set_num_threads(1);
        #pragma omp parallel for reduction(&:converged)
        for (size_t i = 0; i < N; ++i) {
            out_cross_dist[i] += alpha * in_cross_dist[i];
            converged &= (std::abs(out_cross_dist[i] - out_cross_dist_old[i]) < tol);
        }
        if (converged) break;
        omp_set_num_threads(1);
        mkl_set_num_threads(numThreads);
        auto time_1_4 = std::chrono::high_resolution_clock::now();
        time_convergence += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_4 - time_1_2).count();
        std::swap(out_cross_dist, out_cross_dist_old);
        swapped_last = true;
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
    auto time_2 = std::chrono::high_resolution_clock::now();
    if (swapped_last) std::swap(out_cross_dist, out_cross_dist_old);
    dscal_chunked<Real>(N, Real(1) - gamma, out_cross_dist.data());
    auto time_3 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << std::endl);
    DEBUG_ONLY(std::cout << "in for loop 1st dense mm " << time_mm1 << " 2nd dense mm " << time_mm2 << " convergence " << time_convergence << std::endl);
}

template void get_product_rwr_mat_new<double>(const std::vector<double>& graph1_normalized, const std::vector<double>& graph2_normalized,
    const std::vector<double>& in_cross_dist, MKL_INT n1, MKL_INT n2, double rwr_restart_prob, double gamma,
    std::vector<double>& out_cross_dist, int max_iters, double tol);
template void get_product_rwr_mat_new<float>(const std::vector<float>& graph1_normalized, const std::vector<float>& graph2_normalized,
    const std::vector<float>& in_cross_dist, MKL_INT n1, MKL_INT n2, float rwr_restart_prob, float gamma,
    std::vector<float>& out_cross_dist, int max_iters, float tol);

template <typename Real>
void minaa_v3_dense( std::vector<Real>& Cost, const std::vector<Real>& b, int n_row, int n_col, Real epsilon, std::vector<Real>& out_vec ) {
    const unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    const Real inv_epsilon = Real(1) / epsilon;
    const Real inv_n_row = Real(1) / static_cast<Real>(n_row);
    const Real* __restrict__ bptr = b.data();
    const int nT = static_cast<int>(numThreads);
    std::vector<Real> scratch(static_cast<size_t>(nT) * n_row);
    
    #pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        Real* __restrict__ local = scratch.data() + (size_t)tid * n_row;
        #pragma omp for schedule(dynamic, 4)
        for (int j = 0; j < n_col; ++j)
        {
            const Real* __restrict__ col = Cost.data() + (size_t)j * n_row;
            Real min_j = std::numeric_limits<Real>::max();
            #pragma omp simd reduction(min : min_j)
            for (int i = 0; i < n_row; ++i)
            {
                const Real v = col[i] - bptr[i];
                local[i] = v;          // store shifted values
                min_j = (v < min_j) ? v : min_j;
            }
            Real s_j = Real(0);
            #pragma omp simd reduction(+ : s_j)
            for (int i = 0; i < n_row; ++i)
            {
                s_j += std::exp((min_j - local[i]) * inv_epsilon);
            }
            out_vec[j] = min_j - epsilon * (std::log(s_j) + std::log(inv_n_row));
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}

template void minaa_v3_dense<double>(std::vector<double>& Cost, const std::vector<double>& b, int n_row, int n_col, double epsilon, std::vector<double>& out_vec);
template void minaa_v3_dense<float>(std::vector<float>& Cost, const std::vector<float>& b, int n_row, int n_col, float epsilon, std::vector<float>& out_vec);

template <typename Real>
void row_avg_square(const std::vector<Real>& intra_dist, int n, std::vector<Real>& row_avg_sqr) {
    if (intra_dist.size() != (size_t)n * (size_t)n) throw std::runtime_error("intra_dist size mismatch in row_avg_square");
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < n; ++i) {
        Real square_sum = Real(0);
        for (MKL_INT j = 0; j < n; ++j) {
            Real val = intra_dist[(size_t)j * (size_t)n + (size_t)i];
            square_sum += val * val;
        }
        row_avg_sqr[i] = square_sum / static_cast<Real>(n);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}

template void row_avg_square<double>(const std::vector<double>& intra_dist, int n, std::vector<double>& row_avg_sqr);
template void row_avg_square<float>(const std::vector<float>& intra_dist, int n, std::vector<float>& row_avg_sqr);

template <typename Real>
void compute_Cost_fixed(const std::vector<Real>& intra_dist1, const std::vector<Real>& intra_dist2, std::vector<Real>& cross_dist,
        int n1, int n2, Real lambda_e, Real lambda_a, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs) {
    auto time_0 = std::chrono::high_resolution_clock::now();
    // 1) Square intra_dist1 and intra_dist2 elementwise
    std::vector<Real> row_avg_square_dist1(n1), row_avg_square_dist2(n2);
    row_avg_square<Real>(intra_dist1, n1, row_avg_square_dist1);
    row_avg_square<Real>(intra_dist2, n2, row_avg_square_dist2);
    auto time_1 = std::chrono::high_resolution_clock::now();
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        Real dist2_term =  row_avg_square_dist2[j];
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i ;
            cross_dist[idx] = cross_dist[idx] + lambda_e * (dist2_term + row_avg_square_dist1[i]); 
        }
    }
    const Real anchor_penalty = lambda_a * static_cast<Real>(std::log(1e-6));
    const MKL_INT k = static_cast<MKL_INT>(anchor_pairs.first.size());
    #pragma omp parallel for schedule(static)
    for (MKL_INT t = 0; t < k; ++t) {
        MKL_INT i = anchor_pairs.first[t];
        MKL_INT j = anchor_pairs.second[t];
        if (i >= 0 && i < n1 && j >= 0 && j < n2) {
            cross_dist[(size_t)j * (size_t)n1 + (size_t)i] -= anchor_penalty;
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in compute_Cost_fixed before loop  : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " after loop  : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << std::endl);
}

template void compute_Cost_fixed<double>(const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, std::vector<double>& cross_dist,
        int n1, int n2, double lambda_e, double lambda_a, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);
template void compute_Cost_fixed<float>(const std::vector<float>& intra_dist1, const std::vector<float>& intra_dist2, std::vector<float>& cross_dist,
        int n1, int n2, float lambda_e, float lambda_a, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);

template <typename Real>
void compute_cost_fused_2(const std::vector<Real>& row_norm_adj1, const std::vector<Real>& row_norm_adj2, const std::vector<Real>& S, int n1, int n2,
        const std::vector<Real>& intra_dist1, const std::vector<Real>& intra_dist2, const std::vector<Real>& C_fixed, Real lambda_e, Real lambda_n, std::vector<Real>& C,
        std::vector<Real>& buffer, std::vector<Real>& buffer2) {
    size_t matrix_size = (size_t)n1 * n2;
    if (buffer.size() != matrix_size) buffer.resize(matrix_size);
    if (buffer2.size() != matrix_size) buffer2.resize(matrix_size);

    auto time_0 = std::chrono::high_resolution_clock::now();
    dense_right_multiply_transpose_colmajor_matrix<Real>(S, intra_dist2, n1, n2, Real(1), buffer);
    dense_left_multiply_colmajor_matrix<Real>(intra_dist1, buffer, n1, n1, n2, Real(-2) * lambda_e, buffer2);
    auto time_1 = std::chrono::high_resolution_clock::now();
    dense_right_multiply_transpose_colmajor<Real>(S, row_norm_adj2, n1, n2, Real(1), buffer);
    dense_left_transpose_multiply_colmajor<Real>(row_norm_adj1, buffer, n1, n2, Real(1), C);
    auto time_2 = std::chrono::high_resolution_clock::now();
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i;
            C[idx] = buffer2[idx] - lambda_n * std::log(C[idx]) + C_fixed[idx];
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_3 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in compute_cost_fused_2 first 2 dense mm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " second 2 dense mm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " matrix iter : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << " ms" << std::endl);
}

template void compute_cost_fused_2<double>(const std::vector<double>& row_norm_adj1, const std::vector<double>& row_norm_adj2, const std::vector<double>& S, int n1, int n2,
        const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C, std::vector<double>& buffer, std::vector<double>& buffer2);
template void compute_cost_fused_2<float>(const std::vector<float>& row_norm_adj1, const std::vector<float>& row_norm_adj2, const std::vector<float>& S, int n1, int n2,
        const std::vector<float>& intra_dist1, const std::vector<float>& intra_dist2, const std::vector<float>& C_fixed, float lambda_e, float lambda_n, std::vector<float>& C, std::vector<float>& buffer, std::vector<float>& buffer2);

template <typename Real>
Real update_S_mat (std::vector<Real>& S, const std::vector<Real>& S_old, const std::vector<Real>& Cost, const std::vector<Real>& a_row, const std::vector<Real>& b_col, int n1, int n2, Real epsilon) {
    unsigned int numThreads = mkl_get_max_threads();
    std::vector<Real> sum_diss_S (numThreads, Real(0));
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        Real a_val = a_row[j];
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t)j * n1 + i;
            S[idx] = Real(0.95) * std::exp((a_val + b_col[i] - Cost[idx]) / epsilon) / static_cast<Real>(n1 * n2)
                   + Real(0.05) * S_old[idx];
            // sum_diss_S[omp_get_thread_num()] += std::abs(S[idx] - S_old[idx]);
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);

    // return  std::accumulate(sum_diss_S.begin(), sum_diss_S.end(), 0); 
    return  Real(0);
}

template double update_S_mat<double>(std::vector<double>& S, const std::vector<double>& S_old, const std::vector<double>& Cost, const std::vector<double>& a_row, const std::vector<double>& b_col, int n1, int n2, double epsilon);
template float update_S_mat<float>(std::vector<float>& S, const std::vector<float>& S_old, const std::vector<float>& Cost, const std::vector<float>& a_row, const std::vector<float>& b_col, int n1, int n2, float epsilon);

template <typename Real>
std::vector<Real> con_prox_pt_opt(const std::vector<Real>& row_norm_adj1, const std::vector<Real>& row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    std::vector<Real>& cross_dist, const std::vector<Real>& intra_dist1, const std::vector<Real>& intra_dist2, int inner_iters, int outer_iters, Real lambda_n, Real lambda_a,
    Real lambda_p, Real lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs) {
    (void)avg_nnz1_per_n1;
    (void)avg_nnz2_per_n2;

    auto time_0 = std::chrono::high_resolution_clock::now();
    Real lambda_total = lambda_n + lambda_a + lambda_p;
    size_t N = (size_t)n1 * (size_t)n2;
    Real lambda_e = lambda_e_in * static_cast<Real>(N);
    std::vector<Real> S(N, Real(1) / static_cast<Real>(N));      // S = (1/(n1*n2)) * 1_n1 * 1_n2^T
    std::vector<Real> S_old(N);
    std::vector<Real> Cost_buff(N);
    std::vector<Real> Cost(N);
    std::vector<Real> temp(N);
    std::vector<Real> temp2(N);
    std::vector<Real> a_row(n2);
    std::vector<Real> b_col(n1, Real(1) / static_cast<Real>(n2));
    bool has_C_old = false;

    auto time_1 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "initialization in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() << " ms" << std::endl);

    // cross_dist = cross_dist + lambda_e * row-average intra terms - lambda_a * anchor log terms
    compute_Cost_fixed<Real>(intra_dist1, intra_dist2, cross_dist, n1, n2, lambda_e, lambda_a, anchor_pairs);

    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "L_fixed computation in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << " ms" << std::endl << std::endl);
    auto L_compute_time=0, C_compute_time=0, Cost_compute_time=0, compute_minaa=0, compute_minbb=0, update_S=0, inner_loop=0;
    
    // printf("before outer iteration mkl nthreads %d omp nthreads %d\n", mkl_get_max_threads(), omp_get_num_threads());
    for (int it = 0; it < outer_iters; ++it) {
        DEBUG_ONLY(printf("----- outer iterration %d ----\n", it));
        auto time_iter_0 = std::chrono::high_resolution_clock::now();
        cblas_copy_real<Real>(N, S.data(), S_old.data());

        
        auto time_L_done = std::chrono::high_resolution_clock::now();
        compute_cost_fused_2<Real>(row_norm_adj1, row_norm_adj2, S, n1, n2, intra_dist1, intra_dist2, cross_dist, lambda_e, lambda_n, Cost, temp, temp2);
        auto time_C_done = std::chrono::high_resolution_clock::now();
        C_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_C_done - time_L_done).count();

        // ---- C_old / W check ----
        // here Cost_buff is cost_old 
        if (!has_C_old) {
            cblas_copy_real<Real>(N, Cost.data(), Cost_buff.data());
            has_C_old = true;
        } 
        else {
            Real W_old = cblas_dot_real<Real>(N, S.data(), Cost_buff.data());
            Real W     = cblas_dot_real<Real>(N, S.data(), Cost.data());
            if (W <= W_old)     cblas_copy_real<Real>(N, Cost.data(), Cost_buff.data());
            else                cblas_copy_real<Real>(N, Cost_buff.data(), Cost.data());
        }
        // ---- Cost = Cost - lambda_p * log(S) ----
        vln_real<Real>(N, S.data(), S.data());
        cblas_axpy_real<Real>(N, -lambda_p, S.data(), Cost.data());

        auto time_iter_1 = std::chrono::high_resolution_clock::now();
        Cost_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1 - time_C_done).count()  ;
        // ---- Sinkhorn inner iterations ----
        mkl_omatcopy_real<Real>('C', 'T', n1, n2, Real(1), Cost.data(), n1, temp.data(), n2);     //here temp is transpose of cost matrix
        for (int j = 0; j < inner_iters; ++j) {
            auto t_0 = std::chrono::high_resolution_clock::now() ;
            // a_row = minaa_v3_dense(Cost, b_col, n1, n2, lambda_total);
            minaa_v3_dense<Real>(Cost, b_col, n1, n2, lambda_total, a_row);
            auto t_1 = std::chrono::high_resolution_clock::now() ;
            compute_minaa += std::chrono::duration_cast<std::chrono::milliseconds>(t_1 - t_0).count()  ;
            minaa_v3_dense<Real>(temp, a_row, n2, n1, lambda_total, b_col);
            auto t_2 = std::chrono::high_resolution_clock::now() ;
            compute_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_2 - t_1).count()  ;
        }
        auto time_iter_2 = std::chrono::high_resolution_clock::now();        
        inner_loop += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_2 - time_iter_1).count()  ;

        // update S and returns diff_S = sum |S - S_old|
        Real diff_S = update_S_mat<Real>(S, S_old, Cost, a_row, b_col, n1, n2, lambda_total);
        auto time_iter_3 = std::chrono::high_resolution_clock::now();        
        update_S += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_3 - time_iter_2).count()  ;
        
        //diff_S = sum |S - S_old|
        // using temp to store diff_S 
        // mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, S.data(), n1, -1.0, S_old.data(), n1, temp.data(), n1);
        // vdAbs(n1*n2, temp.data(), temp.data());
        // double diff_S = cblas_dasum(n1*n2, temp.data(), 1);
        DEBUG_ONLY(printf("inner iter %d and diff of S %.6f\n\n", it, diff_S));


        // printf("\n ----- end of iteration %d ----\n", it);
        // print_stats(S);
        // printf("\n ");

    }
    auto time_4 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "total time of 10 outer iterations : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_2).count() << " ms" << std::endl);
    DEBUG_ONLY(std::cout << " L_compute_time : " << L_compute_time << " C_compute_time : " << C_compute_time << " Cost_compute_time : " << Cost_compute_time <<
        " mina : " << compute_minaa << " minb : " << compute_minbb << " update_S : " << update_S << " ms" << std::endl);
    DEBUG_ONLY(std::cout << "done con_prox_pt_opt_init" << std::endl);
    return S;
}

template std::vector<double> con_prox_pt_opt<double>(const std::vector<double>& row_norm_adj1, const std::vector<double>& row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    std::vector<double>& cross_dist, const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, int inner_iters, int outer_iters, double lambda_n, double lambda_a,
    double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);
template std::vector<float> con_prox_pt_opt<float>(const std::vector<float>& row_norm_adj1, const std::vector<float>& row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    std::vector<float>& cross_dist, const std::vector<float>& intra_dist1, const std::vector<float>& intra_dist2, int inner_iters, int outer_iters, float lambda_n, float lambda_a,
    float lambda_p, float lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);

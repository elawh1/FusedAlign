#include "helper.h"
#include "helper2.h"
#include <ctime>

namespace {

template <typename Real>
SparseMatrixCSR<Real> transpose_csr(const SparseMatrixCSR<Real>& A) {
    SparseMatrixCSR<Real> T;
    T.rows = A.cols;
    T.cols = A.rows;
    T.row_ptr.assign((size_t)T.rows + 1, 0);
    T.col_ind.resize(A.col_ind.size());
    T.values.resize(A.values.size());

    for (MKL_INT col : A.col_ind) {
        if (col >= 0 && col < T.rows) ++T.row_ptr[(size_t)col + 1];
    }
    for (MKL_INT i = 0; i < T.rows; ++i) T.row_ptr[(size_t)i + 1] += T.row_ptr[(size_t)i];

    std::vector<MKL_INT> next = T.row_ptr;
    for (MKL_INT i = 0; i < A.rows; ++i) {
        for (MKL_INT p = A.row_ptr[(size_t)i]; p < A.row_ptr[(size_t)i + 1]; ++p) {
            const MKL_INT j = A.col_ind[(size_t)p];
            const MKL_INT dst = next[(size_t)j]++;
            T.col_ind[(size_t)dst] = i;
            T.values[(size_t)dst] = A.values[(size_t)p];
        }
    }
    return T;
}

template <typename Real>
void check_sparse_status(sparse_status_t status, const char* where) {
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + " failed with MKL sparse status " + std::to_string(status));
    }
}

template <typename Real>
sparse_matrix_t create_mkl_sparse_csr(const SparseMatrixCSR<Real>& A) {
    sparse_matrix_t handle = nullptr;
    if constexpr (std::is_same_v<Real, float>) {
        check_sparse_status<Real>(
            mkl_sparse_s_create_csr(&handle, SPARSE_INDEX_BASE_ZERO, A.rows, A.cols,
                                    const_cast<MKL_INT*>(A.row_ptr.data()),
                                    const_cast<MKL_INT*>(A.row_ptr.data() + 1),
                                    const_cast<MKL_INT*>(A.col_ind.data()),
                                    const_cast<float*>(A.values.data())),
            "mkl_sparse_s_create_csr");
    } else {
        check_sparse_status<Real>(
            mkl_sparse_d_create_csr(&handle, SPARSE_INDEX_BASE_ZERO, A.rows, A.cols,
                                    const_cast<MKL_INT*>(A.row_ptr.data()),
                                    const_cast<MKL_INT*>(A.row_ptr.data() + 1),
                                    const_cast<MKL_INT*>(A.col_ind.data()),
                                    const_cast<double*>(A.values.data())),
            "mkl_sparse_d_create_csr");
    }
    return handle;
}

template <typename Real>
struct MklSparseHandle {
    sparse_matrix_t handle = nullptr;

    explicit MklSparseHandle(const SparseMatrixCSR<Real>& A) : handle(create_mkl_sparse_csr<Real>(A)) {
        check_sparse_status<Real>(mkl_sparse_optimize(handle), "mkl_sparse_optimize");
    }

    ~MklSparseHandle() {
        if (handle != nullptr) mkl_sparse_destroy(handle);
    }

    MklSparseHandle(const MklSparseHandle&) = delete;
    MklSparseHandle& operator=(const MklSparseHandle&) = delete;
};

matrix_descr general_sparse_descr() {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    return descr;
}

template <typename Real>
void sparse_left_multiply_rowmajor(const MklSparseHandle<Real>& A, MKL_INT rows, const std::vector<Real>& B,
                                   MKL_INT cols, Real alpha, std::vector<Real>& C) {
    const size_t total = (size_t)rows * (size_t)cols;
    if (C.size() != total) C.resize(total);
    matrix_descr descr = general_sparse_descr();

    if constexpr (std::is_same_v<Real, float>) {
        check_sparse_status<Real>(
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, B.data(), cols, cols, Real(0), C.data(), cols),
            "mkl_sparse_s_mm row-major left");
    } else {
        check_sparse_status<Real>(
            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, B.data(), cols, cols, Real(0), C.data(), cols),
            "mkl_sparse_d_mm row-major left");
    }
}

template <typename Real>
void sparse_left_transpose_multiply_rowmajor(const MklSparseHandle<Real>& A, MKL_INT rows, const std::vector<Real>& B,
                                             MKL_INT cols, Real alpha, std::vector<Real>& C) {
    const size_t total = (size_t)rows * (size_t)cols;
    if (C.size() != total) C.resize(total);
    matrix_descr descr = general_sparse_descr();

    if constexpr (std::is_same_v<Real, float>) {
        check_sparse_status<Real>(
            mkl_sparse_s_mm(SPARSE_OPERATION_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, B.data(), cols, cols, Real(0), C.data(), cols),
            "mkl_sparse_s_mm row-major transpose left");
    } else {
        check_sparse_status<Real>(
            mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, B.data(), cols, cols, Real(0), C.data(), cols),
            "mkl_sparse_d_mm row-major transpose left");
    }
}

template <typename Real>
void sparse_left_multiply_colmajor(const MklSparseHandle<Real>& A, MKL_INT rows, MKL_INT inner,
                                   const std::vector<Real>& B, MKL_INT cols, Real alpha, std::vector<Real>& C) {
    const size_t total = (size_t)rows * (size_t)cols;
    if (C.size() != total) C.resize(total);
    matrix_descr descr = general_sparse_descr();

    if constexpr (std::is_same_v<Real, float>) {
        check_sparse_status<Real>(
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_COLUMN_MAJOR, B.data(), cols, inner, Real(0), C.data(), rows),
            "mkl_sparse_s_mm column-major left");
    } else {
        check_sparse_status<Real>(
            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_COLUMN_MAJOR, B.data(), cols, inner, Real(0), C.data(), rows),
            "mkl_sparse_d_mm column-major left");
    }
}

template <typename Real>
void sparse_right_multiply_transpose_colmajor(const std::vector<Real>& B, const MklSparseHandle<Real>& A,
                                             MKL_INT rows, MKL_INT out_cols, Real alpha, std::vector<Real>& C) {
    const size_t total = (size_t)rows * (size_t)out_cols;
    if (C.size() != total) C.resize(total);
    matrix_descr descr = general_sparse_descr();

    // C = B @ A.T. In memory, C column-major is C.T row-major, so compute C.T = A @ B.T.
    if constexpr (std::is_same_v<Real, float>) {
        check_sparse_status<Real>(
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, B.data(), rows, rows, Real(0), C.data(), rows),
            "mkl_sparse_s_mm right transpose");
    } else {
        check_sparse_status<Real>(
            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, B.data(), rows, rows, Real(0), C.data(), rows),
            "mkl_sparse_d_mm right transpose");
    }
}

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
SparseMatrixCSR<Real> make_csr_from_edgelist(const std::vector<Real>& edgelist, int m, int n) {
    std::vector<std::vector<MKL_INT>> rows((size_t)m);
    for (size_t e = 0; e + 1 < edgelist.size(); e += 2) {
        MKL_INT row = static_cast<MKL_INT>(edgelist[e]);
        MKL_INT col = static_cast<MKL_INT>(edgelist[e + 1]);
        if (row >= 0 && row < m && col >= 0 && col < n) rows[(size_t)row].push_back(col);
    }

    SparseMatrixCSR<Real> A;
    A.rows = m;
    A.cols = n;
    A.row_ptr.assign((size_t)m + 1, 0);
    for (int i = 0; i < m; ++i) {
        auto& r = rows[(size_t)i];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        A.row_ptr[(size_t)i + 1] = A.row_ptr[(size_t)i] + static_cast<MKL_INT>(r.size());
    }
    A.col_ind.resize((size_t)A.row_ptr[(size_t)m]);
    A.values.assign(A.col_ind.size(), Real(1));
    for (int i = 0; i < m; ++i) {
        std::copy(rows[(size_t)i].begin(), rows[(size_t)i].end(), A.col_ind.begin() + A.row_ptr[(size_t)i]);
    }
    return A;
}

template SparseMatrixCSR<double> make_csr_from_edgelist<double>(const std::vector<double>& edgelist, int m, int n);
template SparseMatrixCSR<float> make_csr_from_edgelist<float>(const std::vector<float>& edgelist, int m, int n);

template <typename Real>
void masked_dense_intra_cost(std::vector<Real>& emb, const SparseMatrixCSR<Real>& mask, int n, int L, std::vector<Real>& result) {
    if (mask.rows != n || mask.cols != n) throw std::runtime_error("mask dimensions do not match n");
    if (emb.size() != (size_t)n * (size_t)L) throw std::runtime_error("emb size mismatch");
    std::vector<Real> dense_dist((size_t)n * (size_t)n, Real(0));

    if constexpr (std::is_same_v<Real, float>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    (MKL_INT)n, (MKL_INT)n, (MKL_INT)L,
                    -1.0f,
                    emb.data(), (MKL_INT)n,
                    emb.data(), (MKL_INT)n,
                    0.0f,
                    dense_dist.data(), (MKL_INT)n);
        vsExp((MKL_INT)dense_dist.size(), dense_dist.data(), dense_dist.data());
    } else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    (MKL_INT)n, (MKL_INT)n, (MKL_INT)L,
                    -1.0,
                    emb.data(), (MKL_INT)n,
                    emb.data(), (MKL_INT)n,
                    0.0,
                    dense_dist.data(), (MKL_INT)n);
        vdExp((MKL_INT)dense_dist.size(), dense_dist.data(), dense_dist.data());
    }

    result.assign((size_t)n * (size_t)n, Real(0));
    const unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < n; ++i) {
        for (MKL_INT p = mask.row_ptr[(size_t)i]; p < mask.row_ptr[(size_t)i + 1]; ++p) {
            MKL_INT j = mask.col_ind[(size_t)p];
            result[(size_t)j * (size_t)n + (size_t)i] = dense_dist[(size_t)j * (size_t)n + (size_t)i];
        }
    }
    omp_set_num_threads(1);
}

template void masked_dense_intra_cost<double>(std::vector<double>& emb, const SparseMatrixCSR<double>& mask, int n, int L, std::vector<double>& result);
template void masked_dense_intra_cost<float>(std::vector<float>& emb, const SparseMatrixCSR<float>& mask, int n, int L, std::vector<float>& result);

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
SparseMatrixCSR<Real> row_normalize_graph(const SparseMatrixCSR<Real>& graph, int n, MKL_INT avg_nnz_per_n) {
    (void)avg_nnz_per_n;
    if (graph.rows != n || graph.cols != n) throw std::runtime_error("graph size mismatch in row_normalize_graph");
    SparseMatrixCSR<Real> graph_connected;
    graph_connected.rows = n;
    graph_connected.cols = n;
    graph_connected.row_ptr.assign((size_t)n + 1, 0);

    for (MKL_INT i = 0; i < n; ++i) {
        MKL_INT nnz = graph.row_ptr[(size_t)i + 1] - graph.row_ptr[(size_t)i];
        if (nnz == 0) nnz = n;
        graph_connected.row_ptr[(size_t)i + 1] = graph_connected.row_ptr[(size_t)i] + nnz;
    }
    graph_connected.col_ind.resize((size_t)graph_connected.row_ptr[(size_t)n]);
    graph_connected.values.resize(graph_connected.col_ind.size());

    const unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < n; ++i) {
        const MKL_INT in_start = graph.row_ptr[(size_t)i];
        const MKL_INT in_end = graph.row_ptr[(size_t)i + 1];
        const MKL_INT out_start = graph_connected.row_ptr[(size_t)i];
        const MKL_INT nnz = in_end - in_start;
        if (nnz == 0) {
            const Real val = Real(1) / static_cast<Real>(n);
            for (MKL_INT j = 0; j < n; ++j) {
                graph_connected.col_ind[(size_t)out_start + (size_t)j] = j;
                graph_connected.values[(size_t)out_start + (size_t)j] = val;
            }
        } else {
            const Real val = Real(1) / static_cast<Real>(nnz);
            for (MKL_INT p = 0; p < nnz; ++p) {
                graph_connected.col_ind[(size_t)out_start + (size_t)p] = graph.col_ind[(size_t)in_start + (size_t)p];
                graph_connected.values[(size_t)out_start + (size_t)p] = val;
            }
        }
    }
    omp_set_num_threads(1);
    return graph_connected;
}

template SparseMatrixCSR<double> row_normalize_graph<double>(const SparseMatrixCSR<double>& graph, int n, MKL_INT avg_nnz_per_n);
template SparseMatrixCSR<float> row_normalize_graph<float>(const SparseMatrixCSR<float>& graph, int n, MKL_INT avg_nnz_per_n);

void get_batch_rwr_scores(const std::vector<double>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) {
    (void)graph_norm_csr;
    (void)anchors;
    (void)n;
    (void)avg_nnz_per_n;
    (void)directed;
    (void)restart_prob;
    (void)max_iters;
    (void)connect_isolated;
    (void)Xnext_vals;
    throw std::runtime_error("dense graph RWR overload is disabled in fastalign-cpp-sparse; use the CSR overload");
}

template <typename Real>
void get_batch_rwr_scores_rowmjr(const SparseMatrixCSR<Real>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
        bool directed, Real restart_prob, int max_iters, bool connect_isolated, std::vector<Real>& Xnext_vals) {
    (void)avg_nnz_per_n;
    (void)directed;
    (void)connect_isolated;
    auto time_0 = std::chrono::high_resolution_clock::now();
    int l = anchors.size();
    const size_t N = (size_t)n * (size_t)l;
    std::vector<Real> X_vals(N, Real(1));
    std::vector<Real> X_work(N, Real(0));
    MklSparseHandle<Real> graph_handle(graph_norm_csr);
    unsigned int numThreads = mkl_get_max_threads();
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto add_after_mm_time = 0, mm_time = 0, convergence_time = 0, swap_after_mm_time = 0;
    const Real walk_scale = Real(1) - restart_prob;
    constexpr Real tol = Real(1e-6);
    for (int iter = 0; iter < max_iters; ++iter) {
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        sparse_left_transpose_multiply_rowmajor<Real>(graph_handle, n, X_vals, l, walk_scale, X_work);
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        omp_set_num_threads(numThreads);
        #pragma omp parallel for
        for (int j = 0; j < l; ++j) {
            X_work[(size_t)anchors[j] * (size_t)l + (size_t)j] += restart_prob;
        }
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        add_after_mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        Real max_diff = Real(0);
        #pragma omp parallel for reduction(max:max_diff)
        for (size_t idx = 0; idx < N; ++idx) {
            max_diff = std::max(max_diff, static_cast<Real>(std::abs(X_work[idx] - X_vals[idx])));
        }
        auto time_1_2b = std::chrono::high_resolution_clock::now();
        convergence_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2b - time_1_2).count();
        std::swap(X_vals, X_work);
        omp_set_num_threads(1);
        auto time_1_3 = std::chrono::high_resolution_clock::now();
        swap_after_mm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2b).count();
        if (max_diff < tol) {
            break;
        }
    }

    auto time_2_0 = std::chrono::high_resolution_clock::now();
    mkl_omatcopy_real<Real>('R', 'T', n, l, Real(1), X_vals.data(), l, Xnext_vals.data(), n);

    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_batch_rwr_scores_rowmjr generate E " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop + copy " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " mkl_omatcopy_real time " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_2_0).count() <<
        " sparse mm time " << mm_time << " add after sparse mm time " << add_after_mm_time <<
        " convergence time " << convergence_time << " swap after sparse mm time " << swap_after_mm_time << std::endl);
}

template void get_batch_rwr_scores_rowmjr<double>(const SparseMatrixCSR<double>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
        bool directed, double restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals);
template void get_batch_rwr_scores_rowmjr<float>(const SparseMatrixCSR<float>& graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n,
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
void get_product_rwr_mat_new(const SparseMatrixCSR<Real>& graph1_normalized, const SparseMatrixCSR<Real>& graph2_normalized,
    const std::vector<Real>& in_cross_dist, MKL_INT n1, MKL_INT n2, Real rwr_restart_prob, Real gamma,
    std::vector<Real>& out_cross_dist, int max_iters, Real tol, ProductRwrTiming* timing) {
    size_t N = (size_t)n1 * (size_t)n2;
    if (out_cross_dist.size() != N || in_cross_dist.size() != N)    throw std::runtime_error("in cross dist computation size mismatch");
    // const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
    // size_t off = 0;
    auto time_0 = std::chrono::high_resolution_clock::now();
    std::vector<Real> out_cross_dist_old(N, Real(0));
    std::vector<Real> temp(N);
    Real coeff = (Real(1) - rwr_restart_prob) * gamma;
    Real alpha = Real(1) + rwr_restart_prob;
    auto time_handles_0 = std::chrono::high_resolution_clock::now();
    MklSparseHandle<Real> graph1_handle(graph1_normalized);
    MklSparseHandle<Real> graph2_handle(graph2_normalized);
    auto time_handles_1 = std::chrono::high_resolution_clock::now();
    unsigned int numThreads = mkl_get_max_threads();
    auto time_1 = std::chrono::high_resolution_clock::now();
    long long time_mm1 = 0, time_mm2 = 0, time_convergence = 0;
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
    int iterations = 1;
    if (converged || max_iters <= 1) {
        cblas_copy_real<Real>(N, out_cross_dist_old.data(), out_cross_dist.data());
    }
    for (int i = 1; !converged && i < max_iters; ++i) {
        iterations = i + 1;
        swapped_last = false;
        // graph1_normalized and graph2_normalized are row-major. Cross-distance buffers are column-major.
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        sparse_right_multiply_transpose_colmajor<Real>(out_cross_dist_old, graph2_handle, n1, n2, Real(1), temp);
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        time_mm1 += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();

        sparse_left_multiply_colmajor<Real>(graph1_handle, n1, n1, temp, n2, coeff, out_cross_dist);
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
        omp_set_num_threads(1);
        mkl_set_num_threads(numThreads);
        auto time_1_4 = std::chrono::high_resolution_clock::now();
        time_convergence += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_4 - time_1_2).count();
        if (converged) break;
        std::swap(out_cross_dist, out_cross_dist_old);
        swapped_last = true;
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
    auto time_2 = std::chrono::high_resolution_clock::now();
    if (swapped_last) std::swap(out_cross_dist, out_cross_dist_old);
    dscal_chunked<Real>(N, Real(1) - gamma, out_cross_dist.data());
    auto time_3 = std::chrono::high_resolution_clock::now();
    if (timing != nullptr) {
        timing->init_ms += std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count();
        timing->right_multiply_ms += time_mm1;
        timing->left_multiply_ms += time_mm2;
        timing->add_convergence_ms += time_convergence;
        timing->final_scale_ms += std::chrono::duration_cast<std::chrono::milliseconds>(time_3 - time_2).count();
        timing->iterations = iterations;
    }
    std::cout << "cross distance propagation breakdown" << std::endl;
    std::cout << "  iterations: " << iterations << std::endl;
    std::cout << "  init/allocation: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_handles_0 - time_0).count() << " ms" << std::endl;
    std::cout << "  sparse handle create/optimize: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_handles_1 - time_handles_0).count() << " ms" << std::endl;
    std::cout << "  first iteration setup: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_handles_1).count() << " ms" << std::endl;
    std::cout << "  sparse right multiply out @ graph2.T: " << time_mm1 << " ms" << std::endl;
    std::cout << "  sparse left multiply graph1 @ temp: " << time_mm2 << " ms" << std::endl;
    std::cout << "  add input + convergence check: " << time_convergence << " ms" << std::endl;
    std::cout << "  final scale: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3 - time_2).count() << " ms" << std::endl;
    DEBUG_ONLY(std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << std::endl);
    DEBUG_ONLY(std::cout << "in for loop 1st sparse mm " << time_mm1 << " 2nd sparse mm " << time_mm2 << " convergence " << time_convergence << std::endl);
}

template void get_product_rwr_mat_new<double>(const SparseMatrixCSR<double>& graph1_normalized, const SparseMatrixCSR<double>& graph2_normalized,
    const std::vector<double>& in_cross_dist, MKL_INT n1, MKL_INT n2, double rwr_restart_prob, double gamma,
    std::vector<double>& out_cross_dist, int max_iters, double tol, ProductRwrTiming* timing);
template void get_product_rwr_mat_new<float>(const SparseMatrixCSR<float>& graph1_normalized, const SparseMatrixCSR<float>& graph2_normalized,
    const std::vector<float>& in_cross_dist, MKL_INT n1, MKL_INT n2, float rwr_restart_prob, float gamma,
    std::vector<float>& out_cross_dist, int max_iters, float tol, ProductRwrTiming* timing);

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
    const Real h_non_anchor_penalty = -lambda_a * static_cast<Real>(std::log(1e-6));
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        Real dist2_term =  row_avg_square_dist2[j];
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i ;
            cross_dist[idx] = cross_dist[idx] + lambda_e * (dist2_term + row_avg_square_dist1[i]) + h_non_anchor_penalty;
        }
    }
    const MKL_INT k = static_cast<MKL_INT>(anchor_pairs.first.size());
    #pragma omp parallel for schedule(static)
    for (MKL_INT t = 0; t < k; ++t) {
        MKL_INT i = anchor_pairs.first[t];
        MKL_INT j = anchor_pairs.second[t];
        if (i >= 0 && i < n1 && j >= 0 && j < n2) {
            cross_dist[(size_t)j * (size_t)n1 + (size_t)i] -= h_non_anchor_penalty;
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "optimization fixed-cost breakdown" << std::endl;
    std::cout << "  row-average squared intra terms: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count() << " ms" << std::endl;
    std::cout << "  add fixed edge/anchor terms: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count() << " ms" << std::endl;
    DEBUG_ONLY(std::cout << "in compute_Cost_fixed before loop  : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " after loop  : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << std::endl);
}

template void compute_Cost_fixed<double>(const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, std::vector<double>& cross_dist,
        int n1, int n2, double lambda_e, double lambda_a, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);
template void compute_Cost_fixed<float>(const std::vector<float>& intra_dist1, const std::vector<float>& intra_dist2, std::vector<float>& cross_dist,
        int n1, int n2, float lambda_e, float lambda_a, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);

template <typename Real>
void compute_cost_fused_2(const SparseMatrixCSR<Real>& row_norm_adj1, const SparseMatrixCSR<Real>& row_norm_adj2, const std::vector<Real>& S, int n1, int n2,
        const std::vector<Real>& intra_dist1, const std::vector<Real>& intra_dist2, const std::vector<Real>& C_fixed, Real lambda_e, Real lambda_n, std::vector<Real>& C,
        std::vector<Real>& buffer, std::vector<Real>& buffer2, CostFusedTiming* timing) {
    size_t matrix_size = (size_t)n1 * n2;
    if (buffer.size() != matrix_size) buffer.resize(matrix_size);
    if (buffer2.size() != matrix_size) buffer2.resize(matrix_size);

    auto time_0 = std::chrono::high_resolution_clock::now();
    dense_right_multiply_transpose_colmajor_matrix<Real>(S, intra_dist2, n1, n2, Real(1), buffer);
    dense_left_multiply_colmajor_matrix<Real>(intra_dist1, buffer, n1, n1, n2, Real(-2) * lambda_e, buffer2);
    auto time_1 = std::chrono::high_resolution_clock::now();
    SparseMatrixCSR<Real> row_norm_adj2_t = transpose_csr(row_norm_adj2);
    SparseMatrixCSR<Real> row_norm_adj1_t = transpose_csr(row_norm_adj1);
    MklSparseHandle<Real> row_norm_adj2_t_handle(row_norm_adj2_t);
    MklSparseHandle<Real> row_norm_adj1_t_handle(row_norm_adj1_t);
    sparse_right_multiply_transpose_colmajor<Real>(S, row_norm_adj2_t_handle, n1, n2, Real(1), buffer);
    sparse_left_multiply_colmajor<Real>(row_norm_adj1_t_handle, n1, n1, buffer, n2, Real(1), C);
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
    if (timing != nullptr) {
        timing->edge_cost_mm_ms += std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count();
        timing->topology_mm_ms += std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count();
        timing->elementwise_ms += std::chrono::duration_cast<std::chrono::milliseconds>(time_3 - time_2).count();
    }
    DEBUG_ONLY(std::cout << "in compute_cost_fused_2 first 2 dense mm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " graph sparse mm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " matrix iter : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << " ms" << std::endl);
}

template void compute_cost_fused_2<double>(const SparseMatrixCSR<double>& row_norm_adj1, const SparseMatrixCSR<double>& row_norm_adj2, const std::vector<double>& S, int n1, int n2,
        const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C, std::vector<double>& buffer, std::vector<double>& buffer2, CostFusedTiming* timing);
template void compute_cost_fused_2<float>(const SparseMatrixCSR<float>& row_norm_adj1, const SparseMatrixCSR<float>& row_norm_adj2, const std::vector<float>& S, int n1, int n2,
        const std::vector<float>& intra_dist1, const std::vector<float>& intra_dist2, const std::vector<float>& C_fixed, float lambda_e, float lambda_n, std::vector<float>& C, std::vector<float>& buffer, std::vector<float>& buffer2, CostFusedTiming* timing);

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
std::vector<Real> con_prox_pt_opt(const SparseMatrixCSR<Real>& row_norm_adj1, const SparseMatrixCSR<Real>& row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2,
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
    long long copy_S_time = 0;
    long long C_compute_time = 0;
    long long W_check_time = 0;
    long long proximal_log_time = 0;
    long long sinkhorn_transpose_time = 0;
    long long compute_minaa = 0;
    long long compute_minbb = 0;
    long long update_S = 0;
    long long inner_loop = 0;
    CostFusedTiming fused_timing;

    // printf("before outer iteration mkl nthreads %d omp nthreads %d\n", mkl_get_max_threads(), omp_get_num_threads());
    for (int it = 0; it < outer_iters; ++it) {
        DEBUG_ONLY(printf("----- outer iterration %d ----\n", it));
        auto time_iter_0 = std::chrono::high_resolution_clock::now();
        cblas_copy_real<Real>(N, S.data(), S_old.data());
        auto time_copy_done = std::chrono::high_resolution_clock::now();
        copy_S_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_copy_done - time_iter_0).count();


        auto time_L_done = std::chrono::high_resolution_clock::now();
        compute_cost_fused_2<Real>(row_norm_adj1, row_norm_adj2, S, n1, n2, intra_dist1, intra_dist2, cross_dist, lambda_e, lambda_n, Cost, temp, temp2, &fused_timing);
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
        auto time_W_done = std::chrono::high_resolution_clock::now();
        W_check_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_W_done - time_C_done).count();
        // ---- Cost = Cost - lambda_p * log(S) ----
        vln_real<Real>(N, S.data(), S.data());
        cblas_axpy_real<Real>(N, -lambda_p, S.data(), Cost.data());

        auto time_iter_1 = std::chrono::high_resolution_clock::now();
        proximal_log_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1 - time_W_done).count()  ;
        // ---- Sinkhorn inner iterations ----
        mkl_omatcopy_real<Real>('C', 'T', n1, n2, Real(1), Cost.data(), n1, temp.data(), n2);     //here temp is transpose of cost matrix
        auto time_transpose_done = std::chrono::high_resolution_clock::now();
        sinkhorn_transpose_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_transpose_done - time_iter_1).count();
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
        inner_loop += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_2 - time_transpose_done).count()  ;

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
    std::cout << "optimization breakdown" << std::endl;
    std::cout << "  initialization/allocation: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count() << " ms" << std::endl;
    std::cout << "  fixed cost setup: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count() << " ms" << std::endl;
    std::cout << "  outer iterations: " << outer_iters << std::endl;
    std::cout << "  copy S_old: " << copy_S_time << " ms" << std::endl;
    std::cout << "  C update total: " << C_compute_time << " ms" << std::endl;
    std::cout << "    edge dense multiplies intra1 @ S @ intra2.T: " << fused_timing.edge_cost_mm_ms << " ms" << std::endl;
    std::cout << "    topology sparse multiplies adj1.T @ S @ adj2: " << fused_timing.topology_mm_ms << " ms" << std::endl;
    std::cout << "    elementwise C combine/log: " << fused_timing.elementwise_ms << " ms" << std::endl;
    std::cout << "  C_old/W check: " << W_check_time << " ms" << std::endl;
    std::cout << "  proximal log(S): " << proximal_log_time << " ms" << std::endl;
    std::cout << "  Sinkhorn transpose: " << sinkhorn_transpose_time << " ms" << std::endl;
    std::cout << "  Sinkhorn minaa: " << compute_minaa << " ms" << std::endl;
    std::cout << "  Sinkhorn minbb: " << compute_minbb << " ms" << std::endl;
    std::cout << "  Sinkhorn inner loop total: " << inner_loop << " ms" << std::endl;
    std::cout << "  update S: " << update_S << " ms" << std::endl;
    std::cout << "  outer loop wall time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4 - time_2).count() << " ms" << std::endl;
    DEBUG_ONLY(std::cout << "total time of 10 outer iterations : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_2).count() << " ms" << std::endl);
    DEBUG_ONLY(std::cout << " C_compute_time : " << C_compute_time << " proximal_log_time : " << proximal_log_time <<
        " mina : " << compute_minaa << " minb : " << compute_minbb << " update_S : " << update_S << " ms" << std::endl);
    DEBUG_ONLY(std::cout << "done con_prox_pt_opt_init" << std::endl);
    return S;
}

template std::vector<double> con_prox_pt_opt<double>(const SparseMatrixCSR<double>& row_norm_adj1, const SparseMatrixCSR<double>& row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    std::vector<double>& cross_dist, const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, int inner_iters, int outer_iters, double lambda_n, double lambda_a,
    double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);
template std::vector<float> con_prox_pt_opt<float>(const SparseMatrixCSR<float>& row_norm_adj1, const SparseMatrixCSR<float>& row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    std::vector<float>& cross_dist, const std::vector<float>& intra_dist1, const std::vector<float>& intra_dist2, int inner_iters, int outer_iters, float lambda_n, float lambda_a,
    float lambda_p, float lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs);

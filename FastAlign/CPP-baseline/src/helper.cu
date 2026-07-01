#include "cuda_helper.cuh"

namespace {

CudaDenseMatrix make_dense_from_host(int rows, int cols, const std::vector<element_type>& vals) {
    CudaDenseMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    CHECK_CUDA(CUDA_MALLOC(&matrix.d_vals, vals.size() * sizeof(element_type)));
    CHECK_CUDA(cudaMemcpy(matrix.d_vals, vals.data(), vals.size() * sizeof(element_type), cudaMemcpyHostToDevice));
    return matrix;
}

}  // namespace

void destroy_cuda_dense(CudaDenseMatrix& matrix) {
    cuda_free_and_null(matrix.d_vals);
    matrix.rows = 0;
    matrix.cols = 0;
}

CudaDenseMatrix make_cuda_dense_from_edgelist(const std::vector<element_type>& edgelist, int m, int n) {
    const int edge_count = static_cast<int>(edgelist.size() / 2);
    std::vector<element_type> vals(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<element_type>(0));
    for (int e = 0; e < edge_count; ++e) {
        const int row = static_cast<int>(edgelist[2 * e]);
        const int col = static_cast<int>(edgelist[2 * e + 1]);
        if (row < 0 || row >= m || col < 0 || col >= n) {
            throw std::runtime_error("edge index out of bounds in make_cuda_dense_from_edgelist");
        }
        vals[static_cast<size_t>(col) * static_cast<size_t>(m) + static_cast<size_t>(row)] = static_cast<element_type>(1);
    }
    return make_dense_from_host(m, n, vals);
}

CudaDenseMatrix transpose_cuda_dense(const CudaDenseMatrix& matrix) {
    CudaDenseMatrix result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    const size_t total = static_cast<size_t>(result.rows) * static_cast<size_t>(result.cols);
    CHECK_CUDA(CUDA_MALLOC(&result.d_vals, total * sizeof(element_type)));
    const element_type one = static_cast<element_type>(1);
    const element_type zero = static_cast<element_type>(0);
    CHECK_CUBLAS(cuda_cublas_geam(0, CUBLAS_OP_T, CUBLAS_OP_N,
                                  result.rows, result.cols,
                                  &one, matrix.d_vals, matrix.rows,
                                  &zero, matrix.d_vals, result.rows,
                                  result.d_vals, result.rows));
    return result;
}

CudaDenseMatrix row_normalize_graph_cuda(const CudaDenseMatrix& graph, int avg_edges_per_node) {
    (void)avg_edges_per_node;
    const int rows = graph.rows;
    const int cols = graph.cols;
    std::vector<element_type> vals(graph.d_vals, graph.d_vals + static_cast<size_t>(rows) * static_cast<size_t>(cols));

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        element_type sum = static_cast<element_type>(0);
        for (int col = 0; col < cols; ++col) {
            sum += vals[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)];
        }

        if (sum <= static_cast<element_type>(1e-10)) {
            for (int col = 0; col < cols; ++col) {
                vals[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] = static_cast<element_type>(1);
            }
            sum = static_cast<element_type>(cols);
        }

        const element_type inv_sum = static_cast<element_type>(1) / sum;
        for (int col = 0; col < cols; ++col) {
            vals[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] *= inv_sum;
        }
    }

    return make_dense_from_host(rows, cols, vals);
}

void l2_norm_inplace_device(element_type* matrix, size_t n, size_t d) {
    if (n == 0 || d == 0) return;
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < n; ++row) {
        element_type square_sum = static_cast<element_type>(0);
        for (size_t col = 0; col < d; ++col) {
            const element_type v = matrix[col * n + row];
            square_sum += v * v;
        }
        if (square_sum > static_cast<element_type>(0)) {
            const element_type inv_norm = static_cast<element_type>(1) / std::sqrt(square_sum);
            for (size_t col = 0; col < d; ++col) {
                matrix[col * n + row] *= inv_norm;
            }
        }
    }
}

void get_normalized_neg_exp_dist_device(const element_type* rwr1,
                                        const element_type* rwr2,
                                        int n1,
                                        int n2,
                                        int l,
                                        element_type* cross_rwr_dist) {
    const element_type alpha = static_cast<element_type>(-1);
    const element_type beta = static_cast<element_type>(0);
    CHECK_CUBLAS(cuda_cublas_gemm(0, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n1, n2, l,
                                  &alpha, rwr1, n1,
                                  rwr2, n2,
                                  &beta, cross_rwr_dist, n1));
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total; ++idx) {
        cross_rwr_dist[idx] = std::exp(cross_rwr_dist[idx]);
    }
}

void anchor_cost_zero_device(element_type* matrix,
                             int nrows,
                             int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             element_type val) {
    for (size_t t = 0; t < anchors.first.size(); ++t) {
        const int row = anchors.first[t];
        const int col = anchors.second[t];
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
            std::fprintf(stderr, "anchor_cost_zero: out-of-range anchor (%d,%d)\n", row, col);
            continue;
        }
        matrix[static_cast<size_t>(col) * static_cast<size_t>(nrows) + static_cast<size_t>(row)] = val;
    }
}

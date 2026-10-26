#include "cuda_helper.cuh"

namespace {

void validate_anchors(int nrows, int ncols,
                      const std::pair<std::vector<int>, std::vector<int>>& anchors) {
    if (anchors.first.size() != anchors.second.size()) {
        throw std::invalid_argument("anchor size mismatch");
    }
    for (size_t t = 0; t < anchors.first.size(); ++t) {
        const int row = anchors.first[t];
        const int col = anchors.second[t];
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
            throw std::out_of_range("anchor out of range");
        }
    }
}

CudaDenseMatrix sparse_to_dense_col_major(const CudaSparseMatrix& sparse) {
    CudaDenseMatrix dense;
    dense.rows = sparse.rows;
    dense.cols = sparse.cols;
    const size_t total = static_cast<size_t>(dense.rows) * static_cast<size_t>(dense.cols);
    CHECK_CUDA(CUDA_MALLOC(&dense.d_vals, total * sizeof(element_type)));
    std::fill(dense.d_vals, dense.d_vals + total, static_cast<element_type>(0));

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < sparse.rows; ++row) {
        for (int idx = sparse.d_row_offsets[row]; idx < sparse.d_row_offsets[row + 1]; ++idx) {
            const int col = sparse.d_col_indices[idx];
            dense.d_vals[static_cast<size_t>(col) * static_cast<size_t>(sparse.rows) + static_cast<size_t>(row)] =
                sparse.d_vals[idx];
        }
    }
    return dense;
}

void row_avg_square_dense(const element_type* matrix, int n, std::vector<element_type>& out) {
    out.assign(static_cast<size_t>(n), static_cast<element_type>(0));
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        element_type sum = static_cast<element_type>(0);
        for (int col = 0; col < n; ++col) {
            const element_type v = matrix[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(row)];
            sum += v * v;
        }
        out[static_cast<size_t>(row)] = sum / static_cast<element_type>(n);
    }
}

std::vector<element_type> init_h_log(int n1,
                                     int n2,
                                     const std::pair<std::vector<int>, std::vector<int>>& anchors) {
    std::vector<element_type> h_log(static_cast<size_t>(n1) * static_cast<size_t>(n2), static_cast<element_type>(0));
    const element_type anchor_log = static_cast<element_type>(std::log(1e-6));
    for (size_t t = 0; t < anchors.first.size(); ++t) {
        h_log[static_cast<size_t>(anchors.second[t]) * static_cast<size_t>(n1) + static_cast<size_t>(anchors.first[t])] = anchor_log;
    }
    return h_log;
}

void compute_cost_fixed(const CudaDenseMatrix& intra1,
                        const CudaDenseMatrix& intra2,
                        const std::vector<element_type>& h_log,
                        int n1,
                        int n2,
                        element_type lambda_e,
                        element_type lambda_a,
                        std::vector<element_type>& c_fixed) {
    std::vector<element_type> avg1, avg2;
    row_avg_square_dense(intra1.d_vals, n1, avg1);
    row_avg_square_dense(intra2.d_vals, n2, avg2);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            const size_t idx = static_cast<size_t>(j) * static_cast<size_t>(n1) + static_cast<size_t>(i);
            c_fixed[idx] += lambda_e * (avg1[static_cast<size_t>(i)] + avg2[static_cast<size_t>(j)]) - lambda_a * h_log[idx];
        }
    }
}

void minaa_dense(const std::vector<element_type>& cost,
                 const std::vector<element_type>& b,
                 int n_row,
                 int n_col,
                 element_type epsilon,
                 std::vector<element_type>& out) {
    out.assign(static_cast<size_t>(n_col), static_cast<element_type>(0));
    #pragma omp parallel for schedule(dynamic, 4)
    for (int col = 0; col < n_col; ++col) {
        const element_type* col_ptr = cost.data() + static_cast<size_t>(col) * static_cast<size_t>(n_row);
        element_type min_val = std::numeric_limits<element_type>::infinity();
        for (int row = 0; row < n_row; ++row) {
            min_val = std::min(min_val, col_ptr[row] - b[static_cast<size_t>(row)]);
        }
        element_type sum = static_cast<element_type>(0);
        for (int row = 0; row < n_row; ++row) {
            sum += static_cast<element_type>(std::exp(static_cast<double>(-((col_ptr[row] - b[static_cast<size_t>(row)] - min_val) / epsilon))));
        }
        out[static_cast<size_t>(col)] = min_val - epsilon * static_cast<element_type>(std::log(static_cast<double>(sum / static_cast<element_type>(n_row))));
    }
}

void transpose_col_major(const std::vector<element_type>& in,
                         int rows,
                         int cols,
                         std::vector<element_type>& out) {
    out.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), static_cast<element_type>(0));
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            out[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)] =
                in[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)];
        }
    }
}

void compute_cost_dense(const CudaDenseMatrix& row_norm_adj1_t,
                        const CudaDenseMatrix& row_norm_adj2,
                        const CudaDenseMatrix& intra1,
                        const CudaDenseMatrix& intra2,
                        const std::vector<element_type>& c_fixed,
                        const std::vector<element_type>& s,
                        int n1,
                        int n2,
                        element_type lambda_e,
                        element_type lambda_n,
                        std::vector<element_type>& cost,
                        std::vector<element_type>& buffer,
                        std::vector<element_type>& buffer2) {
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    cost.assign(total, static_cast<element_type>(0));
    buffer.assign(total, static_cast<element_type>(0));
    buffer2.assign(total, static_cast<element_type>(0));

    const element_type one = static_cast<element_type>(1);
    const element_type zero = static_cast<element_type>(0);
    const element_type neg_two_lambda_e = static_cast<element_type>(-2) * lambda_e;

    CHECK_CUBLAS(cuda_cublas_gemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n1, n2, n2,
                                  &one, s.data(), n1,
                                  intra2.d_vals, intra2.rows,
                                  &zero, cost.data(), n1));
    CHECK_CUBLAS(cuda_cublas_gemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n1, n2, n1,
                                  &neg_two_lambda_e, intra1.d_vals, intra1.rows,
                                  cost.data(), n1,
                                  &zero, buffer2.data(), n1));

    CHECK_CUBLAS(cuda_cublas_gemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n1, n2, n2,
                                  &one, s.data(), n1,
                                  row_norm_adj2.d_vals, row_norm_adj2.rows,
                                  &zero, cost.data(), n1));
    CHECK_CUBLAS(cuda_cublas_gemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n1, n2, n1,
                                  &one, row_norm_adj1_t.d_vals, row_norm_adj1_t.rows,
                                  cost.data(), n1,
                                  &zero, buffer.data(), n1));

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total; ++idx) {
        const element_type rw = std::max(buffer[idx], static_cast<element_type>(1e-15));
        cost[idx] = c_fixed[idx] + buffer2[idx] - lambda_n * static_cast<element_type>(std::log(static_cast<double>(rw)));
    }
}

element_type update_s(std::vector<element_type>& s,
                      const std::vector<element_type>& s_old,
                      const std::vector<element_type>& cost,
                      const std::vector<element_type>& a_row,
                      const std::vector<element_type>& b_col,
                      int n1,
                      int n2,
                      element_type epsilon) {
    element_type diff_sum = static_cast<element_type>(0);
    #pragma omp parallel for reduction(+:diff_sum) schedule(static)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            const size_t idx = static_cast<size_t>(j) * static_cast<size_t>(n1) + static_cast<size_t>(i);
            const element_type next =
                static_cast<element_type>(0.95) * static_cast<element_type>(std::exp(static_cast<double>((a_row[static_cast<size_t>(j)] + b_col[static_cast<size_t>(i)] - cost[idx]) / epsilon)))
                    / static_cast<element_type>(n1 * n2)
                + static_cast<element_type>(0.05) * s_old[idx];
            diff_sum += std::abs(next - s_old[idx]);
            s[idx] = next;
        }
    }
    return diff_sum;
}

}  // namespace

std::vector<element_type> con_prox_pt_opt_device(const CudaSparseMatrix& row_norm_adj1,
                                                 const CudaSparseMatrix& row_norm_adj2,
                                                 int n1,
                                                 int n2,
                                                 int avg_edges1_per_n1,
                                                 int avg_edges2_per_n2,
                                                 const element_type* cross_dist,
                                                 const CudaDenseMatrix& intra_dist1_dense,
                                                 const CudaDenseMatrix& intra_dist2_dense,
                                                 int inner_iters,
                                                 int outer_iters,
                                                 element_type lambda_n,
                                                 element_type lambda_a,
                                                 element_type lambda_p,
                                                 element_type lambda_e_in,
                                                 const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs,
                                                 bool sparsify) {
    (void)avg_edges1_per_n1;
    (void)avg_edges2_per_n2;
    (void)sparsify;
    validate_anchors(n1, n2, anchor_pairs);

    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const element_type lambda_total = lambda_n + lambda_a + lambda_p;
    const element_type lambda_e = lambda_e_in * static_cast<element_type>(total);

    CudaDenseMatrix row_norm_adj1_dense = sparse_to_dense_col_major(row_norm_adj1);
    CudaDenseMatrix row_norm_adj2_dense = sparse_to_dense_col_major(row_norm_adj2);
    CudaDenseMatrix row_norm_adj1_t = transpose_cuda_dense(row_norm_adj1_dense);

    std::vector<element_type> s(total, static_cast<element_type>(1) / static_cast<element_type>(total));
    std::vector<element_type> s_old(total);
    std::vector<element_type> cost(total);
    std::vector<element_type> cost_old(total);
    std::vector<element_type> c_fixed(cross_dist, cross_dist + total);
    std::vector<element_type> buffer(total);
    std::vector<element_type> buffer2(total);
    std::vector<element_type> cost_t(total);
    std::vector<element_type> a_row(static_cast<size_t>(n2), static_cast<element_type>(0));
    std::vector<element_type> b_col(static_cast<size_t>(n1), static_cast<element_type>(1) / static_cast<element_type>(n2));
    std::vector<element_type> h_log = init_h_log(n1, n2, anchor_pairs);
    compute_cost_fixed(intra_dist1_dense, intra_dist2_dense, h_log, n1, n2, lambda_e, lambda_a, c_fixed);

    bool has_cost_old = false;
    for (int it = 0; it < outer_iters; ++it) {
        s_old = s;
        compute_cost_dense(row_norm_adj1_t, row_norm_adj2_dense, intra_dist1_dense, intra_dist2_dense,
                           c_fixed, s, n1, n2, lambda_e, lambda_n, cost, buffer, buffer2);

        if (!has_cost_old) {
            cost_old = cost;
            has_cost_old = true;
        } else {
            const element_type w_old = std::inner_product(s.begin(), s.end(), cost_old.begin(), static_cast<element_type>(0));
            const element_type w = std::inner_product(s.begin(), s.end(), cost.begin(), static_cast<element_type>(0));
            if (w <= w_old) {
                cost_old = cost;
            } else {
                cost = cost_old;
            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < total; ++idx) {
            cost[idx] -= lambda_p * static_cast<element_type>(std::log(static_cast<double>(std::max(s[idx], static_cast<element_type>(1e-30)))));
        }

        transpose_col_major(cost, n1, n2, cost_t);
        for (int inner = 0; inner < inner_iters; ++inner) {
            minaa_dense(cost, b_col, n1, n2, lambda_total, a_row);
            minaa_dense(cost_t, a_row, n2, n1, lambda_total, b_col);
        }
        const element_type diff_s = update_s(s, s_old, cost, a_row, b_col, n1, n2, lambda_total);
        DEBUG_ONLY(std::printf("inner iter %d and diff of S %.6f\n\n", it, static_cast<double>(diff_s)));
    }

    destroy_cuda_dense(row_norm_adj1_t);
    destroy_cuda_dense(row_norm_adj2_dense);
    destroy_cuda_dense(row_norm_adj1_dense);
    return s;
}

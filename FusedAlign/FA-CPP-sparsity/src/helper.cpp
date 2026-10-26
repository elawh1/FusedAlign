#include "cuda_helper.cuh"

namespace {

CudaDenseMatrix make_cuda_dense_from_host_impl(int rows,
                                               int cols,
                                               const std::vector<element_type>& h_vals) {
    CudaDenseMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    CHECK_CUDA(CUDA_MALLOC(&matrix.d_vals, h_vals.size() * sizeof(element_type)));
    CHECK_CUDA(cudaMemcpy(matrix.d_vals, h_vals.data(), h_vals.size() * sizeof(element_type), cudaMemcpyHostToHost));
    return matrix;
}

CudaSparseMatrix make_cuda_sparse_from_host_csr_impl(int rows,
                                                     int cols,
                                                     const std::vector<int>& h_row_offsets,
                                                     const std::vector<int>& h_col_indices,
                                                     const std::vector<element_type>& h_vals) {
    CudaSparseMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = static_cast<int>(h_vals.size());
    CHECK_CUDA(CUDA_MALLOC(&matrix.d_row_offsets, h_row_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(matrix.d_row_offsets, h_row_offsets.data(),
                          h_row_offsets.size() * sizeof(int), cudaMemcpyHostToHost));
    if (matrix.nnz > 0) {
        CHECK_CUDA(CUDA_MALLOC(&matrix.d_col_indices, h_col_indices.size() * sizeof(int)));
        CHECK_CUDA(CUDA_MALLOC(&matrix.d_vals, h_vals.size() * sizeof(element_type)));
        CHECK_CUDA(cudaMemcpy(matrix.d_col_indices, h_col_indices.data(),
                              h_col_indices.size() * sizeof(int), cudaMemcpyHostToHost));
        CHECK_CUDA(cudaMemcpy(matrix.d_vals, h_vals.data(),
                              h_vals.size() * sizeof(element_type), cudaMemcpyHostToHost));
    }
    return matrix;
}

void copy_cuda_sparse_to_host(const CudaSparseMatrix& matrix,
                              std::vector<int>& h_row_offsets,
                              std::vector<int>& h_col_indices,
                              std::vector<element_type>& h_vals) {
    h_row_offsets.assign(matrix.d_row_offsets, matrix.d_row_offsets + static_cast<size_t>(matrix.rows) + 1);
    h_col_indices.assign(matrix.d_col_indices, matrix.d_col_indices + static_cast<size_t>(matrix.nnz));
    h_vals.assign(matrix.d_vals, matrix.d_vals + static_cast<size_t>(matrix.nnz));
}

sparse_operation_t to_mkl_op(cusparseOperation_t op) {
    return op == CUSPARSE_OPERATION_TRANSPOSE ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE;
}

sparse_layout_t to_mkl_layout(cusparseOrder_t order) {
    return order == CUSPARSE_ORDER_ROW ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
}

sparse_status_t mkl_create_csr(sparse_matrix_t* handle, const CudaSparseMatrix& sparse) {
#if CUDA_USE_FLOAT
    return mkl_sparse_s_create_csr(handle,
                                   SPARSE_INDEX_BASE_ZERO,
                                   sparse.rows,
                                   sparse.cols,
                                   sparse.d_row_offsets,
                                   sparse.d_row_offsets + 1,
                                   sparse.d_col_indices,
                                   sparse.d_vals);
#else
    return mkl_sparse_d_create_csr(handle,
                                   SPARSE_INDEX_BASE_ZERO,
                                   sparse.rows,
                                   sparse.cols,
                                   sparse.d_row_offsets,
                                   sparse.d_row_offsets + 1,
                                   sparse.d_col_indices,
                                   sparse.d_vals);
#endif
}

sparse_status_t mkl_sparse_mm_real(sparse_operation_t operation,
                                   element_type alpha,
                                   sparse_matrix_t a,
                                   const matrix_descr& descr,
                                   sparse_layout_t layout,
                                   const element_type* b,
                                   int columns,
                                   int ldb,
                                   element_type beta,
                                   element_type* c,
                                   int ldc) {
#if CUDA_USE_FLOAT
    return mkl_sparse_s_mm(operation, alpha, a, descr, layout, b, columns, ldb, beta, c, ldc);
#else
    return mkl_sparse_d_mm(operation, alpha, a, descr, layout, b, columns, ldb, beta, c, ldc);
#endif
}

}  // namespace

void destroy_cuda_dense(CudaDenseMatrix& matrix) {
    cuda_free_and_null(matrix.d_vals);
    matrix.rows = 0;
    matrix.cols = 0;
}

CudaSparseMatrix make_cuda_sparse_from_edgelist(const std::vector<element_type>& edgelist, int m, int n) {
    const int edge_count = static_cast<int>(edgelist.size() / 2);
    std::vector<std::vector<int>> rows(static_cast<size_t>(m));
    for (int i = 0; i < edge_count; ++i) {
        const int row = static_cast<int>(edgelist[2 * i]);
        const int col = static_cast<int>(edgelist[2 * i + 1]);
        if (row < 0 || row >= m || col < 0 || col >= n) {
            throw std::runtime_error("edge index out of bounds in make_cuda_sparse_from_edgelist");
        }
        rows[static_cast<size_t>(row)].push_back(col);
    }

    std::vector<int> h_row_offsets(static_cast<size_t>(m) + 1, 0);
    std::vector<int> h_col_indices;
    std::vector<element_type> h_vals;
    for (int row = 0; row < m; ++row) {
        auto& cols = rows[static_cast<size_t>(row)];
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        h_row_offsets[static_cast<size_t>(row)] = static_cast<int>(h_col_indices.size());
        for (int col : cols) {
            h_col_indices.push_back(col);
            h_vals.push_back(static_cast<element_type>(1));
        }
    }
    h_row_offsets[static_cast<size_t>(m)] = static_cast<int>(h_col_indices.size());
    return make_cuda_sparse_from_host_csr_impl(m, n, h_row_offsets, h_col_indices, h_vals);
}

CudaDenseMatrix make_cuda_dense_from_edgelist(const std::vector<element_type>& edgelist, int m, int n) {
    const int edge_count = static_cast<int>(edgelist.size() / 2);
    std::vector<element_type> h_vals(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<element_type>(0));
    for (int i = 0; i < edge_count; ++i) {
        const int row = static_cast<int>(edgelist[2 * i]);
        const int col = static_cast<int>(edgelist[2 * i + 1]);
        if (row < 0 || row >= m || col < 0 || col >= n) {
            throw std::runtime_error("edge index out of bounds in make_cuda_dense_from_edgelist");
        }
        h_vals[static_cast<size_t>(col) * static_cast<size_t>(m) + static_cast<size_t>(row)] = static_cast<element_type>(1);
    }
    return make_cuda_dense_from_host_impl(m, n, h_vals);
}

void destroy_cuda_sparse(CudaSparseMatrix& matrix) {
    cuda_free_and_null(matrix.d_vals);
    cuda_free_and_null(matrix.d_col_indices);
    cuda_free_and_null(matrix.d_row_offsets);
    matrix.rows = 0;
    matrix.cols = 0;
    matrix.nnz = 0;
}

CudaDenseMatrix transpose_cuda_dense(const CudaDenseMatrix& matrix) {
    CudaDenseMatrix result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    const size_t total = static_cast<size_t>(result.rows) * static_cast<size_t>(result.cols);
    CHECK_CUDA(CUDA_MALLOC(&result.d_vals, total * sizeof(element_type)));

    const element_type alpha = static_cast<element_type>(1);
    const element_type beta = static_cast<element_type>(0);
    CHECK_CUBLAS(cuda_cublas_geam(nullptr, CUBLAS_OP_T, CUBLAS_OP_N,
                                  result.rows, result.cols,
                                  &alpha, matrix.d_vals, matrix.rows,
                                  &beta, matrix.d_vals, result.rows,
                                  result.d_vals, result.rows));
    return result;
}

CudaSparseMatrix transpose_cuda_sparse(const CudaSparseMatrix& matrix) {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<element_type> vals;
    copy_cuda_sparse_to_host(matrix, row_offsets, col_indices, vals);

    std::vector<int> out_row_offsets(static_cast<size_t>(matrix.cols) + 1, 0);
    for (int idx = 0; idx < matrix.nnz; ++idx) {
        ++out_row_offsets[static_cast<size_t>(col_indices[static_cast<size_t>(idx)]) + 1];
    }
    for (int row = 0; row < matrix.cols; ++row) {
        out_row_offsets[static_cast<size_t>(row + 1)] += out_row_offsets[static_cast<size_t>(row)];
    }

    std::vector<int> next = out_row_offsets;
    std::vector<int> out_col_indices(static_cast<size_t>(matrix.nnz));
    std::vector<element_type> out_vals(static_cast<size_t>(matrix.nnz));
    for (int row = 0; row < matrix.rows; ++row) {
        for (int idx = row_offsets[static_cast<size_t>(row)]; idx < row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
            const int col = col_indices[static_cast<size_t>(idx)];
            const int out_idx = next[static_cast<size_t>(col)]++;
            out_col_indices[static_cast<size_t>(out_idx)] = row;
            out_vals[static_cast<size_t>(out_idx)] = vals[static_cast<size_t>(idx)];
        }
    }
    return make_cuda_sparse_from_host_csr_impl(matrix.cols, matrix.rows, out_row_offsets, out_col_indices, out_vals);
}

void l2_norm_inplace_device(element_type* d_rwr, size_t n, size_t d) {
    if (n == 0 || d == 0) return;

    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < n; ++row) {
        element_type sum = static_cast<element_type>(0);
        for (size_t col = 0; col < d; ++col) {
            const element_type value = d_rwr[row + col * n];
            sum += value * value;
        }
        if (sum > static_cast<element_type>(0)) {
            const element_type inv_norm = static_cast<element_type>(1) / static_cast<element_type>(std::sqrt(static_cast<double>(sum)));
            for (size_t col = 0; col < d; ++col) {
                d_rwr[row + col * n] *= inv_norm;
            }
        }
    }
}

void get_normalized_neg_exp_dist_device(const element_type* d_rwr1,
                                        const element_type* d_rwr2,
                                        int n1,
                                        int n2,
                                        int l,
                                        element_type* d_cross_rwr_dist) {
    const size_t out_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (out_size == 0) return;

    const element_type alpha = static_cast<element_type>(-1);
    const element_type beta = static_cast<element_type>(0);
    CHECK_CUBLAS(cuda_cublas_gemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n1, n2, l,
                                  &alpha, d_rwr1, n1,
                                  d_rwr2, n2,
                                  &beta, d_cross_rwr_dist, n1));

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < out_size; ++idx) {
        d_cross_rwr_dist[idx] = static_cast<element_type>(std::exp(static_cast<double>(d_cross_rwr_dist[idx])));
    }
}

void anchor_cost_zero_device(element_type* d_A,
                             int nrows,
                             int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             element_type val) {
    for (size_t t = 0; t < anchors.first.size(); ++t) {
        const int row = anchors.first[t];
        const int col = anchors.second[t];
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
            std::printf("error anchor_cost_zero: out-of-range anchor (%d,%d)\n", row, col);
            continue;
        }
        d_A[static_cast<size_t>(col) * static_cast<size_t>(nrows) + static_cast<size_t>(row)] = val;
    }
}

CudaSparseMatrix row_normalize_graph_cuda(const CudaSparseMatrix& graph, int avg_edges_per_node) {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<element_type> vals;
    copy_cuda_sparse_to_host(graph, row_offsets, col_indices, vals);

    const int rows = graph.rows;
    const int cols = graph.cols;
    std::vector<int> isolated;
    for (int row = 0; row < rows; ++row) {
        element_type sum = static_cast<element_type>(0);
        for (int idx = row_offsets[static_cast<size_t>(row)]; idx < row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
            sum += vals[static_cast<size_t>(idx)];
        }
        if (sum < static_cast<element_type>(1e-10)) {
            isolated.push_back(row);
        }
    }
    DEBUG_ONLY(std::printf("avg edges/node %d num of isolated %zu\n", avg_edges_per_node, isolated.size()));

    if (!isolated.empty()) {
        std::vector<std::vector<int>> row_cols(static_cast<size_t>(rows));
        std::vector<std::vector<element_type>> row_vals(static_cast<size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            for (int idx = row_offsets[static_cast<size_t>(row)]; idx < row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
                row_cols[static_cast<size_t>(row)].push_back(col_indices[static_cast<size_t>(idx)]);
                row_vals[static_cast<size_t>(row)].push_back(vals[static_cast<size_t>(idx)]);
            }
        }
        for (int row : isolated) {
            row_cols[static_cast<size_t>(row)].resize(static_cast<size_t>(cols));
            row_vals[static_cast<size_t>(row)].assign(static_cast<size_t>(cols), static_cast<element_type>(1));
            std::iota(row_cols[static_cast<size_t>(row)].begin(), row_cols[static_cast<size_t>(row)].end(), 0);
        }

        row_offsets.assign(static_cast<size_t>(rows) + 1, 0);
        col_indices.clear();
        vals.clear();
        for (int row = 0; row < rows; ++row) {
            row_offsets[static_cast<size_t>(row)] = static_cast<int>(col_indices.size());
            col_indices.insert(col_indices.end(), row_cols[static_cast<size_t>(row)].begin(), row_cols[static_cast<size_t>(row)].end());
            vals.insert(vals.end(), row_vals[static_cast<size_t>(row)].begin(), row_vals[static_cast<size_t>(row)].end());
        }
        row_offsets[static_cast<size_t>(rows)] = static_cast<int>(col_indices.size());
    }

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        element_type sum = static_cast<element_type>(0);
        for (int idx = row_offsets[static_cast<size_t>(row)]; idx < row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
            sum += vals[static_cast<size_t>(idx)];
        }
        if (sum > static_cast<element_type>(1e-10)) {
            for (int idx = row_offsets[static_cast<size_t>(row)]; idx < row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
                vals[static_cast<size_t>(idx)] /= sum;
            }
        }
    }

    return make_cuda_sparse_from_host_csr_impl(rows, cols, row_offsets, col_indices, vals);
}

void cuda_sparse_spmm(cusparseHandle_t handle,
                      const CudaSparseMatrix& sparse,
                      cusparseOperation_t op_sparse,
                      const element_type* d_dense_b,
                      int b_rows,
                      int b_cols,
                      int ldb,
                      cusparseOrder_t b_order,
                      element_type beta,
                      element_type* d_dense_c,
                      int c_rows,
                      int c_cols,
                      int ldc,
                      cusparseOrder_t c_order,
                      element_type alpha) {
    (void)handle;
    (void)b_rows;
    (void)c_rows;
    if (b_cols != c_cols || b_order != c_order) {
        throw std::invalid_argument("cuda_sparse_spmm expects matching dense column counts and layouts");
    }

    sparse_matrix_t mat_a = nullptr;
    sparse_status_t status = mkl_create_csr(&mat_a, sparse);
    if (status != SPARSE_STATUS_SUCCESS) {
        CHECK_CUSPARSE(1);
    }

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    status = mkl_sparse_mm_real(to_mkl_op(op_sparse),
                                alpha,
                                mat_a,
                                descr,
                                to_mkl_layout(b_order),
                                d_dense_b,
                                b_cols,
                                ldb,
                                beta,
                                d_dense_c,
                                ldc);
    mkl_sparse_destroy(mat_a);
    if (status != SPARSE_STATUS_SUCCESS) {
        CHECK_CUSPARSE(1);
    }
}

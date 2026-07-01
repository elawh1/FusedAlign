#include "cuda_helper.cuh"

namespace {

constexpr int kBlockSize = 256;

__global__ void compute_row_inv_l2_norms_kernel(const element_type* matrix, size_t n, size_t d, element_type* inv_norms) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n) return;

    __shared__ element_type partial[kBlockSize];
    element_type sum = static_cast<element_type>(0.0);
    for (size_t col = static_cast<size_t>(threadIdx.x); col < d; col += blockDim.x) {
        const element_type value = matrix[row + col * n];
        sum += value * value;
    }

    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        inv_norms[row] = (partial[0] > static_cast<element_type>(0.0))
                             ? static_cast<element_type>(rsqrt(static_cast<double>(partial[0])))
                             : static_cast<element_type>(0.0);
    }
}

__global__ void scale_rows_kernel(element_type* matrix, size_t n, size_t total, const element_type* inv_norms) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const size_t row = idx % n;
    const element_type scale = inv_norms[row];
    if (scale != static_cast<element_type>(0.0)) {
        matrix[idx] *= scale;
    }
}

__global__ void exp_inplace_kernel(element_type* data, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] = exp(data[idx]);
    }
}

__global__ void anchor_cost_zero_kernel(element_type* matrix, int nrows, const int* rows, const int* cols, int k, element_type val) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < k) {
        const size_t offset = static_cast<size_t>(cols[idx]) * static_cast<size_t>(nrows) + static_cast<size_t>(rows[idx]);
        matrix[offset] = val;
    }
}

struct CublasHandle {
    cublasHandle_t handle = nullptr;
    CublasHandle() {
        CHECK_CUBLAS(cublasCreate(&handle));
    }
    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

CudaDenseMatrix make_cuda_dense_from_host_impl(int rows,
                                               int cols,
                                               const std::vector<element_type>& h_vals) {
    CudaDenseMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    CHECK_CUDA(CUDA_MALLOC(&matrix.d_vals, h_vals.size() * sizeof(element_type)));
    CHECK_CUDA(cudaMemcpy(matrix.d_vals, h_vals.data(), h_vals.size() * sizeof(element_type), cudaMemcpyHostToDevice));
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
    CHECK_CUDA(cudaMemcpy(matrix.d_row_offsets, h_row_offsets.data(), h_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (matrix.nnz > 0) {
        CHECK_CUDA(CUDA_MALLOC(&matrix.d_col_indices, h_col_indices.size() * sizeof(int)));
        CHECK_CUDA(CUDA_MALLOC(&matrix.d_vals, h_vals.size() * sizeof(element_type)));
        CHECK_CUDA(cudaMemcpy(matrix.d_col_indices, h_col_indices.data(), h_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(matrix.d_vals, h_vals.data(), h_vals.size() * sizeof(element_type), cudaMemcpyHostToDevice));
    }
    return matrix;
}

void copy_cuda_dense_to_host(const CudaDenseMatrix& matrix, std::vector<element_type>& h_vals) {
    const size_t total = static_cast<size_t>(matrix.rows) * static_cast<size_t>(matrix.cols);
    h_vals.resize(total);
    CHECK_CUDA(cudaMemcpy(h_vals.data(), matrix.d_vals, total * sizeof(element_type), cudaMemcpyDeviceToHost));
}

void copy_cuda_sparse_to_host(const CudaSparseMatrix& matrix,
                              std::vector<int>& h_row_offsets,
                              std::vector<int>& h_col_indices,
                              std::vector<element_type>& h_vals) {
    h_row_offsets.resize(static_cast<size_t>(matrix.rows) + 1);
    h_col_indices.resize(static_cast<size_t>(matrix.nnz));
    h_vals.resize(static_cast<size_t>(matrix.nnz));
    CHECK_CUDA(cudaMemcpy(h_row_offsets.data(), matrix.d_row_offsets,
                          h_row_offsets.size() * sizeof(int), cudaMemcpyDeviceToHost));
    if (matrix.nnz > 0) {
        CHECK_CUDA(cudaMemcpy(h_col_indices.data(), matrix.d_col_indices,
                              h_col_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_vals.data(), matrix.d_vals,
                              h_vals.size() * sizeof(element_type), cudaMemcpyDeviceToHost));
    }
}

}  // namespace

void destroy_cuda_dense(CudaDenseMatrix& matrix) {
    if (matrix.d_vals != nullptr) {
        CHECK_CUDA(CUDA_FREE(matrix.d_vals));
        matrix.d_vals = nullptr;
    }
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
            h_vals.push_back(static_cast<element_type>(1.0));
        }
    }
    h_row_offsets[static_cast<size_t>(m)] = static_cast<int>(h_col_indices.size());
    return make_cuda_sparse_from_host_csr_impl(m, n, h_row_offsets, h_col_indices, h_vals);
}

CudaDenseMatrix make_cuda_dense_from_edgelist(const std::vector<element_type>& edgelist, int m, int n) {
    const int edge_count = static_cast<int>(edgelist.size() / 2);
    std::vector<element_type> h_vals(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<element_type>(0.0));
    for (int i = 0; i < edge_count; ++i) {
        const int row = static_cast<int>(edgelist[2 * i]);
        const int col = static_cast<int>(edgelist[2 * i + 1]);
        if (row < 0 || row >= m || col < 0 || col >= n) {
            throw std::runtime_error("edge index out of bounds in make_cuda_dense_from_edgelist");
        }
        h_vals[static_cast<size_t>(col) * static_cast<size_t>(m) + static_cast<size_t>(row)] = static_cast<element_type>(1.0);
    }
    return make_cuda_dense_from_host_impl(m, n, h_vals);
}

void destroy_cuda_sparse(CudaSparseMatrix& matrix) {
    if (matrix.d_vals != nullptr) {
        CHECK_CUDA(CUDA_FREE(matrix.d_vals));
        matrix.d_vals = nullptr;
    }
    if (matrix.d_col_indices != nullptr) {
        CHECK_CUDA(CUDA_FREE(matrix.d_col_indices));
        matrix.d_col_indices = nullptr;
    }
    if (matrix.d_row_offsets != nullptr) {
        CHECK_CUDA(CUDA_FREE(matrix.d_row_offsets));
        matrix.d_row_offsets = nullptr;
    }
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

    CublasHandle handle;
    const element_type alpha = static_cast<element_type>(1.0);
    const element_type beta = static_cast<element_type>(0.0);
    CHECK_CUBLAS(cuda_cublas_geam(handle.handle,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  result.rows,
                                  result.cols,
                                  &alpha,
                                  matrix.d_vals,
                                  matrix.rows,
                                  &beta,
                                  matrix.d_vals,
                                  result.rows,
                                  result.d_vals,
                                  result.rows));
    return result;
}

CudaSparseMatrix transpose_cuda_sparse(const CudaSparseMatrix& matrix) {
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_indices;
    std::vector<element_type> h_vals;
    copy_cuda_sparse_to_host(matrix, h_row_offsets, h_col_indices, h_vals);

    std::vector<std::vector<std::pair<int, element_type>>> transposed(static_cast<size_t>(matrix.cols));
    for (int row = 0; row < matrix.rows; ++row) {
        for (int idx = h_row_offsets[static_cast<size_t>(row)]; idx < h_row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
            transposed[static_cast<size_t>(h_col_indices[static_cast<size_t>(idx)])].push_back({row, h_vals[static_cast<size_t>(idx)]});
        }
    }

    std::vector<int> out_row_offsets(static_cast<size_t>(matrix.cols) + 1, 0);
    std::vector<int> out_col_indices;
    std::vector<element_type> out_vals;
    out_col_indices.reserve(h_col_indices.size());
    out_vals.reserve(h_vals.size());
    for (int row = 0; row < matrix.cols; ++row) {
        auto& entries = transposed[static_cast<size_t>(row)];
        std::sort(entries.begin(), entries.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        out_row_offsets[static_cast<size_t>(row)] = static_cast<int>(out_col_indices.size());
        for (const auto& entry : entries) {
            out_col_indices.push_back(entry.first);
            out_vals.push_back(entry.second);
        }
    }
    out_row_offsets[static_cast<size_t>(matrix.cols)] = static_cast<int>(out_col_indices.size());
    return make_cuda_sparse_from_host_csr_impl(matrix.cols, matrix.rows, out_row_offsets, out_col_indices, out_vals);
}

void l2_norm_inplace_device(element_type* d_rwr, size_t n, size_t d) {
    if (n == 0 || d == 0) {
        return;
    }
    element_type* d_inv_norms = nullptr;
    CHECK_CUDA(CUDA_MALLOC(&d_inv_norms, n * sizeof(element_type)));
    compute_row_inv_l2_norms_kernel<<<static_cast<unsigned int>(n), kBlockSize>>>(d_rwr, n, d, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());
    const size_t total = n * d;
    const unsigned int blocks = static_cast<unsigned int>((total + kBlockSize - 1) / kBlockSize);
    scale_rows_kernel<<<blocks, kBlockSize>>>(d_rwr, n, total, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(CUDA_FREE(d_inv_norms));
}

void get_normalized_neg_exp_dist_device(const element_type* d_rwr1,
                                        const element_type* d_rwr2,
                                        int n1,
                                        int n2,
                                        int l,
                                        element_type* d_cross_rwr_dist) {
    const size_t out_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (out_size == 0) {
        return;
    }
    const element_type alpha = static_cast<element_type>(-1.0);
    const element_type beta = static_cast<element_type>(0.0);
    CublasHandle handle;
    CHECK_CUBLAS(cuda_cublas_gemm(handle.handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_T,
                                  n1,
                                  n2,
                                  l,
                                  &alpha,
                                  d_rwr1,
                                  n1,
                                  d_rwr2,
                                  n2,
                                  &beta,
                                  d_cross_rwr_dist,
                                  n1));
    const unsigned int blocks = static_cast<unsigned int>((out_size + kBlockSize - 1) / kBlockSize);
    exp_inplace_kernel<<<blocks, kBlockSize>>>(d_cross_rwr_dist, out_size);
    CHECK_CUDA(cudaGetLastError());
}

void anchor_cost_zero_device(element_type* d_A,
                             int nrows,
                             int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             element_type val) {
    std::vector<int> valid_rows;
    std::vector<int> valid_cols;
    valid_rows.reserve(anchors.first.size());
    valid_cols.reserve(anchors.second.size());
    for (size_t t = 0; t < anchors.first.size(); ++t) {
        const int i = anchors.first[t];
        const int j = anchors.second[t];
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            printf("error anchor_cost_zero: out-of-range anchor (%d,%d)\n", i, j);
            continue;
        }
        valid_rows.push_back(i);
        valid_cols.push_back(j);
    }

    if (valid_rows.empty()) {
        return;
    }

    int* d_rows = nullptr;
    int* d_cols = nullptr;
    const size_t anchor_bytes = valid_rows.size() * sizeof(int);
    CHECK_CUDA(CUDA_MALLOC(&d_rows, anchor_bytes));
    CHECK_CUDA(CUDA_MALLOC(&d_cols, anchor_bytes));
    CHECK_CUDA(cudaMemcpy(d_rows, valid_rows.data(), anchor_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, valid_cols.data(), anchor_bytes, cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((valid_rows.size() + kBlockSize - 1) / kBlockSize);
    anchor_cost_zero_kernel<<<blocks, kBlockSize>>>(d_A, nrows, d_rows, d_cols, static_cast<int>(valid_rows.size()), val);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(CUDA_FREE(d_cols));
    CHECK_CUDA(CUDA_FREE(d_rows));
}

CudaSparseMatrix row_normalize_graph_cuda(const CudaSparseMatrix& graph, int avg_edges_per_node) {
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_indices;
    std::vector<element_type> h_vals;
    copy_cuda_sparse_to_host(graph, h_row_offsets, h_col_indices, h_vals);
    const int rows = graph.rows;
    const int cols = graph.cols;
    const int n = rows;

    std::vector<element_type> graph_row_sum(n, static_cast<element_type>(0.0));
    for (int row = 0; row < n; ++row) {
        element_type sum = static_cast<element_type>(0.0);
        for (int idx = h_row_offsets[static_cast<size_t>(row)]; idx < h_row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
            sum += h_vals[static_cast<size_t>(idx)];
        }
        graph_row_sum[row] = sum;
    }

    std::vector<int> isolated;
    for (int i = 0; i < n; ++i) {
        if (graph_row_sum[i] < static_cast<element_type>(1e-10)) {
            isolated.push_back(i);
        }
    }
    DEBUG_ONLY(printf("avg edges/node %d num of isolated %ld\n", avg_edges_per_node, isolated.size()));

    if (!isolated.empty()) {
        DEBUG_ONLY(printf("connecting isolated nodes to all nodes\n"));
        std::vector<std::vector<int>> row_cols(static_cast<size_t>(rows));
        std::vector<std::vector<element_type>> row_vals(static_cast<size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            for (int idx = h_row_offsets[static_cast<size_t>(row)]; idx < h_row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
                row_cols[static_cast<size_t>(row)].push_back(h_col_indices[static_cast<size_t>(idx)]);
                row_vals[static_cast<size_t>(row)].push_back(h_vals[static_cast<size_t>(idx)]);
            }
        }
        for (int i : isolated) {
            auto& cols_i = row_cols[static_cast<size_t>(i)];
            auto& vals_i = row_vals[static_cast<size_t>(i)];
            cols_i.clear();
            vals_i.clear();
            for (int col = 0; col < cols; ++col) {
                cols_i.push_back(col);
                vals_i.push_back(static_cast<element_type>(1.0));
            }
        }
        h_row_offsets.assign(static_cast<size_t>(rows) + 1, 0);
        h_col_indices.clear();
        h_vals.clear();
        for (int row = 0; row < rows; ++row) {
            h_row_offsets[static_cast<size_t>(row)] = static_cast<int>(h_col_indices.size());
            h_col_indices.insert(h_col_indices.end(), row_cols[static_cast<size_t>(row)].begin(), row_cols[static_cast<size_t>(row)].end());
            h_vals.insert(h_vals.end(), row_vals[static_cast<size_t>(row)].begin(), row_vals[static_cast<size_t>(row)].end());
        }
        h_row_offsets[static_cast<size_t>(rows)] = static_cast<int>(h_col_indices.size());
    }

    for (int row = 0; row < rows; ++row) {
        element_type sum = static_cast<element_type>(0.0);
        for (int idx = h_row_offsets[static_cast<size_t>(row)]; idx < h_row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
            sum += h_vals[static_cast<size_t>(idx)];
        }
        if (sum > static_cast<element_type>(1e-10)) {
            for (int idx = h_row_offsets[static_cast<size_t>(row)]; idx < h_row_offsets[static_cast<size_t>(row + 1)]; ++idx) {
                h_vals[static_cast<size_t>(idx)] /= sum;
            }
        }
        DEBUG_ONLY(if (sum < 1e-10) printf("graph_row_sum[%d] is zero \n", row));
    }

    return make_cuda_sparse_from_host_csr_impl(rows, cols, h_row_offsets, h_col_indices, h_vals);
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
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c = nullptr;
    CHECK_CUSPARSE(cusparseCreateCsr(&mat_a,
                                     sparse.rows,
                                     sparse.cols,
                                     sparse.nnz,
                                     sparse.d_row_offsets,
                                     sparse.d_col_indices,
                                     sparse.d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_TYPE));
    CHECK_CUSPARSE(cusparseCreateDnMat(&mat_b, b_rows, b_cols, ldb,
                                       const_cast<element_type*>(d_dense_b), CUDA_R_TYPE, b_order));
    CHECK_CUSPARSE(cusparseCreateDnMat(&mat_c, c_rows, c_cols, ldc,
                                       d_dense_c, CUDA_R_TYPE, c_order));

    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           op_sparse,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           mat_a,
                                           mat_b,
                                           &beta,
                                           mat_c,
                                           CUDA_R_TYPE,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           &buffer_size));
    void* d_buffer = nullptr;
    if (buffer_size > 0) {
        CHECK_CUDA(CUDA_MALLOC(&d_buffer, buffer_size));
    }
    CHECK_CUSPARSE(cusparseSpMM(handle,
                                op_sparse,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                mat_a,
                                mat_b,
                                &beta,
                                mat_c,
                                CUDA_R_TYPE,
                                CUSPARSE_SPMM_ALG_DEFAULT,
                                d_buffer));
    if (d_buffer != nullptr) {
        CHECK_CUDA(CUDA_FREE(d_buffer));
    }
    CHECK_CUSPARSE(cusparseDestroyDnMat(mat_c));
    CHECK_CUSPARSE(cusparseDestroyDnMat(mat_b));
    CHECK_CUSPARSE(cusparseDestroySpMat(mat_a));
}

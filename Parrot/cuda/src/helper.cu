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

CudaCsrMatrix make_cuda_csr_from_host_impl(int rows,
                                           int cols,
                                           const std::vector<int>& h_row_offsets,
                                           const std::vector<int>& h_col_ind,
                                           const std::vector<element_type>& h_vals);

}  // namespace

void destroy_cuda_csr(CudaCsrMatrix& matrix) {
    if (matrix.descr != nullptr) {
        cusparseDestroySpMat(matrix.descr);
        matrix.descr = nullptr;
    }
    if (matrix.d_vals != nullptr) {
        CHECK_CUDA(PARROT_CUDA_FREE(matrix.d_vals));
        matrix.d_vals = nullptr;
    }
    if (matrix.d_col_ind != nullptr) {
        CHECK_CUDA(PARROT_CUDA_FREE(matrix.d_col_ind));
        matrix.d_col_ind = nullptr;
    }
    if (matrix.d_row_offsets != nullptr) {
        CHECK_CUDA(PARROT_CUDA_FREE(matrix.d_row_offsets));
        matrix.d_row_offsets = nullptr;
    }
    matrix.rows = 0;
    matrix.cols = 0;
    matrix.nnz = 0;
}

namespace {

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

__global__ void csr_row_sum_kernel(const int* row_offsets, const element_type* values, int rows, element_type* row_sums) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    element_type sum = static_cast<element_type>(0.0);
    for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
        sum += values[idx];
    }
    row_sums[row] = sum;
}

__global__ void csr_scale_rows_kernel(const int* row_offsets, element_type* values, const element_type* row_sums, int rows) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const element_type row_sum = row_sums[row];
    if (row_sum <= static_cast<element_type>(1e-10)) return;

    const element_type inv = static_cast<element_type>(1.0) / row_sum;
    for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
        values[idx] *= inv;
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

struct CusparseHandle {
    cusparseHandle_t handle = nullptr;
    CusparseHandle() {
        CHECK_CUSPARSE(cusparseCreate(&handle));
    }
    ~CusparseHandle() {
        if (handle != nullptr) {
            cusparseDestroy(handle);
        }
    }
};

CudaCsrMatrix make_cuda_csr_from_host_impl(int rows,
                                           int cols,
                                           const std::vector<int>& h_row_offsets,
                                           const std::vector<int>& h_col_ind,
                                           const std::vector<element_type>& h_vals) {
    CudaCsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = static_cast<int>(h_vals.size());
    CHECK_CUDA(PARROT_CUDA_MALLOC(&matrix.d_row_offsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&matrix.d_col_ind, matrix.nnz * sizeof(int)));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&matrix.d_vals, matrix.nnz * sizeof(element_type)));
    CHECK_CUDA(cudaMemcpy(matrix.d_row_offsets, h_row_offsets.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_col_ind, h_col_ind.data(), matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_vals, h_vals.data(), matrix.nnz * sizeof(element_type), cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCreateCsr(&matrix.descr,
                                     matrix.rows,
                                     matrix.cols,
                                     matrix.nnz,
                                     matrix.d_row_offsets,
                                     matrix.d_col_ind,
                                     matrix.d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     PARROT_CUDA_R_TYPE));
    return matrix;
}

void copy_cuda_csr_to_host(const CudaCsrMatrix& matrix,
                           std::vector<int>& h_row_offsets,
                           std::vector<int>& h_col_ind,
                           std::vector<element_type>& h_vals) {
    h_row_offsets.resize(static_cast<size_t>(matrix.rows) + 1);
    h_col_ind.resize(matrix.nnz);
    h_vals.resize(matrix.nnz);
    CHECK_CUDA(cudaMemcpy(h_row_offsets.data(), matrix.d_row_offsets, (static_cast<size_t>(matrix.rows) + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_col_ind.data(), matrix.d_col_ind, static_cast<size_t>(matrix.nnz) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vals.data(), matrix.d_vals, static_cast<size_t>(matrix.nnz) * sizeof(element_type), cudaMemcpyDeviceToHost));
}

CudaCsrMatrix transpose_cuda_csr_impl(const CudaCsrMatrix& matrix) {
    CusparseHandle handle;
    CudaCsrMatrix result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.nnz = matrix.nnz;

    CHECK_CUDA(PARROT_CUDA_MALLOC(&result.d_row_offsets, (static_cast<size_t>(result.rows) + 1) * sizeof(int)));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&result.d_col_ind, static_cast<size_t>(result.nnz) * sizeof(int)));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&result.d_vals, static_cast<size_t>(result.nnz) * sizeof(element_type)));

    size_t workspace_size = 0;
    void* d_workspace = nullptr;
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
        handle.handle,
        matrix.rows,
        matrix.cols,
        matrix.nnz,
        matrix.d_vals,
        matrix.d_row_offsets,
        matrix.d_col_ind,
        result.d_vals,
        result.d_row_offsets,
        result.d_col_ind,
        PARROT_CUDA_R_TYPE,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &workspace_size));

    if (workspace_size > 0) {
        CHECK_CUDA(PARROT_CUDA_MALLOC(&d_workspace, workspace_size));
    }

    CHECK_CUSPARSE(cusparseCsr2cscEx2(
        handle.handle,
        matrix.rows,
        matrix.cols,
        matrix.nnz,
        matrix.d_vals,
        matrix.d_row_offsets,
        matrix.d_col_ind,
        result.d_vals,
        result.d_row_offsets,
        result.d_col_ind,
        PARROT_CUDA_R_TYPE,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        d_workspace));

    if (d_workspace != nullptr) {
        CHECK_CUDA(PARROT_CUDA_FREE(d_workspace));
    }

    CHECK_CUSPARSE(cusparseCreateCsr(&result.descr,
                                     result.rows,
                                     result.cols,
                                     result.nnz,
                                     result.d_row_offsets,
                                     result.d_col_ind,
                                     result.d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     PARROT_CUDA_R_TYPE));
    return result;
}

}  // namespace

CudaCsrMatrix make_cuda_csr_from_edgelist(const std::vector<element_type>& edgelist, int m, int n) {
    const int nnz = static_cast<int>(edgelist.size() / 2);
    std::vector<std::pair<int, int>> edges;
    edges.reserve(nnz);
    for (int i = 0; i < nnz; ++i) {
        const int row = static_cast<int>(edgelist[2 * i]);
        const int col = static_cast<int>(edgelist[2 * i + 1]);
        if (row < 0 || row >= m || col < 0 || col >= n) {
            throw std::runtime_error("edge index out of bounds in make_cuda_csr_from_edgelist");
        }
        edges.emplace_back(row, col);
    }
    std::sort(edges.begin(), edges.end());

    std::vector<int> h_row_offsets(static_cast<size_t>(m) + 1, 0);
    std::vector<int> h_col_ind(nnz);
    std::vector<element_type> h_vals(nnz, 1.0);
    for (const auto& edge : edges) {
        h_row_offsets[static_cast<size_t>(edge.first) + 1]++;
    }
    for (int i = 0; i < m; ++i) {
        h_row_offsets[static_cast<size_t>(i) + 1] += h_row_offsets[static_cast<size_t>(i)];
    }
    std::vector<int> cursor = h_row_offsets;
    for (const auto& edge : edges) {
        h_col_ind[cursor[edge.first]++] = edge.second;
    }
    return make_cuda_csr_from_host_impl(m, n, h_row_offsets, h_col_ind, h_vals);
}

CudaCsrMatrix transpose_cuda_csr(const CudaCsrMatrix& matrix) {
    return transpose_cuda_csr_impl(matrix);
}

CudaSpMatView create_spmm_matrix_view(const CudaCsrMatrix& matrix) {
    CudaSpMatView view;
    CHECK_CUSPARSE(cusparseCreateCsr(&view.descr,
                                     matrix.rows,
                                     matrix.cols,
                                     matrix.nnz,
                                     matrix.d_row_offsets,
                                     matrix.d_col_ind,
                                     matrix.d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     PARROT_CUDA_R_TYPE));

    return view;
}

void destroy_spmm_matrix_view(CudaSpMatView& view) {
    if (view.descr != nullptr) {
        CHECK_CUSPARSE(cusparseDestroySpMat(view.descr));
        view.descr = nullptr;
    }
}

void l2_norm_inplace_device(element_type* d_rwr, size_t n, size_t d) {
    if (n == 0 || d == 0) {
        return;
    }
    element_type* d_inv_norms = nullptr;
    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_inv_norms, n * sizeof(element_type)));
    compute_row_inv_l2_norms_kernel<<<static_cast<unsigned int>(n), kBlockSize>>>(d_rwr, n, d, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());
    const size_t total = n * d;
    const unsigned int blocks = static_cast<unsigned int>((total + kBlockSize - 1) / kBlockSize);
    scale_rows_kernel<<<blocks, kBlockSize>>>(d_rwr, n, total, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(PARROT_CUDA_FREE(d_inv_norms));
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
    CHECK_CUBLAS(parrot_cublas_gemm(handle.handle,
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
    (void)ncols;
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
    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_rows, anchor_bytes));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_cols, anchor_bytes));
    CHECK_CUDA(cudaMemcpy(d_rows, valid_rows.data(), anchor_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, valid_cols.data(), anchor_bytes, cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((valid_rows.size() + kBlockSize - 1) / kBlockSize);
    anchor_cost_zero_kernel<<<blocks, kBlockSize>>>(d_A, nrows, d_rows, d_cols, static_cast<int>(valid_rows.size()), val);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(PARROT_CUDA_FREE(d_cols));
    CHECK_CUDA(PARROT_CUDA_FREE(d_rows));
}

CudaCsrMatrix row_normalize_graph_cuda(const CudaCsrMatrix& graph, int avg_nnz_per_n) {
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_ind;
    std::vector<element_type> h_vals;
    copy_cuda_csr_to_host(graph, h_row_offsets, h_col_ind, h_vals);
    const int rows = graph.rows;
    const int cols = graph.cols;
    const int n = rows;
    std::vector<element_type> graph_row_sum(n, static_cast<element_type>(0.0));
    for (int row = 0; row < n; ++row) {
        element_type sum = static_cast<element_type>(0.0);
        for (int idx = h_row_offsets[row]; idx < h_row_offsets[row + 1]; ++idx) {
            sum += h_vals[idx];
        }
        graph_row_sum[row] = sum;
    }

    std::vector<int> isolated;
    for (int i = 0; i < n; ++i) {
        if (graph_row_sum[i] < static_cast<element_type>(1e-10)) {
            isolated.push_back(i);
        }
    }
    DEBUG_ONLY(printf("ang nnz %d num of isolated %ld\n", avg_nnz_per_n, isolated.size()));

    if (!isolated.empty()) {
        DEBUG_ONLY(printf("connecting isolated nodes\n"));
        std::vector<std::vector<int>> add_cols_per_row(static_cast<size_t>(n));
        std::srand(std::time(0));
        for (int i : isolated) {
            std::set<int> random_nodes = {};
            while (random_nodes.size() < static_cast<size_t>(avg_nnz_per_n)) {
                int random_num = std::rand() % n;
                if (random_num == i || random_nodes.find(random_num) != random_nodes.end()) continue;
                random_nodes.insert(random_num);
                add_cols_per_row[static_cast<size_t>(i)].push_back(static_cast<int>(random_num));
                add_cols_per_row[static_cast<size_t>(random_num)].push_back(static_cast<int>(i));
            }
        }

        size_t added_edges = 0;
        for (const auto& cols_added : add_cols_per_row) {
            added_edges += cols_added.size();
        }
        DEBUG_ONLY(printf("edge_list size %d \n", static_cast<int>(added_edges * 2)));

        std::vector<int> merged_row_offsets(n + 1, 0);
        for (int row = 0; row < n; ++row) {
            const int old_nnz = h_row_offsets[row + 1] - h_row_offsets[row];
            const int add_nnz = static_cast<int>(add_cols_per_row[static_cast<size_t>(row)].size());
            merged_row_offsets[row + 1] = merged_row_offsets[row] + old_nnz + add_nnz;
        }

        const int nnz = merged_row_offsets[n];
        std::vector<int> merged_col_ind(nnz);
        std::vector<element_type> merged_vals(nnz, static_cast<element_type>(1.0));
        for (int row = 0; row < n; ++row) {
            int cursor = merged_row_offsets[row];
            for (int idx = h_row_offsets[row]; idx < h_row_offsets[row + 1]; ++idx) {
                merged_col_ind[cursor] = h_col_ind[idx];
                merged_vals[cursor] = h_vals[idx];
                cursor++;
            }
            for (int col_added : add_cols_per_row[static_cast<size_t>(row)]) {
                merged_col_ind[cursor++] = col_added;
            }
        }
        h_row_offsets.swap(merged_row_offsets);
        h_col_ind.swap(merged_col_ind);
        h_vals.swap(merged_vals);
    }

    int* d_row_offsets = nullptr;
    element_type* d_vals = nullptr;
    element_type* d_row_sums = nullptr;
    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_row_offsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_vals, h_vals.size() * sizeof(element_type)));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_row_sums, rows * sizeof(element_type)));
    CHECK_CUDA(cudaMemcpy(d_row_offsets, h_row_offsets.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_vals.data(), h_vals.size() * sizeof(element_type), cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((rows + kBlockSize - 1) / kBlockSize);
    csr_row_sum_kernel<<<blocks, kBlockSize>>>(d_row_offsets, d_vals, rows, d_row_sums);
    CHECK_CUDA(cudaGetLastError());
    csr_scale_rows_kernel<<<blocks, kBlockSize>>>(d_row_offsets, d_vals, d_row_sums, rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_vals.data(), d_vals, h_vals.size() * sizeof(element_type), cudaMemcpyDeviceToHost));

    CHECK_CUDA(PARROT_CUDA_FREE(d_row_sums));
    CHECK_CUDA(PARROT_CUDA_FREE(d_vals));
    CHECK_CUDA(PARROT_CUDA_FREE(d_row_offsets));

    for (int row = 0; row < rows; ++row) {
        element_type sum = static_cast<element_type>(0.0);
        for (int idx = h_row_offsets[row]; idx < h_row_offsets[row + 1]; ++idx) {
            sum += h_vals[idx];
        }
        graph_row_sum[row] = sum;
    }
    for (int i = 0; i < static_cast<int>(graph_row_sum.size()); ++i) {
        if (graph_row_sum[i] < 1e-10) DEBUG_ONLY(printf("333 graph_row_sum[%d] is zero \n", i));
    }

    return make_cuda_csr_from_host_impl(rows, cols, h_row_offsets, h_col_ind, h_vals);
}

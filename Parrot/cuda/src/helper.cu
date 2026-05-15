#include "cuda_helper.cuh"

namespace {

constexpr int kBlockSize = 256;

__global__ void compute_row_inv_l2_norms_kernel(const double* matrix, size_t n, size_t d, double* inv_norms) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n) return;

    __shared__ double partial[kBlockSize];
    double sum = 0.0;
    for (size_t col = static_cast<size_t>(threadIdx.x); col < d; col += blockDim.x) {
        const double value = matrix[row + col * n];
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
        inv_norms[row] = (partial[0] > 0.0) ? rsqrt(partial[0]) : 0.0;
    }
}

CudaCsrMatrix make_cuda_csr_from_host_impl(int rows,
                                           int cols,
                                           const std::vector<int>& h_row_offsets,
                                           const std::vector<int>& h_col_ind,
                                           const std::vector<double>& h_vals);

}  // namespace

void destroy_cuda_csr(CudaCsrMatrix& matrix) {
    if (matrix.descr != nullptr) {
        cusparseDestroySpMat(matrix.descr);
        matrix.descr = nullptr;
    }
    if (matrix.d_vals != nullptr) {
        cudaFree(matrix.d_vals);
        matrix.d_vals = nullptr;
    }
    if (matrix.d_col_ind != nullptr) {
        cudaFree(matrix.d_col_ind);
        matrix.d_col_ind = nullptr;
    }
    if (matrix.d_row_offsets != nullptr) {
        cudaFree(matrix.d_row_offsets);
        matrix.d_row_offsets = nullptr;
    }
    matrix.rows = 0;
    matrix.cols = 0;
    matrix.nnz = 0;
}

namespace {

__global__ void scale_rows_kernel(double* matrix, size_t n, size_t total, const double* inv_norms) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const size_t row = idx % n;
    const double scale = inv_norms[row];
    if (scale != 0.0) {
        matrix[idx] *= scale;
    }
}

__global__ void exp_inplace_kernel(double* data, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] = exp(data[idx]);
    }
}

__global__ void anchor_cost_zero_kernel(double* matrix, int nrows, const int* rows, const int* cols, int k, double val) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < k) {
        const size_t offset = static_cast<size_t>(cols[idx]) * static_cast<size_t>(nrows) + static_cast<size_t>(rows[idx]);
        matrix[offset] = val;
    }
}

__global__ void csr_row_sum_kernel(const int* row_offsets, const double* values, int rows, double* row_sums) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    double sum = 0.0;
    for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
        sum += values[idx];
    }
    row_sums[row] = sum;
}

__global__ void csr_scale_rows_kernel(const int* row_offsets, double* values, const double* row_sums, int rows) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const double row_sum = row_sums[row];
    if (row_sum <= 1e-10) return;

    const double inv = 1.0 / row_sum;
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
                                           const std::vector<double>& h_vals) {
    CudaCsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = static_cast<int>(h_vals.size());
    CHECK_CUDA(cudaMalloc(&matrix.d_row_offsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&matrix.d_col_ind, matrix.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&matrix.d_vals, matrix.nnz * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(matrix.d_row_offsets, h_row_offsets.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_col_ind, h_col_ind.data(), matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_vals, h_vals.data(), matrix.nnz * sizeof(double), cudaMemcpyHostToDevice));

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
                                     CUDA_R_64F));
    return matrix;
}

void copy_cuda_csr_to_host(const CudaCsrMatrix& matrix,
                           std::vector<int>& h_row_offsets,
                           std::vector<int>& h_col_ind,
                           std::vector<double>& h_vals) {
    h_row_offsets.resize(static_cast<size_t>(matrix.rows) + 1);
    h_col_ind.resize(matrix.nnz);
    h_vals.resize(matrix.nnz);
    CHECK_CUDA(cudaMemcpy(h_row_offsets.data(), matrix.d_row_offsets, (static_cast<size_t>(matrix.rows) + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_col_ind.data(), matrix.d_col_ind, static_cast<size_t>(matrix.nnz) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vals.data(), matrix.d_vals, static_cast<size_t>(matrix.nnz) * sizeof(double), cudaMemcpyDeviceToHost));
}

CudaCsrMatrix transpose_cuda_csr_impl(const CudaCsrMatrix& matrix) {
    CusparseHandle handle;
    CudaCsrMatrix result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.nnz = matrix.nnz;

    CHECK_CUDA(cudaMalloc(&result.d_row_offsets, (static_cast<size_t>(result.rows) + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&result.d_col_ind, static_cast<size_t>(result.nnz) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&result.d_vals, static_cast<size_t>(result.nnz) * sizeof(double)));

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
        CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &workspace_size));

    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
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
        CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        d_workspace));

    if (d_workspace != nullptr) {
        CHECK_CUDA(cudaFree(d_workspace));
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
                                     CUDA_R_64F));
    return result;
}

}  // namespace

CudaCsrMatrix make_cuda_csr_from_edgelist(const std::vector<double>& edgelist, int m, int n) {
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
    std::vector<double> h_vals(nnz, 1.0);
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
                                     CUDA_R_64F));

    return view;
}

void destroy_spmm_matrix_view(CudaSpMatView& view) {
    if (view.descr != nullptr) {
        CHECK_CUSPARSE(cusparseDestroySpMat(view.descr));
        view.descr = nullptr;
    }
}

void l2_norm_inplace_device(double* d_rwr, size_t n, size_t d) {
    if (n == 0 || d == 0) {
        return;
    }
    double* d_inv_norms = nullptr;
    CHECK_CUDA(cudaMalloc(&d_inv_norms, n * sizeof(double)));
    compute_row_inv_l2_norms_kernel<<<static_cast<unsigned int>(n), kBlockSize>>>(d_rwr, n, d, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());
    const size_t total = n * d;
    const unsigned int blocks = static_cast<unsigned int>((total + kBlockSize - 1) / kBlockSize);
    scale_rows_kernel<<<blocks, kBlockSize>>>(d_rwr, n, total, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_inv_norms));
}

void get_normalized_neg_exp_dist_device(const double* d_rwr1,
                                        const double* d_rwr2,
                                        int n1,
                                        int n2,
                                        int l,
                                        double* d_cross_rwr_dist) {
    const size_t out_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (out_size == 0) {
        return;
    }
    const double alpha = -1.0;
    const double beta = 0.0;
    CublasHandle handle;
    CHECK_CUBLAS(cublasDgemm(handle.handle,
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

void anchor_cost_zero_device(double* d_A,
                             int nrows,
                             int ncols,
                             const std::pair<std::vector<int>, std::vector<int>>& anchors,
                             double val) {
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
    CHECK_CUDA(cudaMalloc(&d_rows, anchor_bytes));
    CHECK_CUDA(cudaMalloc(&d_cols, anchor_bytes));
    CHECK_CUDA(cudaMemcpy(d_rows, valid_rows.data(), anchor_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, valid_cols.data(), anchor_bytes, cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((valid_rows.size() + kBlockSize - 1) / kBlockSize);
    anchor_cost_zero_kernel<<<blocks, kBlockSize>>>(d_A, nrows, d_rows, d_cols, static_cast<int>(valid_rows.size()), val);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rows));
}

CudaCsrMatrix row_normalize_graph_cuda(const CudaCsrMatrix& graph, int avg_nnz_per_n) {
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_ind;
    std::vector<double> h_vals;
    copy_cuda_csr_to_host(graph, h_row_offsets, h_col_ind, h_vals);
    const int rows = graph.rows;
    const int cols = graph.cols;
    const int n = rows;
    std::vector<double> graph_row_sum(n, 0.0);
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int idx = h_row_offsets[row]; idx < h_row_offsets[row + 1]; ++idx) {
            sum += h_vals[idx];
        }
        graph_row_sum[row] = sum;
    }

    std::vector<int> isolated;
    for (int i = 0; i < n; ++i) {
        if (graph_row_sum[i] < 1e-10) {
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
        std::vector<double> merged_vals(nnz, 1.0);
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
    double* d_vals = nullptr;
    double* d_row_sums = nullptr;
    CHECK_CUDA(cudaMalloc(&d_row_offsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_vals, h_vals.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_sums, rows * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_row_offsets, h_row_offsets.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_vals.data(), h_vals.size() * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((rows + kBlockSize - 1) / kBlockSize);
    csr_row_sum_kernel<<<blocks, kBlockSize>>>(d_row_offsets, d_vals, rows, d_row_sums);
    CHECK_CUDA(cudaGetLastError());
    csr_scale_rows_kernel<<<blocks, kBlockSize>>>(d_row_offsets, d_vals, d_row_sums, rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_vals.data(), d_vals, h_vals.size() * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_row_sums));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_row_offsets));

    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
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

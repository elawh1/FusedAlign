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

void copy_cuda_dense_to_host(const CudaDenseMatrix& matrix, std::vector<element_type>& h_vals) {
    const size_t total = static_cast<size_t>(matrix.rows) * static_cast<size_t>(matrix.cols);
    h_vals.resize(total);
    CHECK_CUDA(cudaMemcpy(h_vals.data(), matrix.d_vals, total * sizeof(element_type), cudaMemcpyDeviceToHost));
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

CudaDenseMatrix row_normalize_graph_cuda(const CudaDenseMatrix& graph, int avg_edges_per_node) {
    std::vector<element_type> h_vals;
    copy_cuda_dense_to_host(graph, h_vals);
    const int rows = graph.rows;
    const int cols = graph.cols;
    const int n = rows;

    std::vector<element_type> graph_row_sum(n, static_cast<element_type>(0.0));
    for (int row = 0; row < n; ++row) {
        element_type sum = static_cast<element_type>(0.0);
        for (int col = 0; col < cols; ++col) {
            sum += h_vals[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)];
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
        for (int i : isolated) {
            for (int col = 0; col < cols; ++col) {
                h_vals[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(i)] = static_cast<element_type>(1.0);
            }
        }
    }

    for (int row = 0; row < rows; ++row) {
        element_type sum = static_cast<element_type>(0.0);
        for (int col = 0; col < cols; ++col) {
            sum += h_vals[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)];
        }
        if (sum > static_cast<element_type>(1e-10)) {
            for (int col = 0; col < cols; ++col) {
                h_vals[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] /= sum;
            }
        }
        DEBUG_ONLY(if (sum < 1e-10) printf("graph_row_sum[%d] is zero \n", row));
    }

    return make_cuda_dense_from_host_impl(rows, cols, h_vals);
}

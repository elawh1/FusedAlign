#include "cuda_helper.cuh"

namespace {

constexpr int kBlockSize = 256;
constexpr int kCustomSpmmBlockSize = 256;

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

}  // namespace

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

namespace {

constexpr int kTransposeTileDim = 32;
constexpr int kTransposeBlockRows = 8;

// Tiled shared-memory transpose: coalesced reads and coalesced writes.
// Converts col-major (rows x cols) → row-major (rows x cols).
// smem padding (+1) eliminates shared memory bank conflicts.
__global__ void col_major_to_row_major_kernel(const element_type* __restrict__ in,
                                              int rows,
                                              int cols,
                                              element_type* __restrict__ out) {
    __shared__ element_type smem[kTransposeTileDim][kTransposeTileDim + 1];

    const int i_base = static_cast<int>(blockIdx.y) * kTransposeTileDim;
    const int j_base = static_cast<int>(blockIdx.x) * kTransposeTileDim;

    // Read phase: threadIdx.x strides over rows (i), coalesced reads from col-major.
    for (int dy = 0; dy < kTransposeTileDim; dy += kTransposeBlockRows) {
        const int gi = i_base + static_cast<int>(threadIdx.x);
        const int gj = j_base + static_cast<int>(threadIdx.y) + dy;
        if (gi < rows && gj < cols) {
            smem[threadIdx.x][threadIdx.y + dy] = in[static_cast<size_t>(gj) * rows + gi];
        }
    }
    __syncthreads();

    // Write phase: threadIdx.x strides over cols (j), coalesced writes to row-major.
    for (int dy = 0; dy < kTransposeTileDim; dy += kTransposeBlockRows) {
        const int gi = i_base + static_cast<int>(threadIdx.y) + dy;
        const int gj = j_base + static_cast<int>(threadIdx.x);
        if (gi < rows && gj < cols) {
            out[static_cast<size_t>(gi) * cols + gj] = smem[threadIdx.y + dy][threadIdx.x];
        }
    }
}

__global__ void csr_spmm_row_major_kernel(const int* __restrict__ row_offsets,
                                          const int* __restrict__ col_ind,
                                          const element_type* __restrict__ vals,
                                          int rows,
                                          int dense_cols,
                                          element_type alpha,
                                          const element_type* __restrict__ B,
                                          element_type beta,
                                          element_type* __restrict__ C) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(dense_cols);
    if (idx >= total) {
        return;
    }

    const int col = static_cast<int>(idx % static_cast<size_t>(dense_cols));
    const int row = static_cast<int>(idx / static_cast<size_t>(dense_cols));
    element_type out_val = static_cast<element_type>(0.0);
    if (beta != static_cast<element_type>(0.0)) {
        out_val = (beta == static_cast<element_type>(1.0)) ? C[idx] : beta * C[idx];
    }
    if (alpha != static_cast<element_type>(0.0)) {
        const int row_start = row_offsets[row];
        const int row_end = row_offsets[row + 1];
        for (int p = row_start; p < row_end; ++p) {
            const int k = col_ind[p];
            out_val += alpha * vals[p] * B[static_cast<size_t>(k) * dense_cols + col];
        }
    }
    C[idx] = out_val;
}

void custom_csr_spmm_row_major(const CudaCsrMatrix& graph,
                               int dense_cols,
                               element_type alpha,
                               const element_type* d_B,
                               element_type beta,
                               element_type* d_C) {
    const size_t total = static_cast<size_t>(graph.rows) * static_cast<size_t>(dense_cols);
    if (total == 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>((total + kCustomSpmmBlockSize - 1) / kCustomSpmmBlockSize);
    csr_spmm_row_major_kernel<<<blocks, kCustomSpmmBlockSize>>>(
        graph.d_row_offsets, graph.d_col_ind, graph.d_vals,
        graph.rows, dense_cols, alpha, d_B, beta, d_C);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void accumulate_cost_from_buffer2_kernel(const element_type* buffer2_row_major,
                                                    const element_type* c_fixed_col_major,
                                                    int n1,
                                                    int n2,
                                                    element_type* out_col_major) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (idx >= total) {
        return;
    }

    const int row = static_cast<int>(idx % static_cast<size_t>(n1));
    const int col = static_cast<int>(idx / static_cast<size_t>(n1));
    const size_t row_major_idx = static_cast<size_t>(row) * static_cast<size_t>(n2) + static_cast<size_t>(col);
    out_col_major[idx] = buffer2_row_major[row_major_idx] + c_fixed_col_major[idx];
}

__global__ void accumulate_log_term_from_buffer_kernel(const element_type* buffer_row_major,
                                                       element_type lambda_n,
                                                       int n1,
                                                       int n2,
                                                       element_type* inout_col_major) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (idx >= total) {
        return;
    }

    const int row = static_cast<int>(idx % static_cast<size_t>(n1));
    const int col = static_cast<int>(idx / static_cast<size_t>(n1));
    const size_t row_major_idx = static_cast<size_t>(row) * static_cast<size_t>(n2) + static_cast<size_t>(col);
    const element_type clamped = buffer_row_major[row_major_idx] > static_cast<element_type>(1e-15)
                                     ? buffer_row_major[row_major_idx]
                                     : static_cast<element_type>(1e-15);
    inout_col_major[idx] -= lambda_n * log(clamped);
}

__global__ void update_s_kernel(element_type* s,
                                const element_type* s_old,
                                const element_type* cost,
                                const element_type* a_row,
                                const element_type* b_col,
                                int n1,
                                int n2,
                                element_type epsilon,
                                element_type* block_sums) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);

    element_type local_sum = static_cast<element_type>(0.0);
    if (idx < total) {
        const int i = static_cast<int>(idx % static_cast<size_t>(n1));
        const int j = static_cast<int>(idx / static_cast<size_t>(n1));
        const element_type new_value =
            static_cast<element_type>(0.95) * static_cast<element_type>(exp(static_cast<double>((a_row[j] + b_col[i] - cost[idx]) / epsilon)))
            / static_cast<element_type>(n1 * n2)
            + static_cast<element_type>(0.05) * s_old[idx];
        s[idx] = new_value;
        local_sum = static_cast<element_type>(fabs(static_cast<double>(new_value - s_old[idx])));
    }

    __shared__ element_type partial[kBlockSize];
    partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = partial[0];
    }
}

__device__ __forceinline__ element_type warp_reduce_min_device(element_type value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmin(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ element_type warp_reduce_sum_device(element_type value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void minaa_v3_dense_kernel(const element_type* cost,
                                      const element_type* b,
                                      int n_row,
                                      int n_col,
                                      element_type epsilon,
                                      element_type* out_vec) {
    const int col_idx = static_cast<int>(blockIdx.x);
    if (col_idx >= n_col) return;

    const element_type* col = cost + static_cast<size_t>(col_idx) * static_cast<size_t>(n_row);

    element_type local_min = static_cast<element_type>(INFINITY);
    for (int row = threadIdx.x; row < n_row; row += blockDim.x) {
        local_min = fmin(local_min, col[row] - b[row]);
    }

    const int lane = static_cast<int>(threadIdx.x & 31);
    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ element_type warp_mins[8];
    local_min = warp_reduce_min_device(local_min);
    if (lane == 0) {
        warp_mins[warp_id] = local_min;
    }
    __syncthreads();

    element_type min_value = static_cast<element_type>(INFINITY);
    if (warp_id == 0) {
        min_value = (lane < num_warps) ? warp_mins[lane] : static_cast<element_type>(INFINITY);
        min_value = warp_reduce_min_device(min_value);
        if (lane == 0) {
            warp_mins[0] = min_value;
        }
    }
    __syncthreads();
    min_value = warp_mins[0];

    element_type local_sum = static_cast<element_type>(0.0);
    const element_type inv_epsilon = static_cast<element_type>(1.0) / epsilon;
    for (int row = threadIdx.x; row < n_row; row += blockDim.x) {
        const element_type shifted = col[row] - b[row] - min_value;
#if CUDA_USE_FLOAT
        local_sum += __expf(-shifted * inv_epsilon);
#else
        local_sum += static_cast<element_type>(exp(static_cast<double>(-shifted * inv_epsilon)));
#endif
    }

    __shared__ element_type warp_sums[8];
    local_sum = warp_reduce_sum_device(local_sum);
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    element_type sum_value = static_cast<element_type>(0.0);
    if (warp_id == 0) {
        sum_value = (lane < num_warps) ? warp_sums[lane] : static_cast<element_type>(0.0);
        sum_value = warp_reduce_sum_device(sum_value);
        if (lane == 0) {
            warp_sums[0] = sum_value;
        }
    }
    __syncthreads();
    sum_value = warp_sums[0];

    if (threadIdx.x == 0) {
        out_vec[col_idx] = min_value - epsilon * static_cast<element_type>(log(static_cast<double>(sum_value / static_cast<element_type>(n_row))));
    }
}

__global__ void minaa_v3_dense_kernel_baseline(const element_type* cost,
                                               const element_type* b,
                                               int n_row,
                                               int n_col,
                                               element_type epsilon,
                                               element_type* out_vec) {
    const int col_idx = static_cast<int>(blockIdx.x);
    if (col_idx >= n_col) return;

    const element_type* col = cost + static_cast<size_t>(col_idx) * static_cast<size_t>(n_row);

    element_type local_min = static_cast<element_type>(INFINITY);
    for (int row = threadIdx.x; row < n_row; row += blockDim.x) {
        local_min = fmin(local_min, col[row] - b[row]);
    }

    __shared__ element_type shared_min[kBlockSize];
    shared_min[threadIdx.x] = local_min;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_min[threadIdx.x] = fmin(shared_min[threadIdx.x], shared_min[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const element_type min_value = shared_min[0];

    element_type local_sum = static_cast<element_type>(0.0);
    for (int row = threadIdx.x; row < n_row; row += blockDim.x) {
        local_sum += static_cast<element_type>(exp(static_cast<double>(-((col[row] - b[row] - min_value) / epsilon))));
    }

    __shared__ element_type shared_sum[kBlockSize];
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_vec[col_idx] = min_value - epsilon * static_cast<element_type>(log(static_cast<double>(shared_sum[0] / static_cast<element_type>(n_row))));
    }
}

int choose_minaa_block_size(int n_row) {
    if (n_row <= 32) return 32;
    if (n_row <= 64) return 64;
    if (n_row <= 128) return 128;
    return 256;
}

void launch_minaa_dense_kernel(const element_type* cost,
                               const element_type* b,
                               int n_row,
                               int n_col,
                               element_type epsilon,
                               element_type* out_vec,
                               bool use_tuned_minaa) {
    if (use_tuned_minaa) {
        const int block_size = choose_minaa_block_size(n_row);
        minaa_v3_dense_kernel<<<static_cast<unsigned int>(n_col), block_size>>>(cost, b, n_row, n_col, epsilon, out_vec);
    } else {
        minaa_v3_dense_kernel_baseline<<<static_cast<unsigned int>(n_col), kBlockSize>>>(cost, b, n_row, n_col, epsilon, out_vec);
    }
    CHECK_CUDA(cudaGetLastError());
}

__global__ void csr_row_avg_square_kernel(const int* row_offsets,
                                          const element_type* values,
                                          int rows,
                                          element_type denom,
                                          element_type* row_avg_sqr) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    element_type square_sum = static_cast<element_type>(0.0);
    for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
        const element_type value = values[idx];
        square_sum += value * value;
    }
    row_avg_sqr[row] = square_sum / denom;
}

__global__ void compute_cost_fixed_kernel(element_type* cross_dist,
                                          const element_type* row_avg_square_dist1,
                                          const element_type* row_avg_square_dist2,
                                          const element_type* h_log,
                                          int n1,
                                          int n2,
                                          element_type lambda_e,
                                          element_type lambda_a) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (idx >= total) return;

    const int i = static_cast<int>(idx % static_cast<size_t>(n1));
    const int j = static_cast<int>(idx / static_cast<size_t>(n1));
    cross_dist[idx] += lambda_e * (row_avg_square_dist2[j] + row_avg_square_dist1[i]) - lambda_a * h_log[idx];
}

__global__ void fill_value_kernel(element_type* out, size_t total, element_type value) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = value;
    }
}

__global__ void scatter_anchor_values_kernel(element_type* out,
                                             const int* anchor_rows,
                                             const int* anchor_cols,
                                             int num_anchors,
                                             int n_rows,
                                             element_type value) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < num_anchors) {
        out[static_cast<size_t>(anchor_rows[t]) + static_cast<size_t>(anchor_cols[t]) * static_cast<size_t>(n_rows)] = value;
    }
}

__global__ void apply_log_penalty_kernel(element_type* s, element_type* cost, size_t total, element_type lambda_p) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const element_type log_s = static_cast<element_type>(log(static_cast<double>(fmax(s[idx], static_cast<element_type>(1e-30)))));
    s[idx] = log_s;
    cost[idx] -= lambda_p * log_s;
}

__global__ void dual_dot_partial_kernel(const element_type* s,
                                        const element_type* x,
                                        const element_type* y,
                                        size_t total,
                                        element_type* partial_x,
                                        element_type* partial_y) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    element_type local_x = static_cast<element_type>(0.0);
    element_type local_y = static_cast<element_type>(0.0);
    for (size_t idx = global_tid; idx < total; idx += stride) {
        const element_type s_val = s[idx];
        local_x += s_val * x[idx];
        local_y += s_val * y[idx];
    }

    __shared__ element_type shared_x[kBlockSize];
    __shared__ element_type shared_y[kBlockSize];
    shared_x[threadIdx.x] = local_x;
    shared_y[threadIdx.x] = local_y;
    __syncthreads();

    for (unsigned int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
        if (threadIdx.x < reduce_stride) {
            shared_x[threadIdx.x] += shared_x[threadIdx.x + reduce_stride];
            shared_y[threadIdx.x] += shared_y[threadIdx.x + reduce_stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_x[blockIdx.x] = shared_x[0];
        partial_y[blockIdx.x] = shared_y[0];
    }
}

__global__ void reduce_dual_dot_partials_kernel(const element_type* partial_x,
                                                const element_type* partial_y,
                                                int num_partials,
                                                element_type* out_pair) {
    element_type local_x = static_cast<element_type>(0.0);
    element_type local_y = static_cast<element_type>(0.0);
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        local_x += partial_x[idx];
        local_y += partial_y[idx];
    }

    __shared__ element_type shared_x[kBlockSize];
    __shared__ element_type shared_y[kBlockSize];
    shared_x[threadIdx.x] = local_x;
    shared_y[threadIdx.x] = local_y;
    __syncthreads();

    for (unsigned int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
        if (threadIdx.x < reduce_stride) {
            shared_x[threadIdx.x] += shared_x[threadIdx.x + reduce_stride];
            shared_y[threadIdx.x] += shared_y[threadIdx.x + reduce_stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_pair[0] = shared_x[0];
        out_pair[1] = shared_y[0];
    }
}

void compute_dual_dot_scores_device(const element_type* d_s,
                                    const element_type* d_cost_old,
                                    const element_type* d_cost_new,
                                    size_t total,
                                    unsigned int partial_blocks,
                                    element_type* d_partial_old,
                                    element_type* d_partial_new,
                                    element_type* d_pair_out,
                                    element_type& out_old,
                                    element_type& out_new) {
    if (total == 0 || partial_blocks == 0) {
        out_old = static_cast<element_type>(0.0);
        out_new = static_cast<element_type>(0.0);
        return;
    }

    dual_dot_partial_kernel<<<partial_blocks, kBlockSize>>>(
        d_s, d_cost_old, d_cost_new, total, d_partial_old, d_partial_new);
    CHECK_CUDA(cudaGetLastError());

    reduce_dual_dot_partials_kernel<<<1, kBlockSize>>>(
        d_partial_old, d_partial_new, static_cast<int>(partial_blocks), d_pair_out);
    CHECK_CUDA(cudaGetLastError());

    element_type host_pair[2] = {static_cast<element_type>(0.0), static_cast<element_type>(0.0)};
    CHECK_CUDA(cudaMemcpy(host_pair, d_pair_out, 2 * sizeof(element_type), cudaMemcpyDeviceToHost));
    out_old = host_pair[0];
    out_new = host_pair[1];
}

void transpose_col_major_matrix(const element_type* d_in,
                                int rows,
                                int cols,
                                element_type* d_out,
                                cublasHandle_t handle) {
    const element_type alpha = static_cast<element_type>(1.0);
    const element_type beta = static_cast<element_type>(0.0);
    CHECK_CUBLAS(cuda_cublas_geam(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             cols,
                             rows,
                             &alpha,
                             d_in,
                             rows,
                             &beta,
                             d_in,
                             cols,
                             d_out,
                             cols));
}

element_type update_s_device(element_type* d_s,
                       const element_type* d_s_old,
                       const element_type* d_cost,
                       const element_type* d_a_row,
                       const element_type* d_b_col,
                       int n1,
                       int n2,
                       element_type epsilon,
                       element_type* d_block_sums,
                       cublasHandle_t blas_handle) {
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const unsigned int blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    if (blocks == 0) {
        return static_cast<element_type>(0.0);
    }
    update_s_kernel<<<blocks, kBlockSize>>>(d_s, d_s_old, d_cost, d_a_row, d_b_col, n1, n2, epsilon, d_block_sums);
    CHECK_CUDA(cudaGetLastError());
#ifdef DEBUG
    element_type diff_sum = static_cast<element_type>(0.0);
    CHECK_CUBLAS(cuda_cublas_asum(blas_handle, static_cast<int>(blocks), d_block_sums, 1, &diff_sum));
    return diff_sum;
#else
    (void)blas_handle;
    return static_cast<element_type>(0.0);
#endif
}

void run_sinkhorn_iterations_cuda_device(const element_type* d_cost,
                                         int n1,
                                         int n2,
                                         int inner_iters,
                                         element_type epsilon,
                                         element_type* d_a_row,
                                         element_type* d_b_col,
                                         element_type* d_cost_t,
                                         cublasHandle_t blas_handle,
                                         bool use_tuned_minaa) {
    if (inner_iters <= 0) {
        return;
    }

    transpose_col_major_matrix(d_cost, n1, n2, d_cost_t, blas_handle);
    for (int iter = 0; iter < inner_iters; ++iter) {
        launch_minaa_dense_kernel(d_cost, d_b_col, n1, n2, epsilon, d_a_row, use_tuned_minaa);
        launch_minaa_dense_kernel(d_cost_t, d_a_row, n2, n1, epsilon, d_b_col, use_tuned_minaa);
    }
}

struct ComputeCostContext {
    cusparseDnMatDescr_t s_row_major = nullptr;
    cusparseDnMatDescr_t cost_row_major = nullptr;
    cusparseDnMatDescr_t cost_graph1_row_major = nullptr;
    cusparseDnMatDescr_t buffer_row_major = nullptr;
    cusparseDnMatDescr_t buffer2_row_major = nullptr;
    CudaSpMatView intra_dist2_view;
    CudaSpMatView intra_dist1_view;
    CudaSpMatView row_norm_adj2_view;
    CudaSpMatView row_norm_adj1_view;
    element_type* d_cost_graph1_row_major = nullptr;
    void* d_workspace_shared = nullptr;
    size_t workspace_shared_size = 0;
    element_type* d_s_row_major_ptr = nullptr;
    element_type* d_cost_row_major_ptr = nullptr;
    bool use_custom_spmm = false;
    bool use_graph1_transpose_view = false;
    bool benchmark_graph1_section = false;
    double graph1_convert_ms_total = 0.0;
    double graph1_spmm2_ms_total = 0.0;
    double graph1_spmm4_ms_total = 0.0;
    size_t graph1_timing_calls = 0;
    element_type alpha_neg2_lambda_e = static_cast<element_type>(0.0);
    size_t matrix_size = 0;
    int n1 = 0;
    int n2 = 0;
    CudaSpmmAlgorithm spmm_alg = {CUSPARSE_SPMM_ALG_DEFAULT, "default"};
};

void destroy_compute_cost_context(ComputeCostContext& context) {
    if (context.d_workspace_shared != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_workspace_shared));
        context.d_workspace_shared = nullptr;
    }
    if (context.buffer2_row_major != nullptr) {
        CHECK_CUSPARSE(cusparseDestroyDnMat(context.buffer2_row_major));
        context.buffer2_row_major = nullptr;
    }
    if (context.buffer_row_major != nullptr) {
        CHECK_CUSPARSE(cusparseDestroyDnMat(context.buffer_row_major));
        context.buffer_row_major = nullptr;
    }
    if (context.cost_graph1_row_major != nullptr) {
        CHECK_CUSPARSE(cusparseDestroyDnMat(context.cost_graph1_row_major));
        context.cost_graph1_row_major = nullptr;
    }
    if (context.cost_row_major != nullptr) {
        CHECK_CUSPARSE(cusparseDestroyDnMat(context.cost_row_major));
        context.cost_row_major = nullptr;
    }
    if (context.s_row_major != nullptr) {
        CHECK_CUSPARSE(cusparseDestroyDnMat(context.s_row_major));
        context.s_row_major = nullptr;
    }
    if (context.d_cost_graph1_row_major != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_cost_graph1_row_major));
        context.d_cost_graph1_row_major = nullptr;
    }
    destroy_spmm_matrix_view(context.row_norm_adj1_view);
    destroy_spmm_matrix_view(context.row_norm_adj2_view);
    destroy_spmm_matrix_view(context.intra_dist1_view);
    destroy_spmm_matrix_view(context.intra_dist2_view);
    context.d_s_row_major_ptr = nullptr;
    context.d_cost_row_major_ptr = nullptr;
    context.use_custom_spmm = false;
    context.use_graph1_transpose_view = false;
    context.benchmark_graph1_section = false;
    context.graph1_convert_ms_total = 0.0;
    context.graph1_spmm2_ms_total = 0.0;
    context.graph1_spmm4_ms_total = 0.0;
    context.graph1_timing_calls = 0;
    context.workspace_shared_size = 0;
    context.alpha_neg2_lambda_e = 0.0;
    context.matrix_size = 0;
    context.n1 = 0;
    context.n2 = 0;
    context.spmm_alg = {CUSPARSE_SPMM_ALG_DEFAULT, "default"};
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    const std::string token(value);
    return token == "1" || token == "true" || token == "TRUE" || token == "on" || token == "ON";
}

bool env_flag_enabled_default_true(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    const std::string token(value);
    if (token == "0" || token == "false" || token == "FALSE" || token == "off" || token == "OFF") {
        return false;
    }
    return true;
}

void initialize_compute_cost_context(ComputeCostContext& context,
                                     cusparseHandle_t sp_handle,
                                     const CudaCsrMatrix& d_row_norm_adj1,
                                     const CudaCsrMatrix& d_row_norm_adj2,
                                     const CudaCsrMatrix& d_intra_dist1,
                                     const CudaCsrMatrix& d_intra_dist2,
                                     const element_type* d_s,
                                     int n1,
                                     int n2,
                                     element_type lambda_e,
                                     element_type* d_cost,
                                     element_type* d_buffer,
                                     element_type* d_buffer2) {
    destroy_compute_cost_context(context);
    context.alpha_neg2_lambda_e = static_cast<element_type>(-2.0) * lambda_e;
    context.matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    context.n1 = n1;
    context.n2 = n2;
    context.use_custom_spmm = env_flag_enabled("CUDA_COST_USE_CUSTOM_SPMM");
    context.use_graph1_transpose_view = !context.use_custom_spmm && env_flag_enabled("CUDA_COST_USE_TRANSPOSE_VIEW");
#ifdef DEBUG
    context.benchmark_graph1_section = true;
#else
    context.benchmark_graph1_section = false;
#endif
    context.d_s_row_major_ptr = const_cast<element_type*>(d_s);
    context.d_cost_row_major_ptr = d_cost;
    context.spmm_alg = select_cuda_spmm_algorithm("CUDA_COST_SPMM_ALG");
    DEBUG_ONLY(std::cout << "optimization cost spmm backend: "
                         << (context.use_custom_spmm ? "custom" : "cusparse")
                         << std::endl);
    DEBUG_ONLY(std::cout << "optimization cost spmm alg: " << context.spmm_alg.name << std::endl);
    DEBUG_ONLY(std::cout << "optimization graph1 mode: "
                         << (context.use_graph1_transpose_view ? "row_major + transpose_view"
                                                               : "col_to_row + direct_row_major_spmm")
                         << std::endl);
    if (!context.use_graph1_transpose_view) {
        CHECK_CUDA(CUDA_MALLOC(&context.d_cost_graph1_row_major, context.matrix_size * sizeof(element_type)));
    }
    if (context.use_custom_spmm) {
        return;
    }

    context.intra_dist2_view = create_spmm_matrix_view(d_intra_dist2);
    context.intra_dist1_view = create_spmm_matrix_view(d_intra_dist1);
    context.row_norm_adj2_view = create_spmm_matrix_view(d_row_norm_adj2);
    context.row_norm_adj1_view = create_spmm_matrix_view(d_row_norm_adj1);

    CHECK_CUSPARSE(cusparseCreateDnMat(&context.s_row_major, n2, n1, n1, const_cast<element_type*>(d_s), CUDA_R_TYPE, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&context.cost_row_major, n2, n1, n1, d_cost, CUDA_R_TYPE, CUSPARSE_ORDER_ROW));
    if (!context.use_graph1_transpose_view) {
        CHECK_CUSPARSE(cusparseCreateDnMat(&context.cost_graph1_row_major, n1, n2, n2, context.d_cost_graph1_row_major, CUDA_R_TYPE, CUSPARSE_ORDER_ROW));
    }
    DEBUG_ONLY(std::cout << "optimization graph1 staging allocation: "
                         << (context.d_cost_graph1_row_major != nullptr ? "allocated" : "not allocated")
                         << std::endl);
    CHECK_CUSPARSE(cusparseCreateDnMat(&context.buffer_row_major, n1, n2, n2, d_buffer, CUDA_R_TYPE, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&context.buffer2_row_major, n1, n2, n2, d_buffer2, CUDA_R_TYPE, CUSPARSE_ORDER_ROW));

    size_t workspace_1 = 0;
    size_t workspace_2 = 0;
    size_t workspace_3 = 0;
    size_t workspace_4 = 0;
    const cusparseDnMatDescr_t graph1_cost_desc = context.use_graph1_transpose_view ? context.cost_row_major
                                                                                     : context.cost_graph1_row_major;
    const cusparseOperation_t graph1_cost_op = context.use_graph1_transpose_view ? CUSPARSE_OPERATION_TRANSPOSE
                                                                                  : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const element_type alpha_one = static_cast<element_type>(1.0);
    const element_type beta_zero = static_cast<element_type>(0.0);

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.intra_dist2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                           CUDA_R_TYPE, context.spmm_alg.value, &workspace_1));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, graph1_cost_op,
                                           &context.alpha_neg2_lambda_e, context.intra_dist1_view.descr, graph1_cost_desc, &beta_zero, context.buffer2_row_major,
                                           CUDA_R_TYPE, context.spmm_alg.value, &workspace_2));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.row_norm_adj2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                           CUDA_R_TYPE, context.spmm_alg.value, &workspace_3));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, graph1_cost_op,
                                           &alpha_one, context.row_norm_adj1_view.descr, graph1_cost_desc, &beta_zero, context.buffer_row_major,
                                           CUDA_R_TYPE, context.spmm_alg.value, &workspace_4));

    context.workspace_shared_size = std::max(std::max(workspace_1, workspace_2), std::max(workspace_3, workspace_4));
    DEBUG_ONLY(std::cout << "optimization SpMM workspace bytes: ["
                         << workspace_1 << ", "
                         << workspace_2 << ", "
                         << workspace_3 << ", "
                         << workspace_4 << "], shared="
                         << context.workspace_shared_size << std::endl);
    if (context.workspace_shared_size > 0) {
        CHECK_CUDA(CUDA_MALLOC(&context.d_workspace_shared, context.workspace_shared_size));
    }

}

void compute_cost_fused_2_cuda_device(cusparseHandle_t sp_handle,
                                      const ComputeCostContext& context,
                                      const CudaCsrMatrix& d_row_norm_adj1,
                                      const CudaCsrMatrix& d_row_norm_adj2,
                                      const CudaCsrMatrix& d_intra_dist1,
                                      const CudaCsrMatrix& d_intra_dist2,
                                      const element_type* d_c_fixed,
                                      element_type lambda_n,
                                      element_type* d_cost,
                                      element_type* d_buffer,
                                      element_type* d_buffer2) {
    const element_type alpha_one = static_cast<element_type>(1.0);
    const element_type beta_zero = static_cast<element_type>(0.0);
    const unsigned int blocks = static_cast<unsigned int>((context.matrix_size + kBlockSize - 1) / kBlockSize);
    const cusparseDnMatDescr_t graph1_cost_desc = context.use_graph1_transpose_view ? context.cost_row_major
                                                                                     : context.cost_graph1_row_major;
    const cusparseOperation_t graph1_cost_op = context.use_graph1_transpose_view ? CUSPARSE_OPERATION_TRANSPOSE
                                                                                  : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const bool benchmark = context.benchmark_graph1_section;
    float convert_ms_local = 0.0f;
    float spmm2_ms_local = 0.0f;
    float spmm4_ms_local = 0.0f;
    cudaEvent_t evt_start = nullptr;
    cudaEvent_t evt_stop = nullptr;
    if (benchmark) {
        CHECK_CUDA(cudaEventCreate(&evt_start));
        CHECK_CUDA(cudaEventCreate(&evt_stop));
    }

    if (context.use_custom_spmm) {
        custom_csr_spmm_row_major(d_intra_dist2, context.n1, alpha_one, context.d_s_row_major_ptr, beta_zero, d_cost);
    } else {
        CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_one, context.intra_dist2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                    CUDA_R_TYPE, context.spmm_alg.value, context.d_workspace_shared));
    }

    if (!context.use_graph1_transpose_view) {
        if (benchmark) CHECK_CUDA(cudaEventRecord(evt_start));
        {
            const dim3 transpose_block(kTransposeTileDim, kTransposeBlockRows);
            const dim3 transpose_grid(
                static_cast<unsigned int>((context.n2 + kTransposeTileDim - 1) / kTransposeTileDim),
                static_cast<unsigned int>((context.n1 + kTransposeTileDim - 1) / kTransposeTileDim));
            col_major_to_row_major_kernel<<<transpose_grid, transpose_block>>>(d_cost, context.n1, context.n2, context.d_cost_graph1_row_major);
        }
        CHECK_CUDA(cudaGetLastError());
        if (benchmark) {
            CHECK_CUDA(cudaEventRecord(evt_stop));
            CHECK_CUDA(cudaEventSynchronize(evt_stop));
            CHECK_CUDA(cudaEventElapsedTime(&convert_ms_local, evt_start, evt_stop));
            CHECK_CUDA(cudaEventRecord(evt_start));
        }
    } else if (benchmark) {
        CHECK_CUDA(cudaEventRecord(evt_start));
    }
    if (context.use_custom_spmm) {
        custom_csr_spmm_row_major(d_intra_dist1, context.n2, context.alpha_neg2_lambda_e, context.d_cost_graph1_row_major, beta_zero, d_buffer2);
    } else {
        CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, graph1_cost_op,
                                    &context.alpha_neg2_lambda_e, context.intra_dist1_view.descr, graph1_cost_desc, &beta_zero, context.buffer2_row_major,
                                    CUDA_R_TYPE, context.spmm_alg.value, context.d_workspace_shared));
    }
    if (benchmark) {
        CHECK_CUDA(cudaEventRecord(evt_stop));
        CHECK_CUDA(cudaEventSynchronize(evt_stop));
        CHECK_CUDA(cudaEventElapsedTime(&spmm2_ms_local, evt_start, evt_stop));
    }

    if (context.use_custom_spmm) {
        custom_csr_spmm_row_major(d_row_norm_adj2, context.n1, alpha_one, context.d_s_row_major_ptr, beta_zero, d_cost);
    } else {
        CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_one, context.row_norm_adj2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                    CUDA_R_TYPE, context.spmm_alg.value, context.d_workspace_shared));
    }

    if (!context.use_graph1_transpose_view) {
        if (benchmark) CHECK_CUDA(cudaEventRecord(evt_start));
        {
            const dim3 transpose_block(kTransposeTileDim, kTransposeBlockRows);
            const dim3 transpose_grid(
                static_cast<unsigned int>((context.n2 + kTransposeTileDim - 1) / kTransposeTileDim),
                static_cast<unsigned int>((context.n1 + kTransposeTileDim - 1) / kTransposeTileDim));
            col_major_to_row_major_kernel<<<transpose_grid, transpose_block>>>(d_cost, context.n1, context.n2, context.d_cost_graph1_row_major);
        }
        CHECK_CUDA(cudaGetLastError());
        if (benchmark) {
            CHECK_CUDA(cudaEventRecord(evt_stop));
            CHECK_CUDA(cudaEventSynchronize(evt_stop));
            float convert_ms_second = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&convert_ms_second, evt_start, evt_stop));
            convert_ms_local += convert_ms_second;
            CHECK_CUDA(cudaEventRecord(evt_start));
        }
    } else if (benchmark) {
        CHECK_CUDA(cudaEventRecord(evt_start));
    }
    if (context.use_custom_spmm) {
        custom_csr_spmm_row_major(d_row_norm_adj1, context.n2, alpha_one, context.d_cost_graph1_row_major, beta_zero, d_buffer);
    } else {
        CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, graph1_cost_op,
                                    &alpha_one, context.row_norm_adj1_view.descr, graph1_cost_desc, &beta_zero, context.buffer_row_major,
                                    CUDA_R_TYPE, context.spmm_alg.value, context.d_workspace_shared));
    }
    if (benchmark) {
        CHECK_CUDA(cudaEventRecord(evt_stop));
        CHECK_CUDA(cudaEventSynchronize(evt_stop));
        CHECK_CUDA(cudaEventElapsedTime(&spmm4_ms_local, evt_start, evt_stop));
    }
    accumulate_cost_from_buffer2_kernel<<<blocks, kBlockSize>>>(
        d_buffer2, d_c_fixed, context.n1, context.n2, d_cost);
    CHECK_CUDA(cudaGetLastError());
    accumulate_log_term_from_buffer_kernel<<<blocks, kBlockSize>>>(
        d_buffer, lambda_n, context.n1, context.n2, d_cost);
    CHECK_CUDA(cudaGetLastError());

    if (benchmark) {
        ComputeCostContext& mutable_context = const_cast<ComputeCostContext&>(context);
        mutable_context.graph1_convert_ms_total += static_cast<double>(convert_ms_local);
        mutable_context.graph1_spmm2_ms_total += static_cast<double>(spmm2_ms_local);
        mutable_context.graph1_spmm4_ms_total += static_cast<double>(spmm4_ms_local);
        mutable_context.graph1_timing_calls += 1;
        CHECK_CUDA(cudaEventDestroy(evt_stop));
        CHECK_CUDA(cudaEventDestroy(evt_start));
    }
}

} // namespace


std::vector<element_type> con_prox_pt_opt_device(const CudaCsrMatrix& row_norm_adj1,
                                                 const CudaCsrMatrix& row_norm_adj2,
                                                 int n1,
                                                 int n2,
                                                 int avg_nnz1_per_n1,
                                                 int avg_nnz2_per_n2,
                                                 const element_type* d_cross_dist,
                                                 const CudaCsrMatrix& intra_dist1_sparse,
                                                 const CudaCsrMatrix& intra_dist2_sparse,
                                                 int inner_iters,
                                                 int outer_iters,
                                                 element_type lambda_n,
                                                 element_type lambda_a,
                                                 element_type lambda_p,
                                                 element_type lambda_e_in,
                                                 const std::pair<std::vector<int>, std::vector<int>>& anchor_pairs,
                                                 bool sparsify) {
    (void)avg_nnz1_per_n1;
    (void)avg_nnz2_per_n2;
    (void)sparsify;

#ifdef DEBUG
    using Clock = std::chrono::high_resolution_clock;
    auto elapsed_ms = [](const Clock::time_point& start, const Clock::time_point& end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };
    auto synced_now = []() {
        CHECK_CUDA(cudaDeviceSynchronize());
        return Clock::now();
    };
    const auto debug_time_0 = synced_now();
#endif
    const element_type lambda_total = lambda_n + lambda_a + lambda_p;
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const element_type lambda_e = lambda_e_in * static_cast<element_type>(matrix_size);
    validate_anchors(n1, n2, anchor_pairs);
#ifdef DEBUG
    const auto debug_time_1 = synced_now();
#endif

    CusparseHandle sp_handle;
    CublasHandle blas_handle;
    CudaCsrMatrix d_row_norm_adj1_tr = transpose_cuda_csr(row_norm_adj1);
    const bool use_dual_dot = env_flag_enabled_default_true("CUDA_OPT_DUAL_DOT");
    const bool use_tuned_minaa = env_flag_enabled_default_true("CUDA_OPT_MINAA_TUNED");
    DEBUG_ONLY(std::cout << "optimization toggles: dual_dot=" << (use_dual_dot ? "on" : "off")
                         << ", minaa_tuned=" << (use_tuned_minaa ? "on" : "off") << std::endl);

    element_type* d_row_avg_square_dist1 = nullptr;
    element_type* d_row_avg_square_dist2 = nullptr;
    CHECK_CUDA(CUDA_MALLOC(&d_row_avg_square_dist1, static_cast<size_t>(n1) * sizeof(element_type)));
    CHECK_CUDA(CUDA_MALLOC(&d_row_avg_square_dist2, static_cast<size_t>(n2) * sizeof(element_type)));

    const unsigned int update_blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    const size_t pool_elems = 6 * matrix_size
        + static_cast<size_t>(n2)             // d_a_row
        + static_cast<size_t>(n1)             // d_b_col
        + static_cast<size_t>(update_blocks)  // d_update_block_sums
        + static_cast<size_t>(update_blocks)  // d_dot_partial_old
        + static_cast<size_t>(update_blocks)  // d_dot_partial_new
        + 2;                                  // d_dot_pair_out
    element_type* d_pool = nullptr;
    CHECK_CUDA(CUDA_MALLOC(&d_pool, pool_elems * sizeof(element_type)));

    element_type* d_s               = d_pool;
    element_type* d_cost            = d_s             + matrix_size;
    element_type* d_cost_buff       = d_cost          + matrix_size;
    element_type* d_c_fixed         = d_cost_buff     + matrix_size;
    element_type* d_buffer          = d_c_fixed       + matrix_size;
    element_type* d_buffer2         = d_buffer        + matrix_size;
    element_type* d_a_row           = d_buffer2       + matrix_size;
    element_type* d_b_col           = d_a_row         + n2;
    element_type* d_update_block_sums = d_b_col       + n1;
    element_type* d_dot_partial_old = d_update_block_sums + update_blocks;
    element_type* d_dot_partial_new = d_dot_partial_old   + update_blocks;
    element_type* d_dot_pair_out    = d_dot_partial_new   + update_blocks;

    element_type* d_h_log = nullptr;
    CHECK_CUDA(CUDA_MALLOC(&d_h_log, matrix_size * sizeof(element_type)));

    std::vector<element_type> S(matrix_size);  // pre-fault pages now so D2H copy at the end is fast
    {
        const unsigned int fill_blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
        fill_value_kernel<<<fill_blocks, kBlockSize>>>(d_s, matrix_size,
            static_cast<element_type>(1.0) / static_cast<element_type>(matrix_size));
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaMemcpy(d_c_fixed, d_cross_dist, matrix_size * sizeof(element_type), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(d_h_log, 0, matrix_size * sizeof(element_type)));
    if (!anchor_pairs.first.empty()) {
        const size_t num_anchors = anchor_pairs.first.size();
        int* d_anchor_rows = nullptr;
        int* d_anchor_cols = nullptr;
        CHECK_CUDA(CUDA_MALLOC(&d_anchor_rows, num_anchors * sizeof(int)));
        CHECK_CUDA(CUDA_MALLOC(&d_anchor_cols, num_anchors * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_anchor_rows, anchor_pairs.first.data(), num_anchors * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_anchor_cols, anchor_pairs.second.data(), num_anchors * sizeof(int), cudaMemcpyHostToDevice));
        const unsigned int anchor_blocks = static_cast<unsigned int>((num_anchors + kBlockSize - 1) / kBlockSize);
        scatter_anchor_values_kernel<<<anchor_blocks, kBlockSize>>>(d_h_log, d_anchor_rows, d_anchor_cols,
            static_cast<int>(num_anchors), n1, static_cast<element_type>(std::log(1e-6)));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(CUDA_FREE(d_anchor_rows));
        CHECK_CUDA(CUDA_FREE(d_anchor_cols));
    }
    CHECK_CUDA(cudaMemset(d_a_row, 0, static_cast<size_t>(n2) * sizeof(element_type)));
    std::vector<element_type> b_col(static_cast<size_t>(n1), static_cast<element_type>(1.0) / static_cast<element_type>(n2));
    CHECK_CUDA(cudaMemcpy(d_b_col, b_col.data(), static_cast<size_t>(n1) * sizeof(element_type), cudaMemcpyHostToDevice));

    const unsigned int blocks_row_1 = static_cast<unsigned int>((n1 + kBlockSize - 1) / kBlockSize);
    const unsigned int blocks_row_2 = static_cast<unsigned int>((n2 + kBlockSize - 1) / kBlockSize);
    cudaStream_t row_avg_stream_1 = nullptr;
    cudaStream_t row_avg_stream_2 = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&row_avg_stream_1, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&row_avg_stream_2, cudaStreamNonBlocking));
    csr_row_avg_square_kernel<<<blocks_row_1, kBlockSize, 0, row_avg_stream_1>>>(
        intra_dist1_sparse.d_row_offsets, intra_dist1_sparse.d_vals, n1,
        static_cast<element_type>(n1), d_row_avg_square_dist1);
    CHECK_CUDA(cudaGetLastError());
    csr_row_avg_square_kernel<<<blocks_row_2, kBlockSize, 0, row_avg_stream_2>>>(
        intra_dist2_sparse.d_row_offsets, intra_dist2_sparse.d_vals, n2,
        static_cast<element_type>(n2), d_row_avg_square_dist2);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(row_avg_stream_1));
    CHECK_CUDA(cudaStreamSynchronize(row_avg_stream_2));
    CHECK_CUDA(cudaStreamDestroy(row_avg_stream_1));
    CHECK_CUDA(cudaStreamDestroy(row_avg_stream_2));
    const unsigned int blocks_matrix = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    compute_cost_fixed_kernel<<<blocks_matrix, kBlockSize>>>(d_c_fixed, d_row_avg_square_dist1, d_row_avg_square_dist2, d_h_log, n1, n2, lambda_e, lambda_a);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(CUDA_FREE(d_h_log));
    CHECK_CUDA(CUDA_FREE(d_row_avg_square_dist2));
    CHECK_CUDA(CUDA_FREE(d_row_avg_square_dist1));

#ifdef DEBUG
    const auto debug_time_2 = synced_now();
#endif

    ComputeCostContext compute_cost_context;
#ifdef DEBUG
    size_t mem_free_before_context = 0;
    size_t mem_total_before_context = 0;
    size_t mem_free_after_context = 0;
    size_t mem_total_after_context = 0;
    CHECK_CUDA(cudaMemGetInfo(&mem_free_before_context, &mem_total_before_context));
#endif
#ifdef DEBUG
    const auto debug_context_start = synced_now();
#endif
    initialize_compute_cost_context(compute_cost_context, sp_handle.handle, d_row_norm_adj1_tr, row_norm_adj2,
                                    intra_dist1_sparse, intra_dist2_sparse, d_s, n1, n2, lambda_e,
                                    d_cost, d_buffer, d_buffer2);
#ifdef DEBUG
    CHECK_CUDA(cudaMemGetInfo(&mem_free_after_context, &mem_total_after_context));
    const auto debug_context_end = synced_now();
#endif

    bool has_cost_old = false;
#ifdef DEBUG
    long long c_compute_time = 0;
    long long cost_compute_time = 0;
    long long inner_loop = 0;
    long long update_s_time = 0;
#endif

    for (int it = 0; it < outer_iters; ++it) {
        DEBUG_ONLY(printf("----- outer iterration %d ----\n", it));

#ifdef DEBUG
        const auto debug_compute_start = synced_now();
#endif
        compute_cost_fused_2_cuda_device(sp_handle.handle, compute_cost_context, d_row_norm_adj1_tr, row_norm_adj2,
                                         intra_dist1_sparse, intra_dist2_sparse, d_c_fixed, lambda_n,
                                         d_cost, d_buffer, d_buffer2);
#ifdef DEBUG
        const auto debug_compute_end = synced_now();
        c_compute_time += elapsed_ms(debug_compute_start, debug_compute_end);
#endif

        if (!has_cost_old) {
            CHECK_CUDA(cudaMemcpy(d_cost_buff, d_cost, matrix_size * sizeof(element_type), cudaMemcpyDeviceToDevice));
            has_cost_old = true;
        } else {
            element_type w_old = static_cast<element_type>(0.0);
            element_type w = static_cast<element_type>(0.0);
            if (use_dual_dot) {
                compute_dual_dot_scores_device(d_s, d_cost_buff, d_cost, matrix_size, update_blocks,
                                               d_dot_partial_old, d_dot_partial_new, d_dot_pair_out,
                                               w_old, w);
            } else {
                CHECK_CUBLAS(cuda_cublas_dot(blas_handle.handle, static_cast<int>(matrix_size), d_s, 1, d_cost_buff, 1, &w_old));
                CHECK_CUBLAS(cuda_cublas_dot(blas_handle.handle, static_cast<int>(matrix_size), d_s, 1, d_cost, 1, &w));
            }
            if (w <= w_old) {
                CHECK_CUDA(cudaMemcpy(d_cost_buff, d_cost, matrix_size * sizeof(element_type), cudaMemcpyDeviceToDevice));
            } else {
                CHECK_CUDA(cudaMemcpy(d_cost, d_cost_buff, matrix_size * sizeof(element_type), cudaMemcpyDeviceToDevice));
            }
        }

        CHECK_CUDA(cudaMemcpy(d_buffer2, d_s, matrix_size * sizeof(element_type), cudaMemcpyDeviceToDevice));
        apply_log_penalty_kernel<<<blocks_matrix, kBlockSize>>>(d_s, d_cost, matrix_size, lambda_p);
        CHECK_CUDA(cudaGetLastError());
#ifdef DEBUG
        const auto debug_penalty_end = synced_now();
        cost_compute_time += elapsed_ms(debug_compute_end, debug_penalty_end);
#endif

        run_sinkhorn_iterations_cuda_device(d_cost, n1, n2, inner_iters, lambda_total, d_a_row, d_b_col, d_buffer,
                                            blas_handle.handle, use_tuned_minaa);
#ifdef DEBUG
        const auto debug_sinkhorn_end = synced_now();
        inner_loop += elapsed_ms(debug_penalty_end, debug_sinkhorn_end);
#endif
        const element_type diff_s = update_s_device(d_s, d_buffer2, d_cost, d_a_row, d_b_col, n1, n2, lambda_total,
                                                    d_update_block_sums, blas_handle.handle);
#ifdef DEBUG
        const auto debug_update_end = synced_now();
        update_s_time += elapsed_ms(debug_sinkhorn_end, debug_update_end);
#endif
        DEBUG_ONLY(printf("inner iter %d and diff of S %.6f\n\n", it, diff_s));
    }
#ifdef DEBUG
    const auto debug_copy_start = synced_now();
#endif
    CHECK_CUDA(cudaMemcpy(S.data(), d_s, matrix_size * sizeof(element_type), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    const auto debug_copy_end = synced_now();
    const auto debug_cleanup_start = debug_copy_end;
#endif
    const bool benchmark_graph1_section = compute_cost_context.benchmark_graph1_section;
    const bool graph1_transpose_view_mode = compute_cost_context.use_graph1_transpose_view;
    const size_t graph1_timing_calls = compute_cost_context.graph1_timing_calls;
    const double graph1_convert_ms_total = compute_cost_context.graph1_convert_ms_total;
    const double graph1_spmm2_ms_total = compute_cost_context.graph1_spmm2_ms_total;
    const double graph1_spmm4_ms_total = compute_cost_context.graph1_spmm4_ms_total;

    destroy_compute_cost_context(compute_cost_context);
    CHECK_CUDA(CUDA_FREE(d_pool));
    destroy_cuda_csr(d_row_norm_adj1_tr);

#ifdef DEBUG
    const auto debug_cleanup_end = synced_now();
    DEBUG_ONLY(std::cout << "optimization breakdown:\n"
                         << "  init/setup: " << elapsed_ms(debug_time_0, debug_time_1) << " ms\n"
                         << "  fixed cost prep: " << elapsed_ms(debug_time_1, debug_time_2) << " ms\n"
                         << "  compute-cost setup: " << elapsed_ms(debug_context_start, debug_context_end) << " ms\n"
                         << "  compute cost loop: " << c_compute_time << " ms\n"
                         << "  cost select + penalty: " << cost_compute_time << " ms\n"
                         << "  sinkhorn: " << inner_loop << " ms\n"
                         << "  update S: " << update_s_time << " ms\n"
                         << "  final copy: " << elapsed_ms(debug_copy_start, debug_copy_end) << " ms\n"
                         << "  cleanup: " << elapsed_ms(debug_cleanup_start, debug_cleanup_end) << " ms\n"
                         << "  total inside optimization: " << elapsed_ms(debug_time_0, debug_cleanup_end) << " ms"
                         << std::endl);
    const size_t context_bytes_used = (mem_free_after_context <= mem_free_before_context)
                                          ? (mem_free_before_context - mem_free_after_context)
                                          : 0;
    DEBUG_ONLY(std::cout << "compute-cost context memory delta: "
                         << context_bytes_used << " bytes ("
                         << (static_cast<double>(context_bytes_used) / (1024.0 * 1024.0)) << " MiB)"
                         << std::endl);
    if (benchmark_graph1_section && graph1_timing_calls > 0) {
        const double inv_calls = 1.0 / static_cast<double>(graph1_timing_calls);
        std::cout << "graph1 section timing ("
                  << (graph1_transpose_view_mode ? "transpose_view" : "baseline")
                  << "):\n"
                  << "  calls: " << graph1_timing_calls << "\n"
                  << "  col->row convert total: " << graph1_convert_ms_total << " ms"
                  << " (avg " << (graph1_convert_ms_total * inv_calls) << " ms)\n"
                  << "  spmm2 total: " << graph1_spmm2_ms_total << " ms"
                  << " (avg " << (graph1_spmm2_ms_total * inv_calls) << " ms)\n"
                  << "  spmm4 total: " << graph1_spmm4_ms_total << " ms"
                  << " (avg " << (graph1_spmm4_ms_total * inv_calls) << " ms)"
                  << std::endl;
    }
    std::cout << "done con_prox_pt_opt_init" << std::endl;
#endif
    return S;
}

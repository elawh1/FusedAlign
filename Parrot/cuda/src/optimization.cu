#include "cuda_helper.cuh"

namespace {

constexpr int kBlockSize = 256;

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

std::vector<double> init_Hlog_with_anchors(int nrows,
                                           int ncols,
                                           const std::pair<std::vector<int>, std::vector<int>>& anchors) {
    if (anchors.first.size() != anchors.second.size()) {
        throw std::invalid_argument("anchor size mismatch inside init_Hlog_with_anchors");
    }
    std::vector<double> H_log(static_cast<size_t>(nrows) * static_cast<size_t>(ncols), 0.0);
    const double anchor_value = std::log(1e-6);
    for (size_t t = 0; t < anchors.first.size(); ++t) {
        const int row = anchors.first[t];
        const int col = anchors.second[t];
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
            throw std::out_of_range("anchor out of range inside init_Hlog_with_anchors");
        }
        H_log[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(nrows)] = anchor_value;
    }
    return H_log;
}

namespace {

__global__ void col_major_to_row_major_kernel(const double* in,
                                              int rows,
                                              int cols,
                                              double* out) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (idx >= total) {
        return;
    }

    const int row = static_cast<int>(idx % static_cast<size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<size_t>(rows));
    out[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)] = in[idx];
}

__global__ void fuse_cost_terms_row_major_buffers_kernel(const double* buffer2_row_major,
                                                         const double* buffer_row_major,
                                                         const double* c_fixed_col_major,
                                                         double lambda_n,
                                                         int n1,
                                                         int n2,
                                                         double* out_col_major) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (idx >= total) {
        return;
    }

    const int row = static_cast<int>(idx % static_cast<size_t>(n1));
    const int col = static_cast<int>(idx / static_cast<size_t>(n1));
    const size_t row_major_idx = static_cast<size_t>(row) * static_cast<size_t>(n2) + static_cast<size_t>(col);
    const double clamped = buffer_row_major[row_major_idx] > 1e-15 ? buffer_row_major[row_major_idx] : 1e-15;
    out_col_major[idx] = buffer2_row_major[row_major_idx] - lambda_n * log(clamped) + c_fixed_col_major[idx];
}

__global__ void update_s_kernel(double* s,
                                const double* s_old,
                                const double* cost,
                                const double* a_row,
                                const double* b_col,
                                int n1,
                                int n2,
                                double epsilon,
                                double* block_sums) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);

    double local_sum = 0.0;
    if (idx < total) {
        const int i = static_cast<int>(idx % static_cast<size_t>(n1));
        const int j = static_cast<int>(idx / static_cast<size_t>(n1));
        const double new_value = 0.95 * exp((a_row[j] + b_col[i] - cost[idx]) / epsilon)
                                 / static_cast<double>(n1 * n2)
                                 + 0.05 * s_old[idx];
        s[idx] = new_value;
        local_sum = fabs(new_value - s_old[idx]);
    }

    __shared__ double partial[kBlockSize];
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

__global__ void minaa_v3_dense_kernel(const double* cost,
                                      const double* b,
                                      int n_row,
                                      int n_col,
                                      double epsilon,
                                      double* out_vec) {
    const int col_idx = static_cast<int>(blockIdx.x);
    if (col_idx >= n_col) return;

    const double* col = cost + static_cast<size_t>(col_idx) * static_cast<size_t>(n_row);

    double local_min = INFINITY;
    for (int row = threadIdx.x; row < n_row; row += blockDim.x) {
        local_min = fmin(local_min, col[row] - b[row]);
    }

    __shared__ double shared_min[kBlockSize];
    shared_min[threadIdx.x] = local_min;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_min[threadIdx.x] = fmin(shared_min[threadIdx.x], shared_min[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const double min_value = shared_min[0];

    double local_sum = 0.0;
    for (int row = threadIdx.x; row < n_row; row += blockDim.x) {
        local_sum += exp(-((col[row] - b[row] - min_value) / epsilon));
    }

    __shared__ double shared_sum[kBlockSize];
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_vec[col_idx] = min_value - epsilon * log(shared_sum[0] / static_cast<double>(n_row));
    }
}

__global__ void csr_row_avg_square_kernel(const int* row_offsets,
                                          const double* values,
                                          int rows,
                                          double denom,
                                          double* row_avg_sqr) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    double square_sum = 0.0;
    for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
        const double value = values[idx];
        square_sum += value * value;
    }
    row_avg_sqr[row] = square_sum / denom;
}

__global__ void compute_cost_fixed_kernel(double* cross_dist,
                                          const double* row_avg_square_dist1,
                                          const double* row_avg_square_dist2,
                                          const double* h_log,
                                          int n1,
                                          int n2,
                                          double lambda_e,
                                          double lambda_a) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (idx >= total) return;

    const int i = static_cast<int>(idx % static_cast<size_t>(n1));
    const int j = static_cast<int>(idx / static_cast<size_t>(n1));
    cross_dist[idx] += lambda_e * (row_avg_square_dist2[j] + row_avg_square_dist1[i]) - lambda_a * h_log[idx];
}

__global__ void apply_log_penalty_kernel(double* s, double* cost, size_t total, double lambda_p) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const double log_s = log(fmax(s[idx], 1e-300));
    s[idx] = log_s;
    cost[idx] -= lambda_p * log_s;
}

void transpose_col_major_matrix(const double* d_in,
                                int rows,
                                int cols,
                                double* d_out,
                                cublasHandle_t handle) {
    const double alpha = 1.0;
    const double beta = 0.0;
    CHECK_CUBLAS(cublasDgeam(handle,
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

double update_s_device(double* d_s,
                       const double* d_s_old,
                       const double* d_cost,
                       const double* d_a_row,
                       const double* d_b_col,
                       int n1,
                       int n2,
                       double epsilon,
                       double* d_block_sums,
                       cublasHandle_t blas_handle) {
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const unsigned int blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    if (blocks == 0) {
        return 0.0;
    }
    update_s_kernel<<<blocks, kBlockSize>>>(d_s, d_s_old, d_cost, d_a_row, d_b_col, n1, n2, epsilon, d_block_sums);
    CHECK_CUDA(cudaGetLastError());
#ifdef DEBUG
    double diff_sum = 0.0;
    CHECK_CUBLAS(cublasDasum(blas_handle, static_cast<int>(blocks), d_block_sums, 1, &diff_sum));
    return diff_sum;
#else
    (void)blas_handle;
    return 0.0;
#endif
}

void run_sinkhorn_iterations_cuda_device(const double* d_cost,
                                         int n1,
                                         int n2,
                                         int inner_iters,
                                         double epsilon,
                                         double* d_a_row,
                                         double* d_b_col,
                                         double* d_cost_t,
                                         cublasHandle_t blas_handle) {
    if (inner_iters <= 0) {
        return;
    }

    transpose_col_major_matrix(d_cost, n1, n2, d_cost_t, blas_handle);
    for (int iter = 0; iter < inner_iters; ++iter) {
        minaa_v3_dense_kernel<<<static_cast<unsigned int>(n2), kBlockSize>>>(d_cost, d_b_col, n1, n2, epsilon, d_a_row);
        CHECK_CUDA(cudaGetLastError());
        minaa_v3_dense_kernel<<<static_cast<unsigned int>(n1), kBlockSize>>>(d_cost_t, d_a_row, n2, n1, epsilon, d_b_col);
        CHECK_CUDA(cudaGetLastError());
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
    double* d_cost_graph1_row_major = nullptr;
    void* d_workspace_1 = nullptr;
    void* d_workspace_2 = nullptr;
    void* d_workspace_3 = nullptr;
    void* d_workspace_4 = nullptr;
    double alpha_neg2_lambda_e = 0.0;
    size_t matrix_size = 0;
    int n1 = 0;
    int n2 = 0;
    ParrotSpmmAlgorithm spmm_alg = {CUSPARSE_SPMM_ALG_DEFAULT, "default"};
};

void destroy_compute_cost_context(ComputeCostContext& context) {
    if (context.d_workspace_4 != nullptr) {
        CHECK_CUDA(cudaFree(context.d_workspace_4));
        context.d_workspace_4 = nullptr;
    }
    if (context.d_workspace_3 != nullptr) {
        CHECK_CUDA(cudaFree(context.d_workspace_3));
        context.d_workspace_3 = nullptr;
    }
    if (context.d_workspace_2 != nullptr) {
        CHECK_CUDA(cudaFree(context.d_workspace_2));
        context.d_workspace_2 = nullptr;
    }
    if (context.d_workspace_1 != nullptr) {
        CHECK_CUDA(cudaFree(context.d_workspace_1));
        context.d_workspace_1 = nullptr;
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
        CHECK_CUDA(cudaFree(context.d_cost_graph1_row_major));
        context.d_cost_graph1_row_major = nullptr;
    }
    destroy_spmm_matrix_view(context.row_norm_adj1_view);
    destroy_spmm_matrix_view(context.row_norm_adj2_view);
    destroy_spmm_matrix_view(context.intra_dist1_view);
    destroy_spmm_matrix_view(context.intra_dist2_view);
    context.alpha_neg2_lambda_e = 0.0;
    context.matrix_size = 0;
    context.n1 = 0;
    context.n2 = 0;
    context.spmm_alg = {CUSPARSE_SPMM_ALG_DEFAULT, "default"};
}

void initialize_compute_cost_context(ComputeCostContext& context,
                                     cusparseHandle_t sp_handle,
                                     const CudaCsrMatrix& d_row_norm_adj1,
                                     const CudaCsrMatrix& d_row_norm_adj2,
                                     const CudaCsrMatrix& d_intra_dist1,
                                     const CudaCsrMatrix& d_intra_dist2,
                                     const double* d_s,
                                     int n1,
                                     int n2,
                                     double lambda_e,
                                     double* d_cost,
                                     double* d_buffer,
                                     double* d_buffer2) {
    destroy_compute_cost_context(context);
    context.alpha_neg2_lambda_e = -2.0 * lambda_e;
    context.matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    context.n1 = n1;
    context.n2 = n2;
    context.spmm_alg = select_parrot_spmm_algorithm("PARROT_COST_SPMM_ALG");
    DEBUG_ONLY(std::cout << "optimization cost spmm alg: " << context.spmm_alg.name << std::endl);
    DEBUG_ONLY(std::cout << "optimization cost graph1 dense order: row_major" << std::endl);
    context.intra_dist2_view = create_spmm_matrix_view(d_intra_dist2);
    context.intra_dist1_view = create_spmm_matrix_view(d_intra_dist1);
    context.row_norm_adj2_view = create_spmm_matrix_view(d_row_norm_adj2);
    context.row_norm_adj1_view = create_spmm_matrix_view(d_row_norm_adj1);

    CHECK_CUSPARSE(cusparseCreateDnMat(&context.s_row_major, n2, n1, n1, const_cast<double*>(d_s), CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&context.cost_row_major, n2, n1, n1, d_cost, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUDA(cudaMalloc(&context.d_cost_graph1_row_major, context.matrix_size * sizeof(double)));
    CHECK_CUSPARSE(cusparseCreateDnMat(&context.cost_graph1_row_major, n1, n2, n2, context.d_cost_graph1_row_major, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&context.buffer_row_major, n1, n2, n2, d_buffer, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&context.buffer2_row_major, n1, n2, n2, d_buffer2, CUDA_R_64F, CUSPARSE_ORDER_ROW));

    size_t workspace_1 = 0;
    size_t workspace_2 = 0;
    size_t workspace_3 = 0;
    size_t workspace_4 = 0;
    const double alpha_one = 1.0;
    const double beta_zero = 0.0;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.intra_dist2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, &workspace_1));
    CHECK_CUDA(cudaMalloc(&context.d_workspace_1, workspace_1));
    CHECK_CUSPARSE(cusparseSpMM_preprocess(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.intra_dist2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, context.d_workspace_1));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &context.alpha_neg2_lambda_e, context.intra_dist1_view.descr, context.cost_graph1_row_major, &beta_zero, context.buffer2_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, &workspace_2));
    CHECK_CUDA(cudaMalloc(&context.d_workspace_2, workspace_2));
    CHECK_CUSPARSE(cusparseSpMM_preprocess(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &context.alpha_neg2_lambda_e, context.intra_dist1_view.descr, context.cost_graph1_row_major, &beta_zero, context.buffer2_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, context.d_workspace_2));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.row_norm_adj2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, &workspace_3));
    CHECK_CUDA(cudaMalloc(&context.d_workspace_3, workspace_3));
    CHECK_CUSPARSE(cusparseSpMM_preprocess(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.row_norm_adj2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, context.d_workspace_3));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.row_norm_adj1_view.descr, context.cost_graph1_row_major, &beta_zero, context.buffer_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, &workspace_4));
    CHECK_CUDA(cudaMalloc(&context.d_workspace_4, workspace_4));
    CHECK_CUSPARSE(cusparseSpMM_preprocess(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, context.row_norm_adj1_view.descr, context.cost_graph1_row_major, &beta_zero, context.buffer_row_major,
                                           CUDA_R_64F, context.spmm_alg.value, context.d_workspace_4));
}

void compute_cost_fused_2_cuda_device(cusparseHandle_t sp_handle,
                                      const ComputeCostContext& context,
                                      const CudaCsrMatrix& d_row_norm_adj1,
                                      const CudaCsrMatrix& d_row_norm_adj2,
                                      const CudaCsrMatrix& d_intra_dist1,
                                      const CudaCsrMatrix& d_intra_dist2,
                                      const double* d_c_fixed,
                                      double lambda_n,
                                      double* d_cost,
                                      double* d_buffer,
                                      double* d_buffer2) {
    const double alpha_one = 1.0;
    const double beta_zero = 0.0;
    const unsigned int blocks = static_cast<unsigned int>((context.matrix_size + kBlockSize - 1) / kBlockSize);

    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_one, context.intra_dist2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                CUDA_R_64F, context.spmm_alg.value, context.d_workspace_1));

    col_major_to_row_major_kernel<<<blocks, kBlockSize>>>(d_cost, context.n1, context.n2, context.d_cost_graph1_row_major);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &context.alpha_neg2_lambda_e, context.intra_dist1_view.descr, context.cost_graph1_row_major, &beta_zero, context.buffer2_row_major,
                                CUDA_R_64F, context.spmm_alg.value, context.d_workspace_2));

    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_one, context.row_norm_adj2_view.descr, context.s_row_major, &beta_zero, context.cost_row_major,
                                CUDA_R_64F, context.spmm_alg.value, context.d_workspace_3));

    col_major_to_row_major_kernel<<<blocks, kBlockSize>>>(d_cost, context.n1, context.n2, context.d_cost_graph1_row_major);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_one, context.row_norm_adj1_view.descr, context.cost_graph1_row_major, &beta_zero, context.buffer_row_major,
                                CUDA_R_64F, context.spmm_alg.value, context.d_workspace_4));
    fuse_cost_terms_row_major_buffers_kernel<<<blocks, kBlockSize>>>(
        d_buffer2, d_buffer, d_c_fixed, lambda_n, context.n1, context.n2, d_cost);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace


std::vector<double> con_prox_pt_opt_device(const CudaCsrMatrix& row_norm_adj1,
                                           const CudaCsrMatrix& row_norm_adj2,
                                           int n1,
                                           int n2,
                                           int avg_nnz1_per_n1,
                                           int avg_nnz2_per_n2,
                                           const double* d_cross_dist,
                                           const CudaCsrMatrix& intra_dist1_sparse,
                                           const CudaCsrMatrix& intra_dist2_sparse,
                                           int inner_iters,
                                           int outer_iters,
                                           double lambda_n,
                                           double lambda_a,
                                           double lambda_p,
                                           double lambda_e_in,
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
    const double lambda_total = lambda_n + lambda_a + lambda_p;
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const double lambda_e = lambda_e_in * matrix_size;
    std::vector<double> S(matrix_size, 1.0 / static_cast<double>(matrix_size));
    std::vector<double> H_log = init_Hlog_with_anchors(n1, n2, anchor_pairs);
#ifdef DEBUG
    const auto debug_time_1 = synced_now();
#endif

    CusparseHandle sp_handle;
    CublasHandle blas_handle;
    CudaCsrMatrix d_row_norm_adj1_tr = transpose_cuda_csr(row_norm_adj1);

    double* d_s = nullptr;
    double* d_s_old = nullptr;
    double* d_cost = nullptr;
    double* d_cost_buff = nullptr;
    double* d_c_fixed = nullptr;
    double* d_a_row = nullptr;
    double* d_b_col = nullptr;
    double* d_buffer = nullptr;
    double* d_buffer2 = nullptr;
    double* d_cost_t = nullptr;
    double* d_h_log = nullptr;
    double* d_row_avg_square_dist1 = nullptr;
    double* d_row_avg_square_dist2 = nullptr;
    double* d_update_block_sums = nullptr;

    CHECK_CUDA(cudaMalloc(&d_s, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_s_old, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cost, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cost_buff, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_c_fixed, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_a_row, static_cast<size_t>(n2) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b_col, static_cast<size_t>(n1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_buffer, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_buffer2, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cost_t, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_h_log, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_avg_square_dist1, static_cast<size_t>(n1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_avg_square_dist2, static_cast<size_t>(n2) * sizeof(double)));
    const unsigned int update_blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    if (update_blocks > 0) {
        CHECK_CUDA(cudaMalloc(&d_update_block_sums, static_cast<size_t>(update_blocks) * sizeof(double)));
    }

    CHECK_CUDA(cudaMemcpy(d_s, S.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c_fixed, d_cross_dist, matrix_size * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_h_log, H_log.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_a_row, 0, static_cast<size_t>(n2) * sizeof(double)));
    std::vector<double> b_col(static_cast<size_t>(n1), 1.0 / static_cast<double>(n2));
    CHECK_CUDA(cudaMemcpy(d_b_col, b_col.data(), static_cast<size_t>(n1) * sizeof(double), cudaMemcpyHostToDevice));
    H_log.clear();
    H_log.shrink_to_fit();

    const unsigned int blocks_row_1 = static_cast<unsigned int>((n1 + kBlockSize - 1) / kBlockSize);
    const unsigned int blocks_row_2 = static_cast<unsigned int>((n2 + kBlockSize - 1) / kBlockSize);
    csr_row_avg_square_kernel<<<blocks_row_1, kBlockSize>>>(intra_dist1_sparse.d_row_offsets, intra_dist1_sparse.d_vals, n1, static_cast<double>(n1), d_row_avg_square_dist1);
    csr_row_avg_square_kernel<<<blocks_row_2, kBlockSize>>>(intra_dist2_sparse.d_row_offsets, intra_dist2_sparse.d_vals, n2, static_cast<double>(n2), d_row_avg_square_dist2);
    CHECK_CUDA(cudaGetLastError());
    const unsigned int blocks_matrix = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    compute_cost_fixed_kernel<<<blocks_matrix, kBlockSize>>>(d_c_fixed, d_row_avg_square_dist1, d_row_avg_square_dist2, d_h_log, n1, n2, lambda_e, lambda_a);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_h_log));
    CHECK_CUDA(cudaFree(d_row_avg_square_dist2));
    CHECK_CUDA(cudaFree(d_row_avg_square_dist1));

#ifdef DEBUG
    const auto debug_time_2 = synced_now();
#endif

    ComputeCostContext compute_cost_context;
#ifdef DEBUG
    const auto debug_context_start = synced_now();
#endif
    initialize_compute_cost_context(compute_cost_context, sp_handle.handle, d_row_norm_adj1_tr, row_norm_adj2,
                                    intra_dist1_sparse, intra_dist2_sparse, d_s, n1, n2, lambda_e,
                                    d_cost, d_buffer, d_buffer2);
#ifdef DEBUG
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
        CHECK_CUDA(cudaMemcpy(d_s_old, d_s, matrix_size * sizeof(double), cudaMemcpyDeviceToDevice));

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
            CHECK_CUDA(cudaMemcpy(d_cost_buff, d_cost, matrix_size * sizeof(double), cudaMemcpyDeviceToDevice));
            has_cost_old = true;
        } else {
            double w_old = 0.0;
            double w = 0.0;
            CHECK_CUBLAS(cublasDdot(blas_handle.handle, static_cast<int>(matrix_size), d_s, 1, d_cost_buff, 1, &w_old));
            CHECK_CUBLAS(cublasDdot(blas_handle.handle, static_cast<int>(matrix_size), d_s, 1, d_cost, 1, &w));
            if (w <= w_old) {
                CHECK_CUDA(cudaMemcpy(d_cost_buff, d_cost, matrix_size * sizeof(double), cudaMemcpyDeviceToDevice));
            } else {
                CHECK_CUDA(cudaMemcpy(d_cost, d_cost_buff, matrix_size * sizeof(double), cudaMemcpyDeviceToDevice));
            }
        }

        apply_log_penalty_kernel<<<blocks_matrix, kBlockSize>>>(d_s, d_cost, matrix_size, lambda_p);
        CHECK_CUDA(cudaGetLastError());
#ifdef DEBUG
        const auto debug_penalty_end = synced_now();
        cost_compute_time += elapsed_ms(debug_compute_end, debug_penalty_end);
#endif

        run_sinkhorn_iterations_cuda_device(d_cost, n1, n2, inner_iters, lambda_total, d_a_row, d_b_col, d_cost_t, blas_handle.handle);
#ifdef DEBUG
        const auto debug_sinkhorn_end = synced_now();
        inner_loop += elapsed_ms(debug_penalty_end, debug_sinkhorn_end);
#endif

        const double diff_s = update_s_device(d_s, d_s_old, d_cost, d_a_row, d_b_col, n1, n2, lambda_total,
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
    CHECK_CUDA(cudaMemcpy(S.data(), d_s, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    const auto debug_copy_end = synced_now();
    const auto debug_cleanup_start = debug_copy_end;
#endif

    destroy_compute_cost_context(compute_cost_context);
    if (d_update_block_sums != nullptr) CHECK_CUDA(cudaFree(d_update_block_sums));
    CHECK_CUDA(cudaFree(d_cost_t));
    CHECK_CUDA(cudaFree(d_buffer2));
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFree(d_b_col));
    CHECK_CUDA(cudaFree(d_a_row));
    CHECK_CUDA(cudaFree(d_c_fixed));
    CHECK_CUDA(cudaFree(d_cost_buff));
    CHECK_CUDA(cudaFree(d_cost));
    CHECK_CUDA(cudaFree(d_s_old));
    CHECK_CUDA(cudaFree(d_s));
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
#endif
    DEBUG_ONLY(std::cout << "done con_prox_pt_opt_init" << std::endl);
    return S;
}

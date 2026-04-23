#include "cuda_helper.cuh"
#include "helper.h"
#include "helper2.h"

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

__global__ void anchor_cost_zero_kernel(double* matrix, MKL_INT nrows, const MKL_INT* rows, const MKL_INT* cols, MKL_INT k, double val) {
    const MKL_INT idx = static_cast<MKL_INT>(blockIdx.x) * blockDim.x + threadIdx.x;
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

struct DeviceCsrMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int* d_row_offsets = nullptr;
    int* d_col_ind = nullptr;
    double* d_vals = nullptr;
    cusparseSpMatDescr_t descr = nullptr;

    explicit DeviceCsrMatrix(sparse_matrix_t matrix) {
        sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
        MKL_INT rows_mkl = 0;
        MKL_INT cols_mkl = 0;
        MKL_INT* row_start = nullptr;
        MKL_INT* row_end = nullptr;
        MKL_INT* col_indx = nullptr;
        double* values = nullptr;
        auto status = mkl_sparse_d_export_csr(matrix,
                                              &indexing,
                                              &rows_mkl,
                                              &cols_mkl,
                                              &row_start,
                                              &row_end,
                                              &col_indx,
                                              &values);
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("mkl_sparse_d_export_csr failed in DeviceCsrMatrix");
        }

        rows = static_cast<int>(rows_mkl);
        cols = static_cast<int>(cols_mkl);
        std::vector<int> h_row_offsets(rows + 1);
        for (int i = 0; i < rows; ++i) {
            h_row_offsets[i] = static_cast<int>(row_start[i]);
        }
        h_row_offsets[rows] = static_cast<int>(row_end[rows - 1]);

        const int base = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
        if (base == 1) {
            for (int i = 0; i <= rows; ++i) {
                h_row_offsets[i]--;
            }
        }

        nnz = h_row_offsets[rows];
        std::vector<int> h_col_ind(nnz);
        std::vector<double> h_vals(nnz);
        for (int i = 0; i < nnz; ++i) {
            h_col_ind[i] = static_cast<int>(col_indx[i]) - base;
            h_vals[i] = values[i];
        }

        CHECK_CUDA(cudaMalloc(&d_row_offsets, (rows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_vals, nnz * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(d_row_offsets, h_row_offsets.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

        CHECK_CUSPARSE(cusparseCreateCsr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         d_row_offsets,
                                         d_col_ind,
                                         d_vals,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_64F));
    }

    ~DeviceCsrMatrix() {
        if (descr != nullptr) {
            cusparseDestroySpMat(descr);
        }
        if (d_vals != nullptr) {
            cudaFree(d_vals);
        }
        if (d_col_ind != nullptr) {
            cudaFree(d_col_ind);
        }
        if (d_row_offsets != nullptr) {
            cudaFree(d_row_offsets);
        }
    }
};

__global__ void log_floor_inplace_kernel(double* data, size_t total, double floor_value) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        const double clamped = data[idx] > floor_value ? data[idx] : floor_value;
        data[idx] = log(clamped);
    }
}

__global__ void fuse_cost_terms_kernel(const double* buffer2,
                                       const double* log_buffer,
                                       const double* c_fixed,
                                       double lambda_n,
                                       size_t total,
                                       double* out) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = buffer2[idx] - lambda_n * log_buffer[idx] + c_fixed[idx];
    }
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
                       double epsilon) {
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const unsigned int blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    double* d_block_sums = nullptr;
    std::vector<double> h_block_sums(blocks, 0.0);

    CHECK_CUDA(cudaMalloc(&d_block_sums, blocks * sizeof(double)));
    update_s_kernel<<<blocks, kBlockSize>>>(d_s, d_s_old, d_cost, d_a_row, d_b_col, n1, n2, epsilon, d_block_sums);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_block_sums.data(), d_block_sums, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_block_sums));

    return std::accumulate(h_block_sums.begin(), h_block_sums.end(), 0.0);
}

void run_sinkhorn_iterations_cuda(const std::vector<double>& cost,
                                  int n1,
                                  int n2,
                                  int inner_iters,
                                  double epsilon,
                                  std::vector<double>& a_row,
                                  std::vector<double>& b_col) {
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (cost.size() != matrix_size || a_row.size() != static_cast<size_t>(n2) || b_col.size() != static_cast<size_t>(n1)) {
        throw std::runtime_error("run_sinkhorn_iterations_cuda size mismatch");
    }
    if (inner_iters <= 0 || matrix_size == 0) {
        return;
    }

    double* d_cost = nullptr;
    double* d_cost_t = nullptr;
    double* d_a_row = nullptr;
    double* d_b_col = nullptr;

    CHECK_CUDA(cudaMalloc(&d_cost, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cost_t, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_a_row, static_cast<size_t>(n2) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b_col, static_cast<size_t>(n1) * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_cost, cost.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_col, b_col.data(), static_cast<size_t>(n1) * sizeof(double), cudaMemcpyHostToDevice));

    CublasHandle blas_handle;
    transpose_col_major_matrix(d_cost, n1, n2, d_cost_t, blas_handle.handle);

    for (int iter = 0; iter < inner_iters; ++iter) {
        minaa_v3_dense_kernel<<<static_cast<unsigned int>(n2), kBlockSize>>>(d_cost, d_b_col, n1, n2, epsilon, d_a_row);
        CHECK_CUDA(cudaGetLastError());

        minaa_v3_dense_kernel<<<static_cast<unsigned int>(n1), kBlockSize>>>(d_cost_t, d_a_row, n2, n1, epsilon, d_b_col);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaMemcpy(a_row.data(), d_a_row, static_cast<size_t>(n2) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(b_col.data(), d_b_col, static_cast<size_t>(n1) * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_b_col));
    CHECK_CUDA(cudaFree(d_a_row));
    CHECK_CUDA(cudaFree(d_cost_t));
    CHECK_CUDA(cudaFree(d_cost));
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

void compute_cost_fused_2_cuda_device(cusparseHandle_t sp_handle,
                                      DeviceCsrMatrix& d_row_norm_adj1,
                                      DeviceCsrMatrix& d_row_norm_adj2,
                                      DeviceCsrMatrix& d_intra_dist1,
                                      DeviceCsrMatrix& d_intra_dist2,
                                      const double* d_s,
                                      int n1,
                                      int n2,
                                      const double* d_c_fixed,
                                      double lambda_e,
                                      double lambda_n,
                                      double* d_cost,
                                      double* d_buffer,
                                      double* d_buffer2) {
    cusparseDnMatDescr_t s_row_major = nullptr;
    cusparseDnMatDescr_t cost_row_major = nullptr;
    cusparseDnMatDescr_t cost_col_major = nullptr;
    cusparseDnMatDescr_t buffer_col_major = nullptr;
    cusparseDnMatDescr_t buffer2_col_major = nullptr;

    CHECK_CUSPARSE(cusparseCreateDnMat(&s_row_major, n2, n1, n1, const_cast<double*>(d_s), CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&cost_row_major, n2, n1, n1, d_cost, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&cost_col_major, n1, n2, n1, d_cost, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&buffer_col_major, n1, n2, n1, d_buffer, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&buffer2_col_major, n1, n2, n1, d_buffer2, CUDA_R_64F, CUSPARSE_ORDER_COL));

    size_t workspace_1 = 0, workspace_2 = 0, workspace_3 = 0, workspace_4 = 0;
    void* d_workspace_1 = nullptr;
    void* d_workspace_2 = nullptr;
    void* d_workspace_3 = nullptr;
    void* d_workspace_4 = nullptr;

    const double alpha_one = 1.0;
    const double beta_zero = 0.0;
    const double alpha_neg2_lambda_e = -2.0 * lambda_e;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, d_intra_dist2.descr, s_row_major, &beta_zero, cost_row_major,
                                           CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_1));
    CHECK_CUDA(cudaMalloc(&d_workspace_1, workspace_1));
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_one, d_intra_dist2.descr, s_row_major, &beta_zero, cost_row_major,
                                CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_workspace_1));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_neg2_lambda_e, d_intra_dist1.descr, cost_col_major, &beta_zero, buffer2_col_major,
                                           CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_2));
    CHECK_CUDA(cudaMalloc(&d_workspace_2, workspace_2));
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_neg2_lambda_e, d_intra_dist1.descr, cost_col_major, &beta_zero, buffer2_col_major,
                                CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_workspace_2));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, d_row_norm_adj2.descr, s_row_major, &beta_zero, cost_row_major,
                                           CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_3));
    CHECK_CUDA(cudaMalloc(&d_workspace_3, workspace_3));
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_one, d_row_norm_adj2.descr, s_row_major, &beta_zero, cost_row_major,
                                CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_workspace_3));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_one, d_row_norm_adj1.descr, cost_col_major, &beta_zero, buffer_col_major,
                                           CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_4));
    CHECK_CUDA(cudaMalloc(&d_workspace_4, workspace_4));
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_one, d_row_norm_adj1.descr, cost_col_major, &beta_zero, buffer_col_major,
                                CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_workspace_4));

    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const unsigned int blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    log_floor_inplace_kernel<<<blocks, kBlockSize>>>(d_buffer, matrix_size, 1e-15);
    CHECK_CUDA(cudaGetLastError());
    fuse_cost_terms_kernel<<<blocks, kBlockSize>>>(d_buffer2, d_buffer, d_c_fixed, lambda_n, matrix_size, d_cost);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_workspace_4));
    CHECK_CUDA(cudaFree(d_workspace_3));
    CHECK_CUDA(cudaFree(d_workspace_2));
    CHECK_CUDA(cudaFree(d_workspace_1));
    CHECK_CUSPARSE(cusparseDestroyDnMat(buffer2_col_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(buffer_col_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(cost_col_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(cost_row_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(s_row_major));
}

} // namespace

void l2_norm_inplace(std::vector<double>& rwr, size_t n, size_t d) {
    if (rwr.size() != n * d) {
        throw std::runtime_error("l2_norm_inplace size mismatch");
    }
    if (rwr.empty()) {
        return;
    }

    double* d_rwr = nullptr;
    double* d_inv_norms = nullptr;
    CHECK_CUDA(cudaMalloc(&d_rwr, rwr.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_inv_norms, n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_rwr, rwr.data(), rwr.size() * sizeof(double), cudaMemcpyHostToDevice));

    compute_row_inv_l2_norms_kernel<<<static_cast<unsigned int>(n), kBlockSize>>>(d_rwr, n, d, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());

    const size_t total = rwr.size();
    const unsigned int blocks = static_cast<unsigned int>((total + kBlockSize - 1) / kBlockSize);
    scale_rows_kernel<<<blocks, kBlockSize>>>(d_rwr, n, total, d_inv_norms);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(rwr.data(), d_rwr, rwr.size() * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_inv_norms));
    CHECK_CUDA(cudaFree(d_rwr));
}

void get_normalized_neg_exp_dist(std::vector<double> rwr1, std::vector<double> rwr2, int n1, int n2, int l, std::vector<double>& cross_rwr_dist) {
    const size_t lhs_size = static_cast<size_t>(n1) * static_cast<size_t>(l);
    const size_t rhs_size = static_cast<size_t>(n2) * static_cast<size_t>(l);
    const size_t out_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (rwr1.size() != lhs_size || rwr2.size() != rhs_size) {
        throw std::runtime_error("get_normalized_neg_exp_dist input size mismatch");
    }

    cross_rwr_dist.resize(out_size);
    if (out_size == 0) {
        return;
    }

    double* d_rwr1 = nullptr;
    double* d_rwr2 = nullptr;
    double* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_rwr1, lhs_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_rwr2, rhs_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, out_size * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_rwr1, rwr1.data(), lhs_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rwr2, rwr2.data(), rhs_size * sizeof(double), cudaMemcpyHostToDevice));

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
                             d_out,
                             n1));

    const unsigned int blocks = static_cast<unsigned int>((out_size + kBlockSize - 1) / kBlockSize);
    exp_inplace_kernel<<<blocks, kBlockSize>>>(d_out, out_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(cross_rwr_dist.data(), d_out, out_size * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_rwr2));
    CHECK_CUDA(cudaFree(d_rwr1));
}

void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors, double val) {
    if (A.size() != static_cast<size_t>(nrows) * static_cast<size_t>(ncols)) {
        throw std::runtime_error("anchor_cost_zero matrix size mismatch");
    }

    std::vector<MKL_INT> valid_rows;
    std::vector<MKL_INT> valid_cols;
    valid_rows.reserve(anchors.first.size());
    valid_cols.reserve(anchors.second.size());
    for (size_t t = 0; t < anchors.first.size(); ++t) {
        const MKL_INT i = anchors.first[t];
        const MKL_INT j = anchors.second[t];
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

    double* d_matrix = nullptr;
    MKL_INT* d_rows = nullptr;
    MKL_INT* d_cols = nullptr;
    const size_t total = A.size() * sizeof(double);
    const size_t anchor_bytes = valid_rows.size() * sizeof(MKL_INT);
    CHECK_CUDA(cudaMalloc(&d_matrix, total));
    CHECK_CUDA(cudaMalloc(&d_rows, anchor_bytes));
    CHECK_CUDA(cudaMalloc(&d_cols, anchor_bytes));
    CHECK_CUDA(cudaMemcpy(d_matrix, A.data(), total, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rows, valid_rows.data(), anchor_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, valid_cols.data(), anchor_bytes, cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((valid_rows.size() + kBlockSize - 1) / kBlockSize);
    anchor_cost_zero_kernel<<<blocks, kBlockSize>>>(d_matrix, nrows, d_rows, d_cols, static_cast<MKL_INT>(valid_rows.size()), val);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(A.data(), d_matrix, total, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rows));
    CHECK_CUDA(cudaFree(d_matrix));
}

void compute_Cost_fixed(sparse_matrix_t& intra_dist1,
                        sparse_matrix_t& intra_dist2,
                        std::vector<double>& cross_dist,
                        const std::vector<double>& H_log,
                        int n1,
                        int n2,
                        double lambda_e,
                        double lambda_a) {
    const auto time_0 = std::chrono::high_resolution_clock::now();
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (cross_dist.size() != matrix_size || H_log.size() != matrix_size) {
        throw std::runtime_error("compute_Cost_fixed size mismatch");
    }

    DeviceCsrMatrix d_intra_dist1(intra_dist1);
    DeviceCsrMatrix d_intra_dist2(intra_dist2);
    double* d_row_avg_square_dist1 = nullptr;
    double* d_row_avg_square_dist2 = nullptr;
    double* d_cross_dist = nullptr;
    double* d_h_log = nullptr;

    CHECK_CUDA(cudaMalloc(&d_row_avg_square_dist1, static_cast<size_t>(n1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_avg_square_dist2, static_cast<size_t>(n2) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cross_dist, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_h_log, matrix_size * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_cross_dist, cross_dist.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h_log, H_log.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int blocks_row_1 = static_cast<unsigned int>((n1 + kBlockSize - 1) / kBlockSize);
    const unsigned int blocks_row_2 = static_cast<unsigned int>((n2 + kBlockSize - 1) / kBlockSize);
    csr_row_avg_square_kernel<<<blocks_row_1, kBlockSize>>>(d_intra_dist1.d_row_offsets, d_intra_dist1.d_vals, n1, static_cast<double>(n1), d_row_avg_square_dist1);
    csr_row_avg_square_kernel<<<blocks_row_2, kBlockSize>>>(d_intra_dist2.d_row_offsets, d_intra_dist2.d_vals, n2, static_cast<double>(n2), d_row_avg_square_dist2);
    CHECK_CUDA(cudaGetLastError());

    const auto time_1 = std::chrono::high_resolution_clock::now();
    const unsigned int blocks_matrix = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    compute_cost_fixed_kernel<<<blocks_matrix, kBlockSize>>>(d_cross_dist,
                                                             d_row_avg_square_dist1,
                                                             d_row_avg_square_dist2,
                                                             d_h_log,
                                                             n1,
                                                             n2,
                                                             lambda_e,
                                                             lambda_a);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(cross_dist.data(), d_cross_dist, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_h_log));
    CHECK_CUDA(cudaFree(d_cross_dist));
    CHECK_CUDA(cudaFree(d_row_avg_square_dist2));
    CHECK_CUDA(cudaFree(d_row_avg_square_dist1));

    const auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "in compute_Cost_fixed before loop  : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count()
              << " after loop  : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count()
              << std::endl;
}

void compute_cost_fused_2(sparse_matrix_t row_norm_adj1,
                          sparse_matrix_t row_norm_adj2,
                          const std::vector<double>& S,
                          int n1,
                          int n2,
                          sparse_matrix_t intra_dist1,
                          sparse_matrix_t intra_dist2,
                          const std::vector<double>& C_fixed,
                          double lambda_e,
                          double lambda_n,
                          std::vector<double>& C) {
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (S.size() != matrix_size || C_fixed.size() != matrix_size) {
        throw std::runtime_error("compute_cost_fused_2 size mismatch");
    }

    C.resize(matrix_size);
    if (matrix_size == 0) {
        return;
    }

    auto time_0 = std::chrono::high_resolution_clock::now();

    CusparseHandle sp_handle;
    DeviceCsrMatrix d_intra_dist1(intra_dist1);
    DeviceCsrMatrix d_intra_dist2(intra_dist2);
    DeviceCsrMatrix d_row_norm_adj1(row_norm_adj1);
    DeviceCsrMatrix d_row_norm_adj2(row_norm_adj2);

    double* d_S = nullptr;
    double* d_C_fixed = nullptr;
    double* d_C = nullptr;
    double* d_buffer = nullptr;
    double* d_buffer2 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_S, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C_fixed, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_buffer, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_buffer2, matrix_size * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_S, S.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C_fixed, C_fixed.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));

    cusparseDnMatDescr_t s_row_major = nullptr;
    cusparseDnMatDescr_t c_row_major = nullptr;
    cusparseDnMatDescr_t c_col_major = nullptr;
    cusparseDnMatDescr_t buffer_row_major = nullptr;
    cusparseDnMatDescr_t buffer_col_major = nullptr;
    cusparseDnMatDescr_t buffer2_col_major = nullptr;

    CHECK_CUSPARSE(cusparseCreateDnMat(&s_row_major, n2, n1, n1, d_S, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&c_row_major, n2, n1, n1, d_C, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&c_col_major, n1, n2, n1, d_C, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&buffer_row_major, n2, n1, n1, d_buffer, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&buffer_col_major, n1, n2, n1, d_buffer, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&buffer2_col_major, n1, n2, n1, d_buffer2, CUDA_R_64F, CUSPARSE_ORDER_COL));

    size_t workspace_intra_1 = 0;
    size_t workspace_intra_2 = 0;
    size_t workspace_rw_1 = 0;
    size_t workspace_rw_2 = 0;
    void* d_workspace_intra_1 = nullptr;
    void* d_workspace_intra_2 = nullptr;
    void* d_workspace_rw_1 = nullptr;
    void* d_workspace_rw_2 = nullptr;

    {
        const double alpha = 1.0;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle.handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               d_intra_dist2.descr,
                                               s_row_major,
                                               &beta,
                                               c_row_major,
                                               CUDA_R_64F,
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &workspace_intra_1));
        CHECK_CUDA(cudaMalloc(&d_workspace_intra_1, workspace_intra_1));
    }

    {
        const double alpha = -2.0 * lambda_e;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle.handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               d_intra_dist1.descr,
                                               c_col_major,
                                               &beta,
                                               buffer2_col_major,
                                               CUDA_R_64F,
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &workspace_intra_2));
        CHECK_CUDA(cudaMalloc(&d_workspace_intra_2, workspace_intra_2));
    }

    {
        const double alpha = 1.0;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle.handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               d_row_norm_adj2.descr,
                                               s_row_major,
                                               &beta,
                                               c_row_major,
                                               CUDA_R_64F,
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &workspace_rw_1));
        CHECK_CUDA(cudaMalloc(&d_workspace_rw_1, workspace_rw_1));
    }

    {
        const double alpha = 1.0;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle.handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               d_row_norm_adj1.descr,
                                               c_col_major,
                                               &beta,
                                               buffer_col_major,
                                               CUDA_R_64F,
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &workspace_rw_2));
        CHECK_CUDA(cudaMalloc(&d_workspace_rw_2, workspace_rw_2));
    }

    const auto time_1 = std::chrono::high_resolution_clock::now();

    {
        const double alpha = 1.0;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM(sp_handle.handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    d_intra_dist2.descr,
                                    s_row_major,
                                    &beta,
                                    c_row_major,
                                    CUDA_R_64F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_workspace_intra_1));
    }

    {
        const double alpha = -2.0 * lambda_e;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM(sp_handle.handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    d_intra_dist1.descr,
                                    c_col_major,
                                    &beta,
                                    buffer2_col_major,
                                    CUDA_R_64F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_workspace_intra_2));
    }

    const auto time_2 = std::chrono::high_resolution_clock::now();

    {
        const double alpha = 1.0;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM(sp_handle.handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    d_row_norm_adj2.descr,
                                    s_row_major,
                                    &beta,
                                    c_row_major,
                                    CUDA_R_64F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_workspace_rw_1));
    }

    {
        const double alpha = 1.0;
        const double beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM(sp_handle.handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    d_row_norm_adj1.descr,
                                    c_col_major,
                                    &beta,
                                    buffer_col_major,
                                    CUDA_R_64F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_workspace_rw_2));
    }

    const unsigned int blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    log_floor_inplace_kernel<<<blocks, kBlockSize>>>(d_buffer, matrix_size, 1e-15);
    CHECK_CUDA(cudaGetLastError());
    const auto time_3 = std::chrono::high_resolution_clock::now();

    fuse_cost_terms_kernel<<<blocks, kBlockSize>>>(d_buffer2, d_buffer, d_C_fixed, lambda_n, matrix_size, d_C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(C.data(), d_C, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));
    const auto time_4 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaFree(d_workspace_rw_2));
    CHECK_CUDA(cudaFree(d_workspace_rw_1));
    CHECK_CUDA(cudaFree(d_workspace_intra_2));
    CHECK_CUDA(cudaFree(d_workspace_intra_1));

    CHECK_CUSPARSE(cusparseDestroyDnMat(buffer2_col_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(buffer_col_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(buffer_row_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(c_col_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(c_row_major));
    CHECK_CUSPARSE(cusparseDestroyDnMat(s_row_major));

    CHECK_CUDA(cudaFree(d_buffer2));
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_fixed));
    CHECK_CUDA(cudaFree(d_S));

    std::cout << "in compute_cost_fused_2 first 2spmm : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count()
              << " second 2spmm : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_3 - time_2).count()
              << " matrix iter : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_4 - time_3).count()
              << " ms" << std::endl;
}

double update_S_mat(std::vector<double>& S,
                    const std::vector<double>& S_old,
                    const std::vector<double>& Cost,
                    const std::vector<double>& a_row,
                    const std::vector<double>& b_col,
                    int n1,
                    int n2,
                    double epsilon) {
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (S.size() != matrix_size || S_old.size() != matrix_size || Cost.size() != matrix_size ||
        a_row.size() != static_cast<size_t>(n2) || b_col.size() != static_cast<size_t>(n1)) {
        throw std::runtime_error("update_S_mat size mismatch");
    }
    if (matrix_size == 0) {
        return 0.0;
    }

    double* d_s = nullptr;
    double* d_s_old = nullptr;
    double* d_cost = nullptr;
    double* d_a_row = nullptr;
    double* d_b_col = nullptr;
    double* d_block_sums = nullptr;

    const unsigned int blocks = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    std::vector<double> h_block_sums(blocks, 0.0);

    CHECK_CUDA(cudaMalloc(&d_s, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_s_old, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cost, matrix_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_a_row, static_cast<size_t>(n2) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b_col, static_cast<size_t>(n1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_block_sums, blocks * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_s, S.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_s_old, S_old.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cost, Cost.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a_row, a_row.data(), static_cast<size_t>(n2) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_col, b_col.data(), static_cast<size_t>(n1) * sizeof(double), cudaMemcpyHostToDevice));

    update_s_kernel<<<blocks, kBlockSize>>>(d_s, d_s_old, d_cost, d_a_row, d_b_col, n1, n2, epsilon, d_block_sums);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(S.data(), d_s, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_block_sums.data(), d_block_sums, blocks * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_block_sums));
    CHECK_CUDA(cudaFree(d_b_col));
    CHECK_CUDA(cudaFree(d_a_row));
    CHECK_CUDA(cudaFree(d_cost));
    CHECK_CUDA(cudaFree(d_s_old));
    CHECK_CUDA(cudaFree(d_s));

    return std::accumulate(h_block_sums.begin(), h_block_sums.end(), 0.0);
}

void minaa_v3_dense(std::vector<double>& Cost,
                    const std::vector<double>& b,
                    int n_row,
                    int n_col,
                    double epsilon,
                    std::vector<double>& out_vec) {
    if (Cost.size() != static_cast<size_t>(n_row) * static_cast<size_t>(n_col) ||
        b.size() != static_cast<size_t>(n_row) ||
        out_vec.size() != static_cast<size_t>(n_col)) {
        throw std::runtime_error("minaa_v3_dense size mismatch");
    }
    if (Cost.empty()) {
        return;
    }

    double* d_cost = nullptr;
    double* d_b = nullptr;
    double* d_out = nullptr;

    CHECK_CUDA(cudaMalloc(&d_cost, Cost.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b, b.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, out_vec.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_cost, Cost.data(), Cost.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));

    minaa_v3_dense_kernel<<<static_cast<unsigned int>(n_col), kBlockSize>>>(d_cost, d_b, n_row, n_col, epsilon, d_out);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(out_vec.data(), d_out, out_vec.size() * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_cost));
}

std::vector<double> con_prox_pt_opt(sparse_matrix_t row_norm_adj1,
                                    sparse_matrix_t row_norm_adj2,
                                    int n1,
                                    int n2,
                                    int avg_nnz1_per_n1,
                                    int avg_nnz2_per_n2,
                                    std::vector<double>& cross_dist,
                                    sparse_matrix_t intra_dist1_sparse,
                                    sparse_matrix_t intra_dist2_sparse,
                                    int inner_iters,
                                    int outer_iters,
                                    double lambda_n,
                                    double lambda_a,
                                    double lambda_p,
                                    double lambda_e_in,
                                    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs,
                                    bool sparsify) {
    (void)avg_nnz1_per_n1;
    (void)avg_nnz2_per_n2;
    (void)sparsify;

    const auto time_0 = std::chrono::high_resolution_clock::now();
    const double lambda_total = lambda_n + lambda_a + lambda_p;
    const size_t matrix_size = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const double lambda_e = lambda_e_in * matrix_size;
    std::vector<double> S(matrix_size, 1.0 / static_cast<double>(matrix_size));
    std::vector<double> H_log = init_Hlog_with_anchors(n1, n2, anchor_pairs);
    const auto time_1 = std::chrono::high_resolution_clock::now();
    std::cout << "initialization in optimization : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count()
              << " ms" << std::endl;

    sparse_matrix_t row_norm_adj1_tr = nullptr;
    sparse_matrix_t row_norm_adj2_tr = nullptr;
    auto st = mkl_sparse_convert_csr(row_norm_adj1, SPARSE_OPERATION_TRANSPOSE, &row_norm_adj1_tr);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("row_norm_adj1 transpose failed");
    st = mkl_sparse_convert_csr(row_norm_adj2, SPARSE_OPERATION_TRANSPOSE, &row_norm_adj2_tr);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("row_norm_adj2 transpose failed");

    CusparseHandle sp_handle;
    CublasHandle blas_handle;
    DeviceCsrMatrix d_row_norm_adj1_tr(row_norm_adj1_tr);
    DeviceCsrMatrix d_row_norm_adj2(row_norm_adj2);
    DeviceCsrMatrix d_intra_dist1(intra_dist1_sparse);
    DeviceCsrMatrix d_intra_dist2(intra_dist2_sparse);

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

    CHECK_CUDA(cudaMemcpy(d_s, S.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c_fixed, cross_dist.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h_log, H_log.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_a_row, 0, static_cast<size_t>(n2) * sizeof(double)));
    std::vector<double> b_col(static_cast<size_t>(n1), 1.0 / static_cast<double>(n2));
    CHECK_CUDA(cudaMemcpy(d_b_col, b_col.data(), static_cast<size_t>(n1) * sizeof(double), cudaMemcpyHostToDevice));
    H_log.clear();
    H_log.shrink_to_fit();

    const unsigned int blocks_row_1 = static_cast<unsigned int>((n1 + kBlockSize - 1) / kBlockSize);
    const unsigned int blocks_row_2 = static_cast<unsigned int>((n2 + kBlockSize - 1) / kBlockSize);
    csr_row_avg_square_kernel<<<blocks_row_1, kBlockSize>>>(d_intra_dist1.d_row_offsets,
                                                            d_intra_dist1.d_vals,
                                                            n1,
                                                            static_cast<double>(n1),
                                                            d_row_avg_square_dist1);
    csr_row_avg_square_kernel<<<blocks_row_2, kBlockSize>>>(d_intra_dist2.d_row_offsets,
                                                            d_intra_dist2.d_vals,
                                                            n2,
                                                            static_cast<double>(n2),
                                                            d_row_avg_square_dist2);
    CHECK_CUDA(cudaGetLastError());
    const unsigned int blocks_matrix = static_cast<unsigned int>((matrix_size + kBlockSize - 1) / kBlockSize);
    compute_cost_fixed_kernel<<<blocks_matrix, kBlockSize>>>(d_c_fixed,
                                                             d_row_avg_square_dist1,
                                                             d_row_avg_square_dist2,
                                                             d_h_log,
                                                             n1,
                                                             n2,
                                                             lambda_e,
                                                             lambda_a);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_h_log));
    CHECK_CUDA(cudaFree(d_row_avg_square_dist2));
    CHECK_CUDA(cudaFree(d_row_avg_square_dist1));

    const auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "L_fixed computation in optimization : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count()
              << " ms" << std::endl << std::endl;

    bool has_cost_old = false;
    double c_compute_time = 0.0;
    double cost_compute_time = 0.0;
    double inner_loop = 0.0;
    double update_s_time = 0.0;

    for (int it = 0; it < outer_iters; ++it) {
        printf("----- outer iterration %d ----\n", it);
        CHECK_CUDA(cudaMemcpy(d_s_old, d_s, matrix_size * sizeof(double), cudaMemcpyDeviceToDevice));

        const auto time_l_done = std::chrono::high_resolution_clock::now();
        compute_cost_fused_2_cuda_device(sp_handle.handle,
                                         d_row_norm_adj1_tr,
                                         d_row_norm_adj2,
                                         d_intra_dist1,
                                         d_intra_dist2,
                                         d_s,
                                         n1,
                                         n2,
                                         d_c_fixed,
                                         lambda_e,
                                         lambda_n,
                                         d_cost,
                                         d_buffer,
                                         d_buffer2);
        const auto time_c_done = std::chrono::high_resolution_clock::now();
        c_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_c_done - time_l_done).count();

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
        const auto time_iter_1 = std::chrono::high_resolution_clock::now();
        cost_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1 - time_c_done).count();

        run_sinkhorn_iterations_cuda_device(d_cost, n1, n2, inner_iters, lambda_total, d_a_row, d_b_col, d_cost_t, blas_handle.handle);
        const auto time_iter_2 = std::chrono::high_resolution_clock::now();
        inner_loop += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_2 - time_iter_1).count();

        const double diff_s = update_s_device(d_s, d_s_old, d_cost, d_a_row, d_b_col, n1, n2, lambda_total);
        const auto time_iter_3 = std::chrono::high_resolution_clock::now();
        update_s_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_3 - time_iter_2).count();
        printf("inner iter %d and diff of S %.6f\n\n", it, diff_s);
    }

    CHECK_CUDA(cudaMemcpy(S.data(), d_s, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));

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
    mkl_sparse_destroy(row_norm_adj2_tr);
    mkl_sparse_destroy(row_norm_adj1_tr);

    const auto time_4 = std::chrono::high_resolution_clock::now();
    std::cout << "total time of 10 outer iterations : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_4 - time_2).count()
              << " ms" << std::endl;
    std::cout << " C_compute_time : " << c_compute_time
              << " Cost_compute_time : " << cost_compute_time
              << " inner loop : " << inner_loop
              << " update_S : " << update_s_time
              << " ms" << std::endl;
    std::cout << "done con_prox_pt_opt_init" << std::endl;
    return S;
}


sparse_matrix_t row_normalize_graph(sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) {
    std::vector<double> all_ones(n, 1.0);
    std::vector<double> graph_row_sum(n, 0.0);
    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                                             1.0,
                                             graph,
                                             descr,
                                             all_ones.data(),
                                             0.0,
                                             graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_mv failed in row_normalize_graph");
    }

    std::vector<MKL_INT> isolated;
    for (MKL_INT i = 0; i < n; ++i) {
        if (graph_row_sum[i] < 1e-10) {
            isolated.push_back(i);
        }
    }
    printf("ang nnz %d num of isolated %ld\n", avg_nnz_per_n, isolated.size());

    sparse_matrix_t graph_connected = nullptr;
    bool owns_graph_connected = false;
    if (!isolated.empty()) {
        printf("connecting isolated nodes\n");
        std::vector<double> edge_list;
        edge_list.reserve(isolated.size() * avg_nnz_per_n * 4);
        std::srand(std::time(0));
        for (MKL_INT i : isolated) {
            std::set<MKL_INT> random_nodes = {};
            while (random_nodes.size() < static_cast<size_t>(avg_nnz_per_n)) {
                MKL_INT random_num = std::rand() % n;
                if (random_num == i || random_nodes.find(random_num) != random_nodes.end()) continue;
                random_nodes.insert(random_num);
                edge_list.push_back(static_cast<double>(i));
                edge_list.push_back(static_cast<double>(random_num));
                edge_list.push_back(static_cast<double>(random_num));
                edge_list.push_back(static_cast<double>(i));
            }
        }

        printf("edge_list size %d \n", static_cast<int>(edge_list.size()));
        sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, n, n);
        status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, graph, 1.0, add_mat, &graph_connected);
        mkl_sparse_destroy(add_mat);
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("mkl_sparse_d_add failed in row_normalize_graph");
        }
        owns_graph_connected = true;
    } else {
        graph_connected = graph;
    }

    sparse_matrix_t graph_connected_csr = nullptr;
    status = mkl_sparse_convert_csr(graph_connected, SPARSE_OPERATION_NON_TRANSPOSE, &graph_connected_csr);
    if (status != SPARSE_STATUS_SUCCESS) {
        if (owns_graph_connected) mkl_sparse_destroy(graph_connected);
        throw std::runtime_error("mkl_sparse_convert_csr failed in row_normalize_graph");
    }

    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    MKL_INT rows = 0, cols = 0;
    MKL_INT* row_start = nullptr;
    MKL_INT* row_end = nullptr;
    MKL_INT* col_indx = nullptr;
    double* values = nullptr;
    status = mkl_sparse_d_export_csr(graph_connected_csr,
                                     &indexing,
                                     &rows,
                                     &cols,
                                     &row_start,
                                     &row_end,
                                     &col_indx,
                                     &values);
    if (status != SPARSE_STATUS_SUCCESS) {
        mkl_sparse_destroy(graph_connected_csr);
        if (owns_graph_connected) mkl_sparse_destroy(graph_connected);
        throw std::runtime_error("mkl_sparse_d_export_csr failed in row_normalize_graph");
    }

    std::vector<int> h_row_offsets(rows + 1);
    for (int i = 0; i < rows; ++i) {
        h_row_offsets[i] = static_cast<int>(row_start[i]);
    }
    h_row_offsets[rows] = static_cast<int>(row_end[rows - 1]);
    const int base = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    if (base == 1) {
        for (int i = 0; i <= rows; ++i) h_row_offsets[i]--;
    }

    const int nnz = h_row_offsets[rows];
    std::vector<int> h_col_ind(nnz);
    std::vector<double> h_vals(nnz);
    for (int i = 0; i < nnz; ++i) {
        h_col_ind[i] = static_cast<int>(col_indx[i]) - base;
        h_vals[i] = values[i];
    }

    int* d_row_offsets = nullptr;
    double* d_vals = nullptr;
    double* d_row_sums = nullptr;
    CHECK_CUDA(cudaMalloc(&d_row_offsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_vals, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_sums, rows * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_row_offsets, h_row_offsets.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((rows + kBlockSize - 1) / kBlockSize);
    csr_row_sum_kernel<<<blocks, kBlockSize>>>(d_row_offsets, d_vals, rows, d_row_sums);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(graph_row_sum.data(), d_row_sums, rows * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < static_cast<int>(graph_row_sum.size()); ++i) {
        if (graph_row_sum[i] < 1e-10) printf("111 graph_row_sum[%d] is zero \n", i);
    }

    csr_scale_rows_kernel<<<blocks, kBlockSize>>>(d_row_offsets, d_vals, d_row_sums, rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_vals.data(), d_vals, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    csr_row_sum_kernel<<<blocks, kBlockSize>>>(d_row_offsets, d_vals, rows, d_row_sums);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(graph_row_sum.data(), d_row_sums, rows * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < static_cast<int>(graph_row_sum.size()); ++i) {
        if (graph_row_sum[i] < 1e-10) printf("222 graph_row_sum[%d] is zero \n", i);
    }

    std::vector<MKL_INT> out_row_start(rows), out_row_end(rows), out_col_ind(nnz);
    for (int i = 0; i < rows; ++i) {
        out_row_start[i] = static_cast<MKL_INT>(h_row_offsets[i]);
        out_row_end[i] = static_cast<MKL_INT>(h_row_offsets[i + 1]);
    }
    for (int i = 0; i < nnz; ++i) {
        out_col_ind[i] = static_cast<MKL_INT>(h_col_ind[i]);
    }

    sparse_matrix_t temp_result = nullptr;
    status = mkl_sparse_d_create_csr(&temp_result,
                                     SPARSE_INDEX_BASE_ZERO,
                                     rows,
                                     cols,
                                     out_row_start.data(),
                                     out_row_end.data(),
                                     out_col_ind.data(),
                                     h_vals.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        CHECK_CUDA(cudaFree(d_row_sums));
        CHECK_CUDA(cudaFree(d_vals));
        CHECK_CUDA(cudaFree(d_row_offsets));
        mkl_sparse_destroy(graph_connected_csr);
        if (owns_graph_connected) mkl_sparse_destroy(graph_connected);
        throw std::runtime_error("mkl_sparse_d_create_csr failed in row_normalize_graph");
    }

    sparse_matrix_t graph_norm_csr = nullptr;
    status = mkl_sparse_copy(temp_result, descr, &graph_norm_csr);
    mkl_sparse_destroy(temp_result);
    if (status != SPARSE_STATUS_SUCCESS) {
        CHECK_CUDA(cudaFree(d_row_sums));
        CHECK_CUDA(cudaFree(d_vals));
        CHECK_CUDA(cudaFree(d_row_offsets));
        mkl_sparse_destroy(graph_connected_csr);
        if (owns_graph_connected) mkl_sparse_destroy(graph_connected);
        throw std::runtime_error("mkl_sparse_copy failed in row_normalize_graph");
    }
    status = mkl_sparse_optimize(graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) {
        CHECK_CUDA(cudaFree(d_row_sums));
        CHECK_CUDA(cudaFree(d_vals));
        CHECK_CUDA(cudaFree(d_row_offsets));
        mkl_sparse_destroy(graph_norm_csr);
        mkl_sparse_destroy(graph_connected_csr);
        if (owns_graph_connected) mkl_sparse_destroy(graph_connected);
        throw std::runtime_error("mkl_sparse_optimize failed in row_normalize_graph");
    }

    CHECK_CUDA(cudaFree(d_row_sums));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_row_offsets));
    mkl_sparse_destroy(graph_connected_csr);
    if (owns_graph_connected) {
        mkl_sparse_destroy(graph_connected);
    }

    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_norm_csr, descr, all_ones.data(), 0.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        mkl_sparse_destroy(graph_norm_csr);
        throw std::runtime_error("mkl_sparse_d_mv failed after normalization");
    }
    for (int i = 0; i < static_cast<int>(graph_row_sum.size()); ++i) {
        if (graph_row_sum[i] < 1e-10) printf("333 graph_row_sum[%d] is zero \n", i);
    }

    return graph_norm_csr;
}

#define PARROT_CUDA_NATIVE_SINKHORN_INNER_LOOP
#define PARROT_CUDA_NATIVE_MINAA_V3_DENSE
#define PARROT_CUDA_NATIVE_UPDATE_S_MAT
#define PARROT_CUDA_NATIVE_COMPUTE_COST_FIXED
#define PARROT_CUDA_NATIVE_CON_PROX_PT_OPT
#define PARROT_CUDA_NATIVE_COMPUTE_COST_FUSED_2
#define PARROT_CUDA_NATIVE_HELPERS
#include "../../src/helper.cpp"

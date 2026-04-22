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

#define PARROT_CUDA_NATIVE_HELPERS
#include "../../src/helper.cpp"

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

#define PARROT_CUDA_NATIVE_HELPERS
#include "../../src/helper.cpp"

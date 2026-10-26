#include "cuda_helper.cuh"

namespace {

__global__ void exp_neg_kernel(element_type* vals, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
#if CUDA_USE_FLOAT
        vals[idx] = __expf(-vals[idx]);
#else
        vals[idx] = exp(-vals[idx]);
#endif
    }
}

__global__ void sparse_intra_cost_kernel(const element_type* emb,
                                         const int* row_offsets,
                                         const int* col_indices,
                                         const element_type* mask_vals,
                                         int n,
                                         int L,
                                         element_type* out) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= n) return;

    for (int edge_idx = row_offsets[row] + static_cast<int>(threadIdx.x);
         edge_idx < row_offsets[row + 1];
         edge_idx += static_cast<int>(blockDim.x)) {
        const int col = col_indices[edge_idx];
        element_type dot = static_cast<element_type>(0.0);
        for (int k = 0; k < L; ++k) {
            dot += emb[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(row)]
                 * emb[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(col)];
        }
#if CUDA_USE_FLOAT
        const element_type value = __expf(-dot) * mask_vals[edge_idx];
#else
        const element_type value = static_cast<element_type>(exp(static_cast<double>(-dot))) * mask_vals[edge_idx];
#endif
        out[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(row)] = value;
    }
}

}  // namespace

CudaDenseMatrix dense_intra_cost_device(
    const element_type* d_emb,
    const CudaSparseMatrix& mask,
    int n,
    int L)
{
    CudaDenseMatrix result;
    result.rows = n;
    result.cols = n;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);

    CHECK_CUDA(CUDA_MALLOC(&result.d_vals, total * sizeof(element_type)));
    CHECK_CUDA(cudaMemset(result.d_vals, 0, total * sizeof(element_type)));
    sparse_intra_cost_kernel<<<static_cast<unsigned int>(n), 128>>>(
        d_emb, mask.d_row_offsets, mask.d_col_indices, mask.d_vals, n, L, result.d_vals);
    CHECK_CUDA(cudaGetLastError());

    return result;
}

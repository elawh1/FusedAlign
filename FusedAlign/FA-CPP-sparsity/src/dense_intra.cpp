#include "cuda_helper.cuh"

CudaDenseMatrix dense_intra_cost_device(const element_type* emb,
                                        const CudaSparseMatrix& mask,
                                        int n,
                                        int L) {
    CudaDenseMatrix result;
    result.rows = n;
    result.cols = n;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);

    CHECK_CUDA(CUDA_MALLOC(&result.d_vals, total * sizeof(element_type)));
    CHECK_CUDA(cudaMemset(result.d_vals, 0, total * sizeof(element_type)));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int row = 0; row < n; ++row) {
        for (int edge_idx = mask.d_row_offsets[row]; edge_idx < mask.d_row_offsets[row + 1]; ++edge_idx) {
            const int col = mask.d_col_indices[edge_idx];
            element_type dot = static_cast<element_type>(0);
            for (int k = 0; k < L; ++k) {
                dot += emb[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(row)]
                     * emb[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(col)];
            }
            result.d_vals[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(row)] =
                static_cast<element_type>(std::exp(static_cast<double>(-dot))) * mask.d_vals[edge_idx];
        }
    }

    return result;
}

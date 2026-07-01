#include "cuda_helper.cuh"

CudaDenseMatrix dense_intra_cost_device(const element_type* emb,
                                        const CudaDenseMatrix& mask,
                                        int n,
                                        int L) {
    CudaDenseMatrix result;
    result.rows = n;
    result.cols = n;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    CHECK_CUDA(CUDA_MALLOC(&result.d_vals, total * sizeof(element_type)));

    const element_type one = static_cast<element_type>(1);
    const element_type zero = static_cast<element_type>(0);
    CHECK_CUBLAS(cuda_cublas_gemm(0, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n, n, L,
                                  &one, emb, n,
                                  emb, n,
                                  &zero, result.d_vals, n));

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total; ++idx) {
        result.d_vals[idx] = std::exp(-result.d_vals[idx]) * mask.d_vals[idx];
    }
    return result;
}

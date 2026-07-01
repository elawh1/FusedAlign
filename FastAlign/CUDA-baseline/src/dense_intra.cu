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

__global__ void apply_mask_kernel(element_type* vals, const element_type* mask, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        vals[idx] *= mask[idx];
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

}  // namespace

CudaDenseMatrix dense_intra_cost_device(
    const element_type* d_emb,
    const CudaDenseMatrix& mask,
    int n,
    int L)
{
    CudaDenseMatrix result;
    result.rows = n;
    result.cols = n;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);

    CHECK_CUDA(CUDA_MALLOC(&result.d_vals, total * sizeof(element_type)));

    CublasHandle handle;
    const element_type alpha = static_cast<element_type>(1.0);
    const element_type beta = static_cast<element_type>(0.0);
    CHECK_CUBLAS(cuda_cublas_gemm(handle.handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_T,
                                  n,
                                  n,
                                  L,
                                  &alpha,
                                  d_emb,
                                  n,
                                  d_emb,
                                  n,
                                  &beta,
                                  result.d_vals,
                                  n));

    const int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    exp_neg_kernel<<<grid, block>>>(result.d_vals, total);
    CHECK_CUDA(cudaGetLastError());
    apply_mask_kernel<<<grid, block>>>(result.d_vals, mask.d_vals, total);
    CHECK_CUDA(cudaGetLastError());

    return result;
}

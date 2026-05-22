#include "cuda_helper.cuh"

__global__ void exp_neg_kernel(element_type* vals, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        vals[idx] = exp(-vals[idx]);
    }
}

CudaCsrMatrix masked_sddmm_intra_cost_cusparse_device(
    const element_type* d_emb,
    const CudaCsrMatrix& mask,
    int n,
    int L)
{
    const int nnz = mask.nnz;

    element_type* d_vals = nullptr;

    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_vals, sizeof(element_type) * nnz));
    CHECK_CUDA(cudaMemset(d_vals, 0, sizeof(element_type) * nnz));

    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseDnMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr;
    cusparseSpMatDescr_t matC = nullptr;

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matA, n, L, n, const_cast<element_type*>(d_emb), PARROT_CUDA_R_TYPE, CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB, n, L, n, const_cast<element_type*>(d_emb), PARROT_CUDA_R_TYPE, CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matC,
        n, n, nnz,
        mask.d_row_offsets,
        mask.d_col_ind,
        d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        PARROT_CUDA_R_TYPE));

    const element_type alpha = static_cast<element_type>(1.0);
    const element_type beta = static_cast<element_type>(0.0);

    size_t workspace_size = 0;
    void* d_workspace = nullptr;

    CHECK_CUSPARSE(cusparseSDDMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        PARROT_CUDA_R_TYPE,
        CUSPARSE_SDDMM_ALG_DEFAULT,
        &workspace_size));

    CHECK_CUDA(PARROT_CUDA_MALLOC(&d_workspace, workspace_size));

    CHECK_CUSPARSE(cusparseSDDMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        PARROT_CUDA_R_TYPE,
        CUSPARSE_SDDMM_ALG_DEFAULT,
        d_workspace));

    CHECK_CUSPARSE(cusparseSDDMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        PARROT_CUDA_R_TYPE,
        CUSPARSE_SDDMM_ALG_DEFAULT,
        d_workspace));

    {
        const int block = 256;
        const int grid = (nnz + block - 1) / block;
        exp_neg_kernel<<<grid, block>>>(d_vals, nnz);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(PARROT_CUDA_FREE(d_workspace));
    cusparseDestroySpMat(matC);
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroy(handle);

    CudaCsrMatrix result;
    result.rows = n;
    result.cols = n;
    result.nnz = nnz;
    result.d_vals = d_vals;
    CHECK_CUDA(PARROT_CUDA_MALLOC(&result.d_row_offsets, sizeof(int) * (static_cast<size_t>(n) + 1)));
    CHECK_CUDA(PARROT_CUDA_MALLOC(&result.d_col_ind, sizeof(int) * static_cast<size_t>(nnz)));
    CHECK_CUDA(cudaMemcpy(result.d_row_offsets, mask.d_row_offsets, sizeof(int) * (static_cast<size_t>(n) + 1), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(result.d_col_ind, mask.d_col_ind, sizeof(int) * static_cast<size_t>(nnz), cudaMemcpyDeviceToDevice));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &result.descr,
        result.rows,
        result.cols,
        result.nnz,
        result.d_row_offsets,
        result.d_col_ind,
        result.d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        PARROT_CUDA_R_TYPE));
    return result;
}

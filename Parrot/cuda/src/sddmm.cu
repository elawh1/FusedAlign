#include "cuda_helper.cuh"

__global__ void exp_neg_kernel(double* vals, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        vals[idx] = exp(-vals[idx]);
    }
}

void masked_sddmm_intra_cost_cusparse(
    std::vector<double>& emb,
    sparse_matrix_t mask,
    int n,
    int L,
    sparse_matrix_t& result)
{
    // -------------------------------------------------------------------------
    // 1) Export MKL CSR from mask
    // -------------------------------------------------------------------------
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start = nullptr, *row_end = nullptr, *col_ind = nullptr;
    double *mask_vals = nullptr;

    auto st = mkl_sparse_d_export_csr(
        mask, &indexing, &rows, &cols,
        &row_start, &row_end, &col_ind, &mask_vals);

    if (st != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Export CSR failed");
    }

    if (indexing != SPARSE_INDEX_BASE_ZERO) {
        throw std::runtime_error("Only zero-based CSR is supported here");
    }

    if (rows != n || cols != n) {
        throw std::runtime_error("mask dimensions do not match n");
    }

    if (emb.size() != static_cast<size_t>(n) * static_cast<size_t>(L)) {
        throw std::runtime_error("emb size mismatch");
    }

    // MKL export gives row_start[i], row_end[i]
    // Build standard CSR row_offsets of length n+1
    std::vector<int> h_row_offsets(n + 1);
    for (int i = 0; i < n; ++i) {
        h_row_offsets[i] = static_cast<int>(row_start[i]);
    }
    h_row_offsets[n] = static_cast<int>(row_end[n - 1]);

    const int nnz = h_row_offsets[n];

    std::vector<int> h_col_ind(nnz);
    for (int k = 0; k < nnz; ++k) {
        h_col_ind[k] = static_cast<int>(col_ind[k]);
    }

    // -------------------------------------------------------------------------
    // 2) Allocate/copy device data
    // -------------------------------------------------------------------------
    double* d_emb = nullptr;
    int* d_row_offsets = nullptr;
    int* d_col_ind = nullptr;
    double* d_vals = nullptr;

    CHECK_CUDA(cudaMalloc(&d_emb, sizeof(double) * emb.size()));
    CHECK_CUDA(cudaMalloc(&d_row_offsets, sizeof(int) * (n + 1)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc(&d_vals, sizeof(double) * nnz));

    CHECK_CUDA(cudaMemcpy(d_emb, emb.data(),
                          sizeof(double) * emb.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_offsets, h_row_offsets.data(),
                          sizeof(int) * (n + 1),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind.data(),
                          sizeof(int) * nnz,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_vals, 0, sizeof(double) * nnz));

    // -------------------------------------------------------------------------
    // 3) cuSPARSE descriptors
    // emb layout is emb[l*n + i], which is column-major (n x L), ld=n
    // We want masked E * E^T
    // -------------------------------------------------------------------------
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseDnMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr;
    cusparseSpMatDescr_t matC = nullptr;

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matA, n, L, n, d_emb, CUDA_R_64F, CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB, n, L, n, d_emb, CUDA_R_64F, CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matC,
        n, n, nnz,
        d_row_offsets,
        d_col_ind,
        d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    const double alpha = 1.0;
    const double beta = 0.0;

    size_t workspace_size = 0;
    void* d_workspace = nullptr;

    auto time_1 = std::chrono::high_resolution_clock::now();
    CHECK_CUSPARSE(cusparseSDDMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_64F,
        CUSPARSE_SDDMM_ALG_DEFAULT,
        &workspace_size));

    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));

    CHECK_CUSPARSE(cusparseSDDMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_64F,
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
        CUDA_R_64F,
        CUSPARSE_SDDMM_ALG_DEFAULT,
        d_workspace));

    // -------------------------------------------------------------------------
    // 4) Apply exp(-x) on sparse values
    // -------------------------------------------------------------------------
    {
        const int block = 256;
        const int grid = (nnz + block - 1) / block;
        exp_neg_kernel<<<grid, block>>>(d_vals, nnz);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "masked_sddmm_intra_cost_cuda" << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << std::endl;


    // -------------------------------------------------------------------------
    // 5) Copy values back to host and rebuild MKL sparse_matrix_t result
    // -------------------------------------------------------------------------
    std::vector<double> h_vals(nnz);
    CHECK_CUDA(cudaMemcpy(h_vals.data(), d_vals,
                          sizeof(double) * nnz,
                          cudaMemcpyDeviceToHost));

    // MKL create_csr wants row_start/row_end split, not row_offsets
    std::vector<MKL_INT> out_row_start(n), out_row_end(n), out_col_ind(nnz);
    for (int i = 0; i < n; ++i) {
        out_row_start[i] = static_cast<MKL_INT>(h_row_offsets[i]);
        out_row_end[i]   = static_cast<MKL_INT>(h_row_offsets[i + 1]);
    }
    for (int k = 0; k < nnz; ++k) {
        out_col_ind[k] = static_cast<MKL_INT>(h_col_ind[k]);
    }

    sparse_matrix_t temp_result = nullptr;
    st = mkl_sparse_d_create_csr(
        &temp_result,
        SPARSE_INDEX_BASE_ZERO,
        n,
        n,
        out_row_start.data(),
        out_row_end.data(),
        out_col_ind.data(),
        h_vals.data());

    if (st != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_create_csr failed");
    }

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_matrix_t copied_result = nullptr;
    st = mkl_sparse_copy(temp_result, descr, &copied_result);
    mkl_sparse_destroy(temp_result);
    if (st != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_copy failed");
    }

    if (result != nullptr) {
        mkl_sparse_destroy(result);
    }
    result = copied_result;
    mkl_sparse_optimize(result);

    // -------------------------------------------------------------------------
    // 6) Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_workspace);
    cusparseDestroySpMat(matC);
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroy(handle);

    cudaFree(d_emb);
    cudaFree(d_row_offsets);
    cudaFree(d_col_ind);
    cudaFree(d_vals);
}

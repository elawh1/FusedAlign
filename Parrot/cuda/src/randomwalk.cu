#include "cuda_helper.cuh"

namespace {

__global__ void abs_diff_block_max_kernel(const double* a, const double* b, size_t total, double* block_max) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double local_max = 0.0;
    if (idx < total) {
        local_max = fabs(a[idx] - b[idx]);
    }

    __shared__ double partial[256];
    partial[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmax(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = partial[0];
    }
}

}

void get_batch_rwr_scores_cusparse_device(const CudaCsrMatrix& graph, std::vector<int> anchors, int n, float restart_prob, int max_iters, double* d_X_out) {
    const int rows = static_cast<int>(n);
    const int cols = static_cast<int>(n);
    const int nnz = graph.nnz;
    const int l = static_cast<int>(anchors.size());
    std::vector<double> E(static_cast<size_t>(n) * static_cast<size_t>(l), 0.0);
    for (int j = 0; j < l; ++j) {
        E[ anchors[j] + j * n ] = restart_prob;
    }
    double *d_X = nullptr, *d_Xnext = nullptr, *d_E = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_X, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Xnext, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_E, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_X, E.data(), static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_Xnext, 0, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_E, E.data(), static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t cusparseH;
    cublasHandle_t cublasH;
    CHECK_CUSPARSE(cusparseCreate(&cusparseH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matX, matXnext; 

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        rows, cols, nnz,
        graph.d_row_offsets,
        graph.d_col_ind,
        graph.d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    // Column-major dense matrices, ld = n
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matX,
        n, l, n,
        d_X,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matXnext,
        n, l, n,
        d_Xnext,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));
    
    const double alpha = static_cast<double>(1.0 - restart_prob);
    const double beta = 1.0;

    size_t bufferSize = 0;
    void* dBuffer = nullptr;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        cusparseH,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matX,
        &beta,
        matXnext,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
        cusparseH,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matX,
        &beta,
        matXnext,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));

    for(int i=0; i < max_iters-1; i++) {
        CHECK_CUSPARSE(cusparseSpMM(
            cusparseH,
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matX,
            &beta,
            matXnext,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));

        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(
            cublasH,
            n * l,
            &one,
            d_E,
            1,
            d_Xnext,
            1));

        std::swap(d_X, d_Xnext);
        CHECK_CUSPARSE(cusparseDnMatSetValues(matX, d_X));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matXnext, d_Xnext));

    }
    CHECK_CUDA(cudaMemcpy(d_X_out, d_X, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matX));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matXnext));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroy(cusparseH));
    CHECK_CUBLAS(cublasDestroy(cublasH));

    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Xnext));
    CHECK_CUDA(cudaFree(d_E));

}


void get_product_rwr_mat_cusparse_device(const CudaCsrMatrix& graph1_normalized,
                                         const CudaCsrMatrix& graph2_normalized,
                                         const double* d_in_cross_dist,
                                         int n1, int n2,
                                         double rwr_restart_prob, double gamma,
                                         double* d_out_cross_dist,
                                         int max_iters, double tol) {
    const size_t N = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    double *d_out_cross_dist_old = nullptr;
    double *d_temp = nullptr;
    double *d_CC = nullptr;
    double *d_block_max = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_out_cross_dist_old, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_temp,               N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_CC,                 N * sizeof(double)));
    const unsigned int diff_blocks = static_cast<unsigned int>((N + 255) / 256);
    CHECK_CUDA(cudaMalloc((void**)&d_block_max, diff_blocks * sizeof(double)));

    CHECK_CUDA(cudaMemset(d_out_cross_dist, 0, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_temp, 0, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_CC,   0, N * sizeof(double)));

    cusparseHandle_t sp_handle = nullptr;
    cublasHandle_t blas_handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&sp_handle));
    CHECK_CUBLAS(cublasCreate(&blas_handle));

    cusparseSpMatDescr_t G1 = nullptr, G2 = nullptr;
    cusparseDnMatDescr_t OUT_n1xn2 = nullptr;
    cusparseDnMatDescr_t TEMP_n1xn2 = nullptr;
    cusparseDnMatDescr_t OUT_n2xn1 = nullptr;
    cusparseDnMatDescr_t CC_n2xn1 = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(
        &G1,
        graph1_normalized.rows, graph1_normalized.cols, graph1_normalized.nnz,
        graph1_normalized.d_row_offsets,
        graph1_normalized.d_col_ind,
        graph1_normalized.d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &G2,
        graph2_normalized.rows, graph2_normalized.cols, graph2_normalized.nnz,
        graph2_normalized.d_row_offsets,
        graph2_normalized.d_col_ind,
        graph2_normalized.d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

     // Same underlying out_cross_dist buffer, but two shape views:
    // view 1: n1 x n2, ld=n1
    // view 2: n2 x n1, ld=n2
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &OUT_n1xn2, n1, n2, n1, d_out_cross_dist, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &OUT_n2xn1, n2, n1, n2, d_out_cross_dist, CUDA_R_64F, CUSPARSE_ORDER_COL));
        
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &TEMP_n1xn2, n1, n2, n1, d_temp, CUDA_R_64F, CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &CC_n2xn1, n2, n1, n2, d_CC, CUDA_R_64F, CUSPARSE_ORDER_COL));


    void* dBuffer1 = nullptr;
    void* dBuffer2 = nullptr;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    {
        double alpha = 1.0;
        double beta  = 0.0;

        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G1,
            OUT_n1xn2,
            &beta,
            TEMP_n1xn2,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            &bufferSize1));

        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));

        CHECK_CUSPARSE(cusparseSpMM_preprocess(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G1,
            OUT_n1xn2,
            &beta,
            TEMP_n1xn2,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer1));
    }

    {
        double alpha = (1.0 - rwr_restart_prob) * gamma;
        double beta  = (1.0 + rwr_restart_prob);

        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G2,
            OUT_n2xn1,
            &beta,
            CC_n2xn1,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            &bufferSize2));

        CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));

        CHECK_CUSPARSE(cusparseSpMM_preprocess(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G2,
            OUT_n2xn1,
            &beta,
            CC_n2xn1,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer2));
    }


    std::vector<double> h_block_max(diff_blocks, 0.0);
    for (int i = 0; i < max_iters; ++i) {
        CHECK_CUDA(cudaMemcpy(
            d_out_cross_dist_old,
            d_out_cross_dist,
            N * sizeof(double),
            cudaMemcpyDeviceToDevice));

        {
            const double alpha = 1.0;
            const double beta  = 0.0;
            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n2, n1,
                &alpha,
                d_in_cross_dist, n1,
                &beta, d_in_cross_dist, n2,
                d_CC, n2));
        }

        {
            double alpha = 1.0;
            double beta  = 0.0;

            CHECK_CUSPARSE(cusparseSpMM(
                sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                G1,
                OUT_n1xn2,
                &beta,
                TEMP_n1xn2,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer1));
        }

        {
            const double alpha = 1.0;
            const double beta  = 0.0;
            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n2, n1,
                &alpha,
                d_temp, n1,
                &beta, d_temp, n2,
                d_out_cross_dist, n2));
        }

        {
            double alpha = (1.0 - rwr_restart_prob) * gamma;
            double beta  = (1.0 + rwr_restart_prob);

            CHECK_CUSPARSE(cusparseSpMM(
                sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                G2,
                OUT_n2xn1,
                &beta,
                CC_n2xn1,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer2));
        }

        {
            const double alpha = 1.0;
            const double beta  = 0.0;
            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n1, n2,
                &alpha,
                d_CC, n2,
                &beta, d_CC, n1,
                d_out_cross_dist, n1));
        }
        abs_diff_block_max_kernel<<<diff_blocks, 256>>>(d_out_cross_dist, d_out_cross_dist_old, N, d_block_max);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(h_block_max.data(), d_block_max, diff_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double max_diff = 0.0;
        for (double value : h_block_max) max_diff = std::max(max_diff, value);
        if (max_diff < tol) break;
    }
    const double scale = 1.0 - gamma;
    CHECK_CUBLAS(cublasDscal(blas_handle, static_cast<int>(N), &scale, d_out_cross_dist, 1));

    CHECK_CUDA(cudaFree(dBuffer1));
    CHECK_CUDA(cudaFree(dBuffer2));

    CHECK_CUSPARSE(cusparseDestroyDnMat(CC_n2xn1));
    CHECK_CUSPARSE(cusparseDestroyDnMat(OUT_n2xn1));
    CHECK_CUSPARSE(cusparseDestroyDnMat(TEMP_n1xn2));
    CHECK_CUSPARSE(cusparseDestroyDnMat(OUT_n1xn2));

    CHECK_CUSPARSE(cusparseDestroySpMat(G2));
    CHECK_CUSPARSE(cusparseDestroySpMat(G1));

    CHECK_CUSPARSE(cusparseDestroy(sp_handle));
    CHECK_CUBLAS(cublasDestroy(blas_handle));

    CHECK_CUDA(cudaFree(d_CC));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_out_cross_dist_old));
    CHECK_CUDA(cudaFree(d_block_max));
}

#include "cuda_helper.cuh"
#include "helper.h"

void get_batch_rwr_scores_cusparse(sparse_matrix_t graph, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) {
    MKL_INT rows, cols;
    MKL_INT *row_start, *row_end, *col_indx;
    double *values;
    sparse_index_base_t indexing;

    mkl_sparse_d_export_csr(
        graph,
        &indexing,
        &rows,
        &cols,
        &row_start,
        &row_end,
        &col_indx,
        &values);

    if (rows != n || cols != n) {
        throw std::runtime_error("get_batch_rwr_scores_cusparse graph dimension mismatch");
    }

    std::vector<int> h_rowPtr(rows + 1);
    const int base = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;

    for (int i = 0; i < rows; i++) {
        h_rowPtr[i] = static_cast<int>(row_start[i]) - base;
    }
    h_rowPtr[rows] = static_cast<int>(row_end[rows - 1]) - base;

    const int nnz = h_rowPtr[rows];
    std::vector<int> h_colInd(nnz);
    for (int i = 0; i < nnz; ++i) {
        h_colInd[i] = static_cast<int>(col_indx[i]) - base;
    }

    auto time_0 = std::chrono::high_resolution_clock::now();
    const int l = static_cast<int>(anchors.size());
    std::vector<double> E(static_cast<size_t>(n) * static_cast<size_t>(l), 0.0);
    for (int j = 0; j < l; ++j) {
        E[ anchors[j] + j * n ] = restart_prob;   // one-hot at (row=anchor[j], col=j)
    }
    Xnext_vals.assign(static_cast<size_t>(n) * static_cast<size_t>(l), 0.0);

    int *d_rowPtr = nullptr, *d_colInd = nullptr;
    double *d_vals = nullptr, *d_X = nullptr, *d_Xnext = nullptr, *d_E = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals, nnz * sizeof(double)));

    CHECK_CUDA(cudaMalloc((void**)&d_X, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Xnext, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_E, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, values, nnz * sizeof(double), cudaMemcpyHostToDevice));

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
        d_rowPtr,
        d_colInd,
        d_vals,
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

    auto time_1 = std::chrono::high_resolution_clock::now();
    auto add_after_spmm_time = 0, spmm_time=0, swap_after_spmm_time=0;
    for(int i=0; i < max_iters-1; i++) {
        auto time_1_0 = std::chrono::high_resolution_clock::now();

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

        auto time_1_1 = std::chrono::high_resolution_clock::now();
        spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();

        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(
            cublasH,
            n * l,
            &one,
            d_E,
            1,
            d_Xnext,
            1));

        auto time_1_2 = std::chrono::high_resolution_clock::now();
        add_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();

        std::swap(d_X, d_Xnext);
        CHECK_CUSPARSE(cusparseDnMatSetValues(matX, d_X));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matXnext, d_Xnext));

        auto time_1_3 = std::chrono::high_resolution_clock::now();
        swap_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2).count();

    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "in get_batch_rwr_scores generate E " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   << 
    " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()  <<
    " spmm time " << spmm_time  <<  " add after spmm time " << add_after_spmm_time  <<  " swap after spmm time " << swap_after_spmm_time  << std::endl;


    Xnext_vals.assign(n * l, 0.0);
    CHECK_CUDA(cudaMemcpy(Xnext_vals.data(), d_X, static_cast<size_t>(n) * static_cast<size_t>(l) * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matX));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matXnext));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroy(cusparseH));
    CHECK_CUBLAS(cublasDestroy(cublasH));

    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Xnext));
    CHECK_CUDA(cudaFree(d_E));

}


std::vector<double> get_product_rwr_mat_cusparse( sparse_matrix_t graph1_normalized, sparse_matrix_t graph2_normalized, const std::vector<double>& in_cross_dist, 
    int n1, int n2,
    double rwr_restart_prob, double gamma, int max_iters, double tol) {
    auto time_0 = std::chrono::high_resolution_clock::now();
    const size_t N = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (in_cross_dist.size() != N) {
        throw std::runtime_error("get_product_rwr_mat_cusparse size mismatch");
    }
    auto time_1 = std::chrono::high_resolution_clock::now();
    std::vector<double> out_cross_dist(N, 0.0);
    std::vector<double> out_cross_dist_old(N, 0.0);
    std::vector<double> temp(n1 * n2);
    std::vector<double> CC(n1 * n2);


    MKL_INT rows_g1, cols_g1;
    MKL_INT *row_start_g1, *row_end_g1, *col_indx_g1;
    double *values_g1;
    sparse_index_base_t indexing_g1;

    mkl_sparse_d_export_csr(
        graph1_normalized,
        &indexing_g1,
        &rows_g1,
        &cols_g1,
        &row_start_g1,
        &row_end_g1,
        &col_indx_g1,
        &values_g1);

    std::vector<int> h_rowPtr_g1(rows_g1 + 1);

    for (int i = 0; i < rows_g1; i++) {
        h_rowPtr_g1[i] = row_start_g1[i];
    }
    h_rowPtr_g1[rows_g1] = row_end_g1[rows_g1 - 1];

    // If MKL indexing is one-based, convert to zero-based for cuSPARSE
    if (indexing_g1 == SPARSE_INDEX_BASE_ONE) {
        for (int i = 0; i <= rows_g1; i++) h_rowPtr_g1[i]--;
        for (int i = 0; i < row_end_g1[rows_g1 - 1]; i++) col_indx_g1[i]--;
    }
    int nnz_g1 = h_rowPtr_g1[rows_g1];

    MKL_INT rows_g2, cols_g2;
    MKL_INT *row_start_g2, *row_end_g2, *col_indx_g2;
    double *values_g2;
    sparse_index_base_t indexing_g2;

    mkl_sparse_d_export_csr(
        graph2_normalized,
        &indexing_g2,
        &rows_g2,
        &cols_g2,
        &row_start_g2,
        &row_end_g2,
        &col_indx_g2,
        &values_g2);

    std::vector<int> h_rowPtr_g2(rows_g2 + 1);

    for (int i = 0; i < rows_g2; i++) {
        h_rowPtr_g2[i] = row_start_g2[i];
    }
    h_rowPtr_g2[rows_g2] = row_end_g2[rows_g2 - 1];

    // If MKL indexing is one-based, convert to zero-based for cuSPARSE
    if (indexing_g2 == SPARSE_INDEX_BASE_ONE) {
        for (int i = 0; i <= rows_g2; i++) h_rowPtr_g2[i]--;
        for (int i = 0; i < row_end_g2[rows_g2 - 1]; i++) col_indx_g2[i]--;
    }
    int nnz_g2 = h_rowPtr_g2[rows_g2];
    // ------------------------------------------------------------
    // Allocate GPU memory
    // ------------------------------------------------------------
    int *d_rowPtr_g1 = nullptr, *d_colInd_g1 = nullptr;
    double *d_vals_g1 = nullptr;
    int *d_rowPtr_g2 = nullptr, *d_colInd_g2 = nullptr;
    double *d_vals_g2 = nullptr;
    double *d_out_cross_dist = nullptr;
    double *d_out_cross_dist_old = nullptr;
    double *d_in_cross_dist = nullptr;
    double *d_temp = nullptr;
    double *d_CC = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr_g1, (rows_g1 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd_g1, nnz_g1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_g1, nnz_g1 * sizeof(double)));

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr_g2, (rows_g2 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd_g2, nnz_g2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_g2, nnz_g2 * sizeof(double)));

    CHECK_CUDA(cudaMalloc((void**)&d_out_cross_dist,     N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_out_cross_dist_old, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_in_cross_dist,      N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_temp,               N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_CC,                 N * sizeof(double)));

    // ------------------------------------------------------------
    // Copy data to GPU
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_rowPtr_g1, h_rowPtr_g1.data(), (rows_g1 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd_g1, col_indx_g1, nnz_g1 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_g1, values_g1, nnz_g1 * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_rowPtr_g2, h_rowPtr_g2.data(), (rows_g2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd_g2, col_indx_g2, nnz_g2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_g2, values_g2, nnz_g2 * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_out_cross_dist, out_cross_dist.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_cross_dist,  in_cross_dist.data(),  N * sizeof(double), cudaMemcpyHostToDevice));
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
        rows_g1, cols_g1, nnz_g1,
        d_rowPtr_g1,
        d_colInd_g1,
        d_vals_g1,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &G2,
        rows_g2, cols_g2, nnz_g2,
        d_rowPtr_g2,
        d_colInd_g2,
        d_vals_g2,
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


    auto time_2 = std::chrono::high_resolution_clock::now();
    auto time_only_spmm = time_2 - time_2;
    for (int i = 0; i < max_iters; ++i) {
        auto time_2_0 = std::chrono::high_resolution_clock::now();

        // out_cross_dist_old = out_cross_dist
        CHECK_CUDA(cudaMemcpy(
            d_out_cross_dist_old,
            d_out_cross_dist,
            N * sizeof(double),
            cudaMemcpyDeviceToDevice));

        // CC = in_cross_dist^T
        {
            const double alpha = 1.0;
            const double beta  = 0.0;

            // input  d_in_cross_dist: n1 x n2, ld = n1
            // output d_CC          : n2 x n1, ld = n2
            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n2, n1,
                &alpha,
                d_in_cross_dist, n1,
                &beta,
                d_in_cross_dist, n2, // unused because beta=0
                d_CC, n2));
        }

        // temp = G1 * out_cross_dist
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

        // out_cross_dist = temp^T
        {
            const double alpha = 1.0;
            const double beta  = 0.0;

            // temp: n1 x n2 -> out_cross_dist viewed as n2 x n1
            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n2, n1,
                &alpha,
                d_temp, n1,
                &beta,
                d_temp, n2, // unused because beta=0
                d_out_cross_dist, n2));
        }

        // CC = coeff * G2 * out_cross_dist + (1+rwr_restart_prob) * CC
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

        // out_cross_dist = CC^T
        {
            const double alpha = 1.0;
            const double beta  = 0.0;

            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n1, n2,
                &alpha,
                d_CC, n2,
                &beta,
                d_CC, n1, // unused because beta=0
                d_out_cross_dist, n1));
        }

        // host-side convergence check
        CHECK_CUDA(cudaMemcpy(
            out_cross_dist.data(),
            d_out_cross_dist,
            N * sizeof(double),
            cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaMemcpy(
            out_cross_dist_old.data(),
            d_out_cross_dist_old,
            N * sizeof(double),
            cudaMemcpyDeviceToHost));


        auto time_2_1 = std::chrono::high_resolution_clock::now();
        time_only_spmm += time_2_1 - time_2_0;
        //compute max of absolute values of (out_cross_dist - out_cross_dist_old)
        if (out_cross_dist_converge(out_cross_dist, out_cross_dist_old, tol)) break;
        auto time_2_2 = std::chrono::high_resolution_clock::now();

        
     }
    auto time_3 = std::chrono::high_resolution_clock::now();
    
    //out_cross_dist = (1 - self.gamma) * out_cross_dist
    cblas_dscal(n1 * n2, 1.0 - gamma, out_cross_dist.data(), 1);
    auto time_4 = std::chrono::high_resolution_clock::now();

    std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   << 
    " init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   << 
    " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()   << 
    " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() <<
    " time only spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(time_only_spmm).count()
    << std::endl;
    

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
    CHECK_CUDA(cudaFree(d_in_cross_dist));
    CHECK_CUDA(cudaFree(d_out_cross_dist_old));
    CHECK_CUDA(cudaFree(d_out_cross_dist));

    CHECK_CUDA(cudaFree(d_rowPtr_g1));
    CHECK_CUDA(cudaFree(d_colInd_g1));
    CHECK_CUDA(cudaFree(d_vals_g1));

    CHECK_CUDA(cudaFree(d_rowPtr_g2));
    CHECK_CUDA(cudaFree(d_colInd_g2));
    CHECK_CUDA(cudaFree(d_vals_g2));

    return out_cross_dist;
}

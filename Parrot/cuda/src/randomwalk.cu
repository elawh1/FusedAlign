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

void ensure_batch_rwr_handles(BatchRwrContext& context) {
    if (context.cusparse == nullptr) {
        CHECK_CUSPARSE(cusparseCreate(&context.cusparse));
    }
    if (context.cublas == nullptr) {
        CHECK_CUBLAS(cublasCreate(&context.cublas));
    }
}

void reset_batch_rwr_dense_mats(BatchRwrContext& context) {
    if (context.mat_x != nullptr) {
        CHECK_CUSPARSE(cusparseDestroyDnMat(context.mat_x));
        context.mat_x = nullptr;
    }
    if (context.mat_xnext != nullptr) {
        CHECK_CUSPARSE(cusparseDestroyDnMat(context.mat_xnext));
        context.mat_xnext = nullptr;
    }
    context.current_n = 0;
    context.current_l = 0;
}

void ensure_batch_rwr_dense_capacity(BatchRwrContext& context, size_t dense_elems) {
    if (context.dense_capacity >= dense_elems) {
        return;
    }

    reset_batch_rwr_dense_mats(context);
    if (context.d_x != nullptr) {
        CHECK_CUDA(cudaFree(context.d_x));
        context.d_x = nullptr;
    }
    if (context.d_xnext != nullptr) {
        CHECK_CUDA(cudaFree(context.d_xnext));
        context.d_xnext = nullptr;
    }
    if (context.d_e != nullptr) {
        CHECK_CUDA(cudaFree(context.d_e));
        context.d_e = nullptr;
    }

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&context.d_x), dense_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&context.d_xnext), dense_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&context.d_e), dense_elems * sizeof(double)));
    context.dense_capacity = dense_elems;
}

void ensure_batch_rwr_dense_mats(BatchRwrContext& context, int n, int l) {
    if (context.mat_x != nullptr && context.mat_xnext != nullptr &&
        context.current_n == n && context.current_l == l) {
        CHECK_CUSPARSE(cusparseDnMatSetValues(context.mat_x, context.d_x));
        CHECK_CUSPARSE(cusparseDnMatSetValues(context.mat_xnext, context.d_xnext));
        return;
    }

    reset_batch_rwr_dense_mats(context);
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &context.mat_x,
        n, l, n,
        context.d_x,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &context.mat_xnext,
        n, l, n,
        context.d_xnext,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));
    context.current_n = n;
    context.current_l = l;
}

void ensure_batch_rwr_workspace(BatchRwrContext& context, size_t required_bytes) {
    if (context.buffer_capacity >= required_bytes) {
        return;
    }
    if (context.d_buffer != nullptr) {
        CHECK_CUDA(cudaFree(context.d_buffer));
        context.d_buffer = nullptr;
        context.buffer_capacity = 0;
    }
    if (required_bytes > 0) {
        CHECK_CUDA(cudaMalloc(&context.d_buffer, required_bytes));
        context.buffer_capacity = required_bytes;
    }
}

}

void get_batch_rwr_scores_cusparse_device(const CudaCsrMatrix& graph, std::vector<int> anchors, int n, float restart_prob, int max_iters, double* d_X_out) {
    BatchRwrContext context;
    get_batch_rwr_scores_cusparse_device(graph, std::move(anchors), n, restart_prob, max_iters, d_X_out, context);
    destroy_batch_rwr_context(context);
}

void get_batch_rwr_scores_cusparse_device(const CudaCsrMatrix& graph, std::vector<int> anchors, int n, float restart_prob, int max_iters, double* d_X_out, BatchRwrContext& context) {
#ifdef DEBUG
    using Clock = std::chrono::high_resolution_clock;
    auto elapsed_ms = [](const Clock::time_point& start, const Clock::time_point& end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };
    static int batch_rwr_call_counter = 0;
    const int call_id = ++batch_rwr_call_counter;
#endif

    const int rows = static_cast<int>(n);
    const int cols = static_cast<int>(n);
    const int nnz = graph.nnz;
    const int l = static_cast<int>(anchors.size());
    const size_t dense_elems = static_cast<size_t>(n) * static_cast<size_t>(l);

#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());

    const auto seed_build_start = Clock::now();
#endif
    std::vector<double> E(static_cast<size_t>(n) * static_cast<size_t>(l), 0.0);
    for (int j = 0; j < l; ++j) {
        E[ anchors[j] + j * n ] = restart_prob;
    }
#ifdef DEBUG
    const auto seed_build_end = Clock::now();

    const bool reused_handles = context.cusparse != nullptr && context.cublas != nullptr;
    const bool reused_dense_capacity = context.dense_capacity >= dense_elems;
    const bool reused_dense_shape = context.mat_x != nullptr && context.mat_xnext != nullptr &&
                                    context.current_n == n && context.current_l == l;
    const auto context_setup_start = Clock::now();
#endif
    ensure_batch_rwr_handles(context);
    ensure_batch_rwr_dense_capacity(context, dense_elems);
    ensure_batch_rwr_dense_mats(context, n, l);
#ifdef DEBUG
    const auto context_setup_end = Clock::now();

    const auto upload_start = Clock::now();
#endif
    CHECK_CUDA(cudaMemcpy(context.d_x, E.data(), dense_elems * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(context.d_xnext, 0, dense_elems * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(context.d_e, E.data(), dense_elems * sizeof(double), cudaMemcpyHostToDevice));
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto upload_end = Clock::now();

    const bool recreated_mat_a = context.mat_a != nullptr;
    const auto spmat_setup_start = Clock::now();
#endif
    if (context.mat_a != nullptr) {
        CHECK_CUSPARSE(cusparseDestroySpMat(context.mat_a));
        context.mat_a = nullptr;
    }

    CHECK_CUSPARSE(cusparseCreateCsr(
        &context.mat_a,
        rows, cols, nnz,
        graph.d_row_offsets,
        graph.d_col_ind,
        graph.d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));
#ifdef DEBUG
    const auto spmat_setup_end = Clock::now();
#endif
    
    const double alpha = static_cast<double>(1.0 - restart_prob);
    const double beta = 1.0;

    size_t bufferSize = 0;
#ifdef DEBUG
    const auto preprocess_start = Clock::now();
#endif
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        context.cusparse,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        context.mat_a,
        context.mat_x,
        &beta,
        context.mat_xnext,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

#ifdef DEBUG
    const bool reused_workspace = context.buffer_capacity >= bufferSize;
#endif
    ensure_batch_rwr_workspace(context, bufferSize);
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
        context.cusparse,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        context.mat_a,
        context.mat_x,
        &beta,
        context.mat_xnext,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        context.d_buffer));
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto preprocess_end = Clock::now();

    const auto iter_loop_start = Clock::now();
#endif
    for(int i=0; i < max_iters-1; i++) {
        CHECK_CUSPARSE(cusparseSpMM(
            context.cusparse,
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            context.mat_a,
            context.mat_x,
            &beta,
            context.mat_xnext,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            context.d_buffer));

        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(
            context.cublas,
            n * l,
            &one,
            context.d_e,
            1,
            context.d_xnext,
            1));

        std::swap(context.d_x, context.d_xnext);
        CHECK_CUSPARSE(cusparseDnMatSetValues(context.mat_x, context.d_x));
        CHECK_CUSPARSE(cusparseDnMatSetValues(context.mat_xnext, context.d_xnext));

    }
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto iter_loop_end = Clock::now();

    const auto output_copy_start = Clock::now();
#endif
    CHECK_CUDA(cudaMemcpy(d_X_out, context.d_x, dense_elems * sizeof(double), cudaMemcpyDeviceToDevice));
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto output_copy_end = Clock::now();

    DEBUG_ONLY(std::cout
        << "batch RWR call " << call_id
        << " (n=" << n << ", l=" << l << ", nnz=" << nnz << ")\n"
        << "  seed build: " << elapsed_ms(seed_build_start, seed_build_end) << " ms\n"
        << "  context setup: " << elapsed_ms(context_setup_start, context_setup_end) << " ms"
        << " [handles=" << (reused_handles ? "reused" : "new")
        << ", dense_capacity=" << (reused_dense_capacity ? "reused" : "resized")
        << ", dense_views=" << (reused_dense_shape ? "reused" : "rebuilt") << "]\n"
        << "  seed uploads: " << elapsed_ms(upload_start, upload_end) << " ms\n"
        << "  sparse descriptor setup: " << elapsed_ms(spmat_setup_start, spmat_setup_end) << " ms"
        << " [mat_a=" << (recreated_mat_a ? "recreated" : "created") << "]\n"
        << "  spmm preprocess: " << elapsed_ms(preprocess_start, preprocess_end) << " ms"
        << " [workspace=" << (reused_workspace ? "reused" : "resized") << ", bytes=" << bufferSize << "]\n"
        << "  iteration loop: " << elapsed_ms(iter_loop_start, iter_loop_end) << " ms\n"
        << "  output copy: " << elapsed_ms(output_copy_start, output_copy_end) << " ms\n"
        << "  total inside function: " << elapsed_ms(seed_build_start, output_copy_end) << " ms"
        << std::endl);
#endif

}

void destroy_batch_rwr_context(BatchRwrContext& context) {
    if (context.mat_a != nullptr) {
        CHECK_CUSPARSE(cusparseDestroySpMat(context.mat_a));
        context.mat_a = nullptr;
    }
    reset_batch_rwr_dense_mats(context);
    if (context.cusparse != nullptr) {
        CHECK_CUSPARSE(cusparseDestroy(context.cusparse));
        context.cusparse = nullptr;
    }
    if (context.cublas != nullptr) {
        CHECK_CUBLAS(cublasDestroy(context.cublas));
        context.cublas = nullptr;
    }
    if (context.d_x != nullptr) {
        CHECK_CUDA(cudaFree(context.d_x));
        context.d_x = nullptr;
    }
    if (context.d_xnext != nullptr) {
        CHECK_CUDA(cudaFree(context.d_xnext));
        context.d_xnext = nullptr;
    }
    if (context.d_e != nullptr) {
        CHECK_CUDA(cudaFree(context.d_e));
        context.d_e = nullptr;
    }
    if (context.d_buffer != nullptr) {
        CHECK_CUDA(cudaFree(context.d_buffer));
        context.d_buffer = nullptr;
    }
    context.dense_capacity = 0;
    context.buffer_capacity = 0;
}


void get_product_rwr_mat_cusparse_device(const CudaCsrMatrix& graph1_normalized,
                                         const CudaCsrMatrix& graph2_normalized,
                                         const double* d_in_cross_dist,
                                         int n1, int n2,
                                         double rwr_restart_prob, double gamma,
                                         double* d_out_cross_dist,
                                         int max_iters, double tol) {
    const size_t N = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    double* d_prev = d_out_cross_dist;
    double* d_next = nullptr;
    double *d_temp = nullptr;
    double *d_block_max = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_next,               N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_temp,               N * sizeof(double)));
    const unsigned int diff_blocks = static_cast<unsigned int>((N + 255) / 256);
    CHECK_CUDA(cudaMalloc((void**)&d_block_max, diff_blocks * sizeof(double)));

    CHECK_CUDA(cudaMemset(d_prev, 0, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_temp, 0, N * sizeof(double)));

    cusparseHandle_t sp_handle = nullptr;
    cublasHandle_t blas_handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&sp_handle));
    CHECK_CUBLAS(cublasCreate(&blas_handle));

    cusparseSpMatDescr_t G1 = nullptr, G2 = nullptr;
    cusparseDnMatDescr_t PREV_n1xn2 = nullptr;
    cusparseDnMatDescr_t TEMP_n1xn2 = nullptr;
    cusparseDnMatDescr_t TEMP_t_n2xn1 = nullptr;
    cusparseDnMatDescr_t NEXT_t_n2xn1 = nullptr;

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

    // Column-major view of the current iterate.
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &PREV_n1xn2, n1, n2, n1, d_prev, CUDA_R_64F, CUSPARSE_ORDER_COL));
    // Column-major intermediate.
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &TEMP_n1xn2, n1, n2, n1, d_temp, CUDA_R_64F, CUSPARSE_ORDER_COL));
    // Row-major n2 x n1 view over the same TEMP storage, equivalent to TEMP^T.
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &TEMP_t_n2xn1, n2, n1, n1, d_temp, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    // Row-major n2 x n1 view over NEXT storage, equivalent to NEXT^T while NEXT itself
    // is kept in n1 x n2 column-major layout for the rest of the pipeline.
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &NEXT_t_n2xn1, n2, n1, n1, d_next, CUDA_R_64F, CUSPARSE_ORDER_ROW));

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
            PREV_n1xn2,
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
            PREV_n1xn2,
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
            TEMP_t_n2xn1,
            &beta,
            NEXT_t_n2xn1,
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
            TEMP_t_n2xn1,
            &beta,
            NEXT_t_n2xn1,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer2));
    }

    double* d_final = d_prev;
    for (int i = 0; i < max_iters; ++i) {
        CHECK_CUDA(cudaMemcpy(d_next, d_in_cross_dist, N * sizeof(double), cudaMemcpyDeviceToDevice));

        {
            double alpha = 1.0;
            double beta  = 0.0;

            CHECK_CUSPARSE(cusparseSpMM(
                sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                G1,
                PREV_n1xn2,
                &beta,
                TEMP_n1xn2,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer1));
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
                TEMP_t_n2xn1,
                &beta,
                NEXT_t_n2xn1,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer2));
        }

        abs_diff_block_max_kernel<<<diff_blocks, 256>>>(d_next, d_prev, N, d_block_max);
        CHECK_CUDA(cudaGetLastError());
        double max_diff = 0.0;
        int max_idx = 0;
        CHECK_CUBLAS(cublasIdamax(blas_handle, static_cast<int>(diff_blocks), d_block_max, 1, &max_idx));
        if (max_idx > 0) {
            CHECK_CUDA(cudaMemcpy(&max_diff, d_block_max + (max_idx - 1), sizeof(double), cudaMemcpyDeviceToHost));
        }
        d_final = d_next;
        if (max_diff < tol) {
            break;
        }

        std::swap(d_prev, d_next);
        CHECK_CUSPARSE(cusparseDnMatSetValues(PREV_n1xn2, d_prev));
        CHECK_CUSPARSE(cusparseDnMatSetValues(NEXT_t_n2xn1, d_next));
        d_final = d_prev;
    }
    const double scale = 1.0 - gamma;
    CHECK_CUBLAS(cublasDscal(blas_handle, static_cast<int>(N), &scale, d_final, 1));
    if (d_final != d_out_cross_dist) {
        CHECK_CUDA(cudaMemcpy(d_out_cross_dist, d_final, N * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA(cudaFree(dBuffer1));
    CHECK_CUDA(cudaFree(dBuffer2));

    CHECK_CUSPARSE(cusparseDestroyDnMat(NEXT_t_n2xn1));
    CHECK_CUSPARSE(cusparseDestroyDnMat(TEMP_t_n2xn1));
    CHECK_CUSPARSE(cusparseDestroyDnMat(TEMP_n1xn2));
    CHECK_CUSPARSE(cusparseDestroyDnMat(PREV_n1xn2));

    CHECK_CUSPARSE(cusparseDestroySpMat(G2));
    CHECK_CUSPARSE(cusparseDestroySpMat(G1));

    CHECK_CUSPARSE(cusparseDestroy(sp_handle));
    CHECK_CUBLAS(cublasDestroy(blas_handle));

    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_next));
    CHECK_CUDA(cudaFree(d_block_max));
}

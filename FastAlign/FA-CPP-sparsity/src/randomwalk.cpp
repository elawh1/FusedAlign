#include "cuda_helper.cuh"

void get_batch_rwr_scores_sparse_device(const CudaSparseMatrix& graph,
                                        std::vector<int> anchors,
                                        int n,
                                        element_type restart_prob,
                                        int max_iters,
                                        element_type* X_out,
                                        BatchRwrContext& context) {
    const int l = static_cast<int>(anchors.size());
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(l);
    if (total == 0) return;

    if (context.cublas == nullptr) CHECK_CUBLAS(cublasCreate(&context.cublas));
    if (context.cusparse == nullptr) CHECK_CUSPARSE(cusparseCreate(&context.cusparse));
    if (context.dense_capacity < total) {
        cuda_free_and_null(context.d_x);
        cuda_free_and_null(context.d_xnext);
        CHECK_CUDA(CUDA_MALLOC(&context.d_x, total * sizeof(element_type)));
        CHECK_CUDA(CUDA_MALLOC(&context.d_xnext, total * sizeof(element_type)));
        context.dense_capacity = total;
    }
    context.current_n = n;
    context.current_l = l;

    std::fill(context.d_x, context.d_x + total, static_cast<element_type>(0));
    std::fill(context.d_xnext, context.d_xnext + total, static_cast<element_type>(0));
    for (int j = 0; j < l; ++j) {
        const int anchor = anchors[static_cast<size_t>(j)];
        if (anchor < 0 || anchor >= n) {
            throw std::out_of_range("anchor out of range inside get_batch_rwr_scores_sparse_device");
        }
        context.d_x[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(anchor)] = restart_prob;
    }

    const element_type one = static_cast<element_type>(1);
    const element_type zero = static_cast<element_type>(0);
    for (int iter = 0; iter < std::max(0, max_iters - 1); ++iter) {
        cuda_sparse_spmm(context.cusparse,
                         graph,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         context.d_x,
                         n,
                         l,
                         n,
                         CUSPARSE_ORDER_COL,
                         zero,
                         context.d_xnext,
                         n,
                         l,
                         n,
                         CUSPARSE_ORDER_COL,
                         one);
        for (int j = 0; j < l; ++j) {
            context.d_xnext[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(anchors[static_cast<size_t>(j)])] += restart_prob;
        }
        std::swap(context.d_x, context.d_xnext);
    }
    std::copy(context.d_x, context.d_x + total, X_out);
}

void destroy_batch_rwr_context(BatchRwrContext& context) {
    if (context.cublas != nullptr) {
        CHECK_CUBLAS(cublasDestroy(context.cublas));
        context.cublas = nullptr;
    }
    if (context.cusparse != nullptr) {
        CHECK_CUSPARSE(cusparseDestroy(context.cusparse));
        context.cusparse = nullptr;
    }
    cuda_free_and_null(context.d_x);
    cuda_free_and_null(context.d_xnext);
    cuda_free_and_null(context.d_anchors);
    context.dense_capacity = 0;
    context.anchor_capacity = 0;
    context.current_n = 0;
    context.current_l = 0;
}

void get_product_rwr_mat_sparse_device(const CudaSparseMatrix& graph1_normalized,
                                       const CudaSparseMatrix& graph2_normalized,
                                       const element_type* in_cross_dist,
                                       int n1,
                                       int n2,
                                       element_type rwr_restart_prob,
                                       element_type gamma,
                                       element_type* out_cross_dist_result,
                                       int max_iters,
                                       element_type tol) {
    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    std::vector<element_type> old_vals(total, static_cast<element_type>(0));
    std::vector<element_type> new_vals(total, static_cast<element_type>(0));
    std::vector<element_type> temp(total, static_cast<element_type>(0));

    const element_type one = static_cast<element_type>(1);
    const element_type zero = static_cast<element_type>(0);
    const element_type coeff = (static_cast<element_type>(1) - rwr_restart_prob) * gamma;
    const element_type restart_scale = static_cast<element_type>(1) + rwr_restart_prob;

    cublasHandle_t blas_handle = nullptr;
    cusparseHandle_t sparse_handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&blas_handle));
    CHECK_CUSPARSE(cusparseCreate(&sparse_handle));

    element_type* final_vals = old_vals.data();
    for (int iter = 0; iter < max_iters; ++iter) {
        cuda_sparse_spmm(sparse_handle,
                         graph2_normalized,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         old_vals.data(),
                         n2,
                         n1,
                         n1,
                         CUSPARSE_ORDER_ROW,
                         zero,
                         temp.data(),
                         n2,
                         n1,
                         n1,
                         CUSPARSE_ORDER_ROW,
                         one);

        cuda_sparse_spmm(sparse_handle,
                         graph1_normalized,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         temp.data(),
                         n1,
                         n2,
                         n1,
                         CUSPARSE_ORDER_COL,
                         zero,
                         new_vals.data(),
                         n1,
                         n2,
                         n1,
                         CUSPARSE_ORDER_COL,
                         coeff);

        CHECK_CUBLAS(cuda_cublas_axpy(blas_handle, static_cast<int>(total), &restart_scale,
                                      in_cross_dist, 1, new_vals.data(), 1));

        element_type max_diff = static_cast<element_type>(0);
        #pragma omp parallel for reduction(max:max_diff) schedule(static)
        for (size_t idx = 0; idx < total; ++idx) {
            const element_type diff = std::abs(new_vals[idx] - old_vals[idx]);
            if (diff > max_diff) max_diff = diff;
        }
        final_vals = new_vals.data();
        if (max_diff < tol) break;
        old_vals.swap(new_vals);
        final_vals = old_vals.data();
    }

    std::copy(final_vals, final_vals + total, out_cross_dist_result);
    const element_type scale = static_cast<element_type>(1) - gamma;
    CHECK_CUBLAS(cuda_cublas_scal(blas_handle, static_cast<int>(total), &scale, out_cross_dist_result, 1));

    CHECK_CUSPARSE(cusparseDestroy(sparse_handle));
    CHECK_CUBLAS(cublasDestroy(blas_handle));
}

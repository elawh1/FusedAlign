#include "cuda_helper.cuh"

void get_batch_rwr_scores_dense_device(const CudaDenseMatrix& graph,
                                       std::vector<int> anchors,
                                       int n,
                                       element_type restart_prob,
                                       int max_iters,
                                       element_type* X_out,
                                       BatchRwrContext& context) {
    const int l = static_cast<int>(anchors.size());
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(l);
    if (total == 0) return;

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
            throw std::out_of_range("anchor out of range inside get_batch_rwr_scores_dense_device");
        }
        context.d_x[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(anchor)] = restart_prob;
    }

    const element_type one = static_cast<element_type>(1);
    const element_type zero = static_cast<element_type>(0);
    for (int iter = 0; iter < std::max(0, max_iters - 1); ++iter) {
        CHECK_CUBLAS(cuda_cublas_gemm(0, CUBLAS_OP_N, CUBLAS_OP_N,
                                      n, l, n,
                                      &one, graph.d_vals, graph.rows,
                                      context.d_x, n,
                                      &zero, context.d_xnext, n));
        for (int j = 0; j < l; ++j) {
            context.d_xnext[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(anchors[static_cast<size_t>(j)])] += restart_prob;
        }
        std::swap(context.d_x, context.d_xnext);
    }
    std::copy(context.d_x, context.d_x + total, X_out);
}

void destroy_batch_rwr_context(BatchRwrContext& context) {
    cuda_free_and_null(context.d_x);
    cuda_free_and_null(context.d_xnext);
    cuda_free_and_null(context.d_anchors);
    context.dense_capacity = 0;
    context.anchor_capacity = 0;
    context.current_n = 0;
    context.current_l = 0;
}

void get_product_rwr_mat_dense_device(const CudaDenseMatrix& graph1_normalized,
                                      const CudaDenseMatrix& graph2_normalized,
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

    element_type* final_vals = old_vals.data();
    for (int iter = 0; iter < max_iters; ++iter) {
        CHECK_CUBLAS(cuda_cublas_gemm(0, CUBLAS_OP_N, CUBLAS_OP_T,
                                      n1, n2, n2,
                                      &one, old_vals.data(), n1,
                                      graph2_normalized.d_vals, graph2_normalized.rows,
                                      &zero, temp.data(), n1));
        CHECK_CUBLAS(cuda_cublas_gemm(0, CUBLAS_OP_N, CUBLAS_OP_N,
                                      n1, n2, n1,
                                      &coeff, graph1_normalized.d_vals, graph1_normalized.rows,
                                      temp.data(), n1,
                                      &zero, new_vals.data(), n1));
        CHECK_CUBLAS(cuda_cublas_axpy(0, static_cast<int>(total), &restart_scale,
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

    const element_type scale = static_cast<element_type>(1) - gamma;
    std::copy(final_vals, final_vals + total, out_cross_dist_result);
    CHECK_CUBLAS(cuda_cublas_scal(0, static_cast<int>(total), &scale, out_cross_dist_result, 1));
}

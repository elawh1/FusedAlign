#include "cuda_helper.cuh"

namespace {

__global__ void abs_diff_block_max_kernel(const element_type* a, const element_type* b, size_t total, element_type* block_max) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    element_type local_max = static_cast<element_type>(0.0);
    if (idx < total) {
        local_max = static_cast<element_type>(fabs(static_cast<double>(a[idx] - b[idx])));
    }

    __shared__ element_type partial[256];
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

__global__ void reduce_block_max_to_scalar_kernel(element_type* block_max, unsigned int n) {
    element_type local_max = static_cast<element_type>(0.0);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmax(local_max, block_max[i]);
    }
    __shared__ element_type smem[256];
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_max[0] = smem[0];
    }
}

__global__ void write_restart_entries_kernel(element_type* x,
                                             const int* anchors,
                                             int n,
                                             int l,
                                             element_type restart_prob) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= l) {
        return;
    }

    const size_t offset = static_cast<size_t>(anchors[j]) + static_cast<size_t>(j) * static_cast<size_t>(n);
    x[offset] = restart_prob;
}

__global__ void add_restart_entries_kernel(element_type* x,
                                           const int* anchors,
                                           int n,
                                           int l,
                                           element_type restart_prob) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= l) {
        return;
    }

    const size_t offset = static_cast<size_t>(anchors[j]) + static_cast<size_t>(j) * static_cast<size_t>(n);
    x[offset] += restart_prob;
}

void ensure_batch_rwr_handles(BatchRwrContext& context) {
    if (context.cublas == nullptr) {
        CHECK_CUBLAS(cublasCreate(&context.cublas));
    }
    if (context.cusparse == nullptr) {
        CHECK_CUSPARSE(cusparseCreate(&context.cusparse));
    }
}

void reset_batch_rwr_dense_mats(BatchRwrContext& context) {
    context.current_n = 0;
    context.current_l = 0;
}

void ensure_batch_rwr_dense_capacity(BatchRwrContext& context, size_t dense_elems) {
    if (context.dense_capacity >= dense_elems) {
        return;
    }

    reset_batch_rwr_dense_mats(context);
    if (context.d_x != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_x));
        context.d_x = nullptr;
    }
    if (context.d_xnext != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_xnext));
        context.d_xnext = nullptr;
    }

    CHECK_CUDA(CUDA_MALLOC(reinterpret_cast<void**>(&context.d_x), dense_elems * sizeof(element_type)));
    CHECK_CUDA(CUDA_MALLOC(reinterpret_cast<void**>(&context.d_xnext), dense_elems * sizeof(element_type)));
    context.dense_capacity = dense_elems;
}

void ensure_batch_rwr_dense_mats(BatchRwrContext& context, int n, int l) {
    context.current_n = n;
    context.current_l = l;
}

void ensure_batch_rwr_anchor_capacity(BatchRwrContext& context, size_t required_anchors) {
    if (context.anchor_capacity >= required_anchors) {
        return;
    }
    if (context.d_anchors != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_anchors));
        context.d_anchors = nullptr;
        context.anchor_capacity = 0;
    }
    if (required_anchors > 0) {
        CHECK_CUDA(CUDA_MALLOC(reinterpret_cast<void**>(&context.d_anchors), required_anchors * sizeof(int)));
        context.anchor_capacity = required_anchors;
    }
}

bool cross_timing_enabled() {
#ifdef DEBUG
    return true;
#else
    return false;
#endif
}

using CrossClock = std::chrono::high_resolution_clock;

CrossClock::time_point cross_timing_now(bool enabled) {
    if (enabled) {
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    return CrossClock::now();
}

long long cross_timing_elapsed_ms(bool enabled, const CrossClock::time_point& start) {
    if (!enabled) {
        return 0;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto end = CrossClock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

}  // namespace

void get_batch_rwr_scores_sparse_device(const CudaSparseMatrix& graph, std::vector<int> anchors, int n, element_type restart_prob, int max_iters, element_type* d_X_out, BatchRwrContext& context) {
#ifdef DEBUG
    using Clock = std::chrono::high_resolution_clock;
    auto elapsed_ms = [](const Clock::time_point& start, const Clock::time_point& end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };
    static int batch_rwr_call_counter = 0;
    const int call_id = ++batch_rwr_call_counter;
#endif

    const int l = static_cast<int>(anchors.size());
    const size_t dense_elems = static_cast<size_t>(n) * static_cast<size_t>(l);
    const int total_updates = std::max(0, max_iters - 1);
    const unsigned int anchor_blocks = static_cast<unsigned int>((static_cast<unsigned int>(l) + 255U) / 256U);

#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto seed_build_start = Clock::now();
#endif
    for (int j = 0; j < l; ++j) {
        const int anchor = anchors[static_cast<size_t>(j)];
        if (anchor < 0 || anchor >= n) {
            throw std::out_of_range("anchor out of range inside get_batch_rwr_scores_sparse_device");
        }
    }
#ifdef DEBUG
    const auto seed_build_end = Clock::now();
    const bool reused_handles = context.cublas != nullptr;
    const bool reused_dense_capacity = context.dense_capacity >= dense_elems;
    const bool reused_dense_shape = context.current_n == n && context.current_l == l;
    const bool reused_anchor_capacity = context.anchor_capacity >= static_cast<size_t>(l);
    const auto context_setup_start = Clock::now();
#endif
    ensure_batch_rwr_handles(context);
    ensure_batch_rwr_dense_capacity(context, dense_elems);
    ensure_batch_rwr_dense_mats(context, n, l);
    ensure_batch_rwr_anchor_capacity(context, static_cast<size_t>(l));
#ifdef DEBUG
    const auto context_setup_end = Clock::now();
    const auto upload_start = Clock::now();
#endif
    if (l > 0) {
        CHECK_CUDA(cudaMemcpy(context.d_anchors,
                              anchors.data(),
                              static_cast<size_t>(l) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMemset(context.d_x, 0, dense_elems * sizeof(element_type)));
    CHECK_CUDA(cudaMemset(context.d_xnext, 0, dense_elems * sizeof(element_type)));
    if (anchor_blocks > 0) {
        write_restart_entries_kernel<<<anchor_blocks, 256>>>(
            context.d_x, context.d_anchors, n, l, static_cast<element_type>(restart_prob));
        CHECK_CUDA(cudaGetLastError());
    }
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto upload_end = Clock::now();
    const auto iter_loop_start = Clock::now();
    long long iter_gemm_time = 0;
    long long iter_add_time = 0;
    long long iter_swap_time = 0;
#endif
    int iterations_run = 0;
    const element_type alpha = static_cast<element_type>(1.0);
    const element_type beta = static_cast<element_type>(0.0);

    for (int i = 0; i < total_updates; ++i) {
#ifdef DEBUG
        const auto gemm_start = Clock::now();
#endif
        cuda_sparse_spmm(context.cusparse,
                         graph,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         context.d_x,
                         n,
                         l,
                         n,
                         CUSPARSE_ORDER_COL,
                         beta,
                         context.d_xnext,
                         n,
                         l,
                         n,
                         CUSPARSE_ORDER_COL,
                         alpha);

#ifdef DEBUG
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto gemm_end = Clock::now();
        iter_gemm_time += elapsed_ms(gemm_start, gemm_end);
        const auto add_start = gemm_end;
#endif
        if (anchor_blocks > 0) {
            add_restart_entries_kernel<<<anchor_blocks, 256>>>(
                context.d_xnext, context.d_anchors, n, l, static_cast<element_type>(restart_prob));
            CHECK_CUDA(cudaGetLastError());
        }

#ifdef DEBUG
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto add_end = Clock::now();
        iter_add_time += elapsed_ms(add_start, add_end);
#endif
        ++iterations_run;

#ifdef DEBUG
        const auto swap_start = add_end;
#endif
        std::swap(context.d_x, context.d_xnext);
#ifdef DEBUG
        const auto swap_end = Clock::now();
        iter_swap_time += elapsed_ms(swap_start, swap_end);
#endif
    }
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto iter_loop_end = Clock::now();
    const auto output_copy_start = Clock::now();
#endif
    if (dense_elems > 0) {
        CHECK_CUDA(cudaMemcpy(d_X_out, context.d_x, dense_elems * sizeof(element_type), cudaMemcpyDeviceToDevice));
    }
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto output_copy_end = Clock::now();

    DEBUG_ONLY(std::cout
        << "batch RWR call " << call_id
        << " (n=" << n << ", l=" << l << ")\n"
        << "  seed build: " << elapsed_ms(seed_build_start, seed_build_end) << " ms\n"
        << "  context setup: " << elapsed_ms(context_setup_start, context_setup_end) << " ms"
        << " [handles=" << (reused_handles ? "reused" : "new")
        << ", dense_capacity=" << (reused_dense_capacity ? "reused" : "resized")
        << ", dense_shape=" << (reused_dense_shape ? "reused" : "rebuilt")
        << ", anchor_capacity=" << (reused_anchor_capacity ? "reused" : "resized") << "]\n"
        << "  seed uploads: " << elapsed_ms(upload_start, upload_end) << " ms\n"
        << "  dense order: col_major\n"
        << "  iteration loop: " << elapsed_ms(iter_loop_start, iter_loop_end) << " ms\n"
        << "    spmm: " << iter_gemm_time << " ms\n"
        << "    add restart: " << iter_add_time << " ms\n"
        << "    swap: " << iter_swap_time << " ms\n"
        << "    iterations executed: " << iterations_run << " / " << total_updates << "\n"
        << "  output copy: " << elapsed_ms(output_copy_start, output_copy_end) << " ms\n"
        << "  total inside function: " << elapsed_ms(seed_build_start, output_copy_end) << " ms"
        << std::endl);
#endif
}

void destroy_batch_rwr_context(BatchRwrContext& context) {
    reset_batch_rwr_dense_mats(context);
    if (context.cublas != nullptr) {
        CHECK_CUBLAS(cublasDestroy(context.cublas));
        context.cublas = nullptr;
    }
    if (context.cusparse != nullptr) {
        CHECK_CUSPARSE(cusparseDestroy(context.cusparse));
        context.cusparse = nullptr;
    }
    if (context.d_x != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_x));
        context.d_x = nullptr;
    }
    if (context.d_xnext != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_xnext));
        context.d_xnext = nullptr;
    }
    if (context.d_anchors != nullptr) {
        CHECK_CUDA(CUDA_FREE(context.d_anchors));
        context.d_anchors = nullptr;
    }
    context.dense_capacity = 0;
    context.anchor_capacity = 0;
}

void get_product_rwr_mat_sparse_device(const CudaSparseMatrix& graph1_normalized,
                                      const CudaSparseMatrix& graph2_normalized,
                                      const element_type* d_in_cross_dist,
                                      int n1, int n2,
                                      element_type rwr_restart_prob, element_type gamma,
                                      element_type* d_out_cross_dist_result,
                                      int max_iters, element_type tol) {
    const size_t N = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    DEBUG_ONLY(std::cout << "cross propagation dense order: col_major, graph multiply: csr_spmm" << std::endl);

    const bool timing_enabled = cross_timing_enabled();
    const auto total_start = cross_timing_now(timing_enabled);
    long long allocation_ms = 0;
    long long setup_ms = 0;
    long long loop_spmm1_ms = 0;
    long long loop_spmm2_ms = 0;
    long long loop_restart_ms = 0;
    long long loop_diff_ms = 0;
    long long loop_swap_ms = 0;
    long long final_copy_ms = 0;
    long long cleanup_ms = 0;
    int iterations_run = 0;
    auto section_start = cross_timing_now(timing_enabled);

    element_type* d_out_cross_dist_old = d_out_cross_dist_result;
    element_type* d_out_cross_dist = nullptr;
    element_type* d_out_cross_dist_work = nullptr;
    element_type* d_temp = nullptr;
    element_type* d_block_max = nullptr;

    CHECK_CUDA(CUDA_MALLOC((void**)&d_out_cross_dist_work, N * sizeof(element_type)));
    d_out_cross_dist = d_out_cross_dist_work;
    CHECK_CUDA(CUDA_MALLOC((void**)&d_temp, N * sizeof(element_type)));
    const unsigned int diff_blocks = static_cast<unsigned int>((N + 255) / 256);
    CHECK_CUDA(CUDA_MALLOC((void**)&d_block_max, diff_blocks * sizeof(element_type)));

    CHECK_CUDA(cudaMemset(d_out_cross_dist_old, 0, N * sizeof(element_type)));
    CHECK_CUDA(cudaMemset(d_temp, 0, N * sizeof(element_type)));
    allocation_ms += cross_timing_elapsed_ms(timing_enabled, section_start);

    section_start = cross_timing_now(timing_enabled);
    cublasHandle_t blas_handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&blas_handle));
    cusparseHandle_t sparse_handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&sparse_handle));
    setup_ms += cross_timing_elapsed_ms(timing_enabled, section_start);

    element_type* d_out_cross_dist_final = d_out_cross_dist_old;
    for (int i = 0; i < max_iters; ++i) {
        section_start = cross_timing_now(timing_enabled);
        {
            const element_type alpha = static_cast<element_type>(1.0);
            const element_type beta = static_cast<element_type>(0.0);
            cuda_sparse_spmm(sparse_handle,
                             graph2_normalized,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             d_out_cross_dist_old,
                             n2,
                             n1,
                             n1,
                             CUSPARSE_ORDER_ROW,
                             beta,
                             d_temp,
                             n2,
                             n1,
                             n1,
                             CUSPARSE_ORDER_ROW,
                             alpha);
        }
        loop_spmm1_ms += cross_timing_elapsed_ms(timing_enabled, section_start);

        section_start = cross_timing_now(timing_enabled);
        {
            const element_type alpha = (static_cast<element_type>(1.0) - rwr_restart_prob) * gamma;
            const element_type beta = static_cast<element_type>(0.0);
            cuda_sparse_spmm(sparse_handle,
                             graph1_normalized,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             d_temp,
                             n1,
                             n2,
                             n1,
                             CUSPARSE_ORDER_COL,
                             beta,
                             d_out_cross_dist,
                             n1,
                             n2,
                             n1,
                             CUSPARSE_ORDER_COL,
                             alpha);
        }
        loop_spmm2_ms += cross_timing_elapsed_ms(timing_enabled, section_start);

        section_start = cross_timing_now(timing_enabled);
        const element_type restart_scale = static_cast<element_type>(1.0) + rwr_restart_prob;
        CHECK_CUBLAS(cuda_cublas_axpy(blas_handle, static_cast<int>(N), &restart_scale, d_in_cross_dist, 1, d_out_cross_dist, 1));
        loop_restart_ms += cross_timing_elapsed_ms(timing_enabled, section_start);

        section_start = cross_timing_now(timing_enabled);
        abs_diff_block_max_kernel<<<diff_blocks, 256>>>(d_out_cross_dist, d_out_cross_dist_old, N, d_block_max);
        CHECK_CUDA(cudaGetLastError());
        reduce_block_max_to_scalar_kernel<<<1, 256>>>(d_block_max, diff_blocks);
        CHECK_CUDA(cudaGetLastError());
        element_type max_diff = static_cast<element_type>(0.0);
        CHECK_CUDA(cudaMemcpy(&max_diff, d_block_max, sizeof(element_type), cudaMemcpyDeviceToHost));
        loop_diff_ms += cross_timing_elapsed_ms(timing_enabled, section_start);
        d_out_cross_dist_final = d_out_cross_dist;
        ++iterations_run;
        if (max_diff < tol) {
            break;
        }

        section_start = cross_timing_now(timing_enabled);
        std::swap(d_out_cross_dist_old, d_out_cross_dist);
        d_out_cross_dist_final = d_out_cross_dist_old;
        loop_swap_ms += cross_timing_elapsed_ms(timing_enabled, section_start);
    }
    section_start = cross_timing_now(timing_enabled);
    const element_type scale = static_cast<element_type>(1.0) - gamma;
    CHECK_CUBLAS(cuda_cublas_scal(blas_handle, static_cast<int>(N), &scale, d_out_cross_dist_final, 1));
    if (d_out_cross_dist_final != d_out_cross_dist_result) {
        CHECK_CUDA(cudaMemcpy(d_out_cross_dist_result, d_out_cross_dist_final, N * sizeof(element_type), cudaMemcpyDeviceToDevice));
    }
    final_copy_ms += cross_timing_elapsed_ms(timing_enabled, section_start);

    section_start = cross_timing_now(timing_enabled);
    CHECK_CUSPARSE(cusparseDestroy(sparse_handle));
    CHECK_CUBLAS(cublasDestroy(blas_handle));
    CHECK_CUDA(CUDA_FREE(d_temp));
    CHECK_CUDA(CUDA_FREE(d_out_cross_dist_work));
    CHECK_CUDA(CUDA_FREE(d_block_max));
    cleanup_ms += cross_timing_elapsed_ms(timing_enabled, section_start);

    if (timing_enabled) {
        const long long total_ms = cross_timing_elapsed_ms(timing_enabled, total_start);
        std::cout << "cross propagation breakdown (csr_spmm):\n"
                  << "  allocation/init: " << allocation_ms << " ms\n"
                  << "  handle setup: " << setup_ms << " ms\n"
                  << "  loop total: "
                  << (loop_spmm1_ms + loop_spmm2_ms + loop_restart_ms + loop_diff_ms + loop_swap_ms)
                  << " ms\n"
                  << "    graph2 spmm: " << loop_spmm1_ms << " ms\n"
                  << "    graph1 spmm: " << loop_spmm2_ms << " ms\n"
                  << "    restart axpy: " << loop_restart_ms << " ms\n"
                  << "    convergence diff: " << loop_diff_ms << " ms\n"
                  << "    swap: " << loop_swap_ms << " ms\n"
                  << "    iterations executed: " << iterations_run << " / " << max_iters << "\n"
                  << "  final scale/copy: " << final_copy_ms << " ms\n"
                  << "  cleanup: " << cleanup_ms << " ms\n"
                  << "  total inside cross propagation: " << total_ms << " ms"
                  << std::endl;
    }
}

#include "cuda_helper.cuh"

namespace {

struct BatchRwrSpmmConfig {
    cusparseSpMMAlg_t alg;
    const char* name;
    const char* source;
    bool preprocess_expected_to_help;
};

struct BatchRwrEarlyStopConfig {
    bool enabled;
    double tol;
    int min_iters;
    int check_every;
    bool use_row_l2_normalized_diff;
    const char* mode_name;
    const char* source;
};

struct BatchRwrRestartConfig {
    bool use_sparse_restart;
    const char* mode_name;
    const char* source;
};

BatchRwrSpmmConfig get_batch_rwr_spmm_config() {
    const char* env_value = std::getenv("PARROT_RWR_SPMM_ALG");
    if (env_value == nullptr) {
        return {CUSPARSE_SPMM_ALG_DEFAULT, "default", "builtin", false};
    }

    const std::string requested(env_value);
    if (requested == "default") {
        return {CUSPARSE_SPMM_ALG_DEFAULT, "default", "env", false};
    }
    if (requested == "csr1" || requested == "csr_alg1") {
        return {CUSPARSE_SPMM_CSR_ALG1, "csr_alg1", "env", true};
    }
    if (requested == "csr2" || requested == "csr_alg2") {
        return {CUSPARSE_SPMM_CSR_ALG2, "csr_alg2", "env", false};
    }
    if (requested == "csr3" || requested == "csr_alg3") {
        return {CUSPARSE_SPMM_CSR_ALG3, "csr_alg3", "env", true};
    }

#ifdef DEBUG
    std::cerr << "unknown PARROT_RWR_SPMM_ALG='" << requested
              << "', falling back to default" << std::endl;
#endif
    return {CUSPARSE_SPMM_ALG_DEFAULT, "default", "fallback", false};
}

BatchRwrRestartConfig get_batch_rwr_restart_config() {
    const char* mode_env = std::getenv("PARROT_RWR_RESTART_MODE");
    if (mode_env == nullptr) {
        return {true, "sparse", "builtin"};
    }

    const std::string mode_value(mode_env);
    if (mode_value == "sparse") {
        return {true, "sparse", "env"};
    }
    if (mode_value == "dense") {
        return {false, "dense", "env"};
    }

#ifdef DEBUG
    std::cerr << "unknown PARROT_RWR_RESTART_MODE='" << mode_value
              << "', falling back to sparse" << std::endl;
#endif
    return {true, "sparse", "fallback"};
}

BatchRwrEarlyStopConfig get_batch_rwr_early_stop_config() {
    const char* enabled_env = std::getenv("PARROT_RWR_EARLY_STOP");
    if (enabled_env == nullptr) {
        return {false, 1e-6, 10, 5, true, "row_l2_normalized", "builtin"};
    }

    const std::string enabled_value(enabled_env);
    const bool enabled = enabled_value == "1" || enabled_value == "true" || enabled_value == "on";
    if (!enabled) {
        return {false, 1e-6, 10, 5, true, "row_l2_normalized", "env"};
    }

    double tol = 1e-6;
    int min_iters = 10;
    int check_every = 5;
    bool use_row_l2_normalized_diff = true;
    const char* mode_name = "row_l2_normalized";

    if (const char* tol_env = std::getenv("PARROT_RWR_EARLY_STOP_TOL")) {
        char* end = nullptr;
        const double parsed = std::strtod(tol_env, &end);
        if (end != tol_env && parsed > 0.0) {
            tol = parsed;
        }
    }
    if (const char* min_iters_env = std::getenv("PARROT_RWR_EARLY_STOP_MIN_ITERS")) {
        char* end = nullptr;
        const long parsed = std::strtol(min_iters_env, &end, 10);
        if (end != min_iters_env && parsed > 0) {
            min_iters = static_cast<int>(parsed);
        }
    }
    if (const char* check_every_env = std::getenv("PARROT_RWR_EARLY_STOP_CHECK_EVERY")) {
        char* end = nullptr;
        const long parsed = std::strtol(check_every_env, &end, 10);
        if (end != check_every_env && parsed > 0) {
            check_every = static_cast<int>(parsed);
        }
    }

    if (const char* mode_env = std::getenv("PARROT_RWR_EARLY_STOP_MODE")) {
        const std::string mode_value(mode_env);
        if (mode_value == "raw") {
            use_row_l2_normalized_diff = false;
            mode_name = "raw";
        } else if (mode_value == "row_l2_normalized" ||
                   mode_value == "normalized" ||
                   mode_value == "row_norm") {
            use_row_l2_normalized_diff = true;
            mode_name = "row_l2_normalized";
        }
    }

    return {true, tol, min_iters, check_every, use_row_l2_normalized_diff, mode_name, "env"};
}

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

__global__ void write_restart_entries_kernel(double* e,
                                             const int* anchors,
                                             int n,
                                             int l,
                                             double restart_prob) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= l) {
        return;
    }

    const size_t offset = static_cast<size_t>(anchors[j]) + static_cast<size_t>(j) * static_cast<size_t>(n);
    e[offset] = restart_prob;
}

__global__ void add_restart_entries_kernel(double* x,
                                           const int* anchors,
                                           int n,
                                           int l,
                                           double restart_prob) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= l) {
        return;
    }

    const size_t offset = static_cast<size_t>(anchors[j]) + static_cast<size_t>(j) * static_cast<size_t>(n);
    x[offset] += restart_prob;
}

__global__ void compute_row_inv_l2_norms_kernel(const double* matrix,
                                                size_t n,
                                                size_t d,
                                                double* inv_norms) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n) {
        return;
    }

    __shared__ double partial[256];
    double sum = 0.0;
    for (size_t col = static_cast<size_t>(threadIdx.x); col < d; col += blockDim.x) {
        const double value = matrix[row + col * n];
        sum += value * value;
    }

    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        inv_norms[row] = partial[0] > 0.0 ? 1.0 / std::sqrt(partial[0]) : 0.0;
    }
}

__global__ void normalized_abs_diff_block_max_kernel(const double* a,
                                                     const double* b,
                                                     const double* inv_norms_a,
                                                     const double* inv_norms_b,
                                                     size_t n,
                                                     size_t total,
                                                     double* block_max) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double local_max = 0.0;
    if (idx < total) {
        const size_t row = idx % n;
        const double a_normalized = a[idx] * inv_norms_a[row];
        const double b_normalized = b[idx] * inv_norms_b[row];
        local_max = fabs(a_normalized - b_normalized);
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

void ensure_batch_rwr_anchor_capacity(BatchRwrContext& context, size_t required_anchors) {
    if (context.anchor_capacity >= required_anchors) {
        return;
    }
    if (context.d_anchors != nullptr) {
        CHECK_CUDA(cudaFree(context.d_anchors));
        context.d_anchors = nullptr;
        context.anchor_capacity = 0;
    }
    if (required_anchors > 0) {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&context.d_anchors), required_anchors * sizeof(int)));
        context.anchor_capacity = required_anchors;
    }
}

void ensure_batch_rwr_diff_capacity(BatchRwrContext& context, size_t required_blocks) {
    if (context.diff_capacity >= required_blocks) {
        return;
    }
    if (context.d_block_max != nullptr) {
        CHECK_CUDA(cudaFree(context.d_block_max));
        context.d_block_max = nullptr;
        context.diff_capacity = 0;
    }
    if (required_blocks > 0) {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&context.d_block_max), required_blocks * sizeof(double)));
        context.diff_capacity = required_blocks;
    }
}

void ensure_batch_rwr_row_norm_capacity(BatchRwrContext& context, size_t required_rows) {
    if (context.row_norm_capacity >= required_rows) {
        return;
    }
    if (context.d_row_inv_norms_x != nullptr) {
        CHECK_CUDA(cudaFree(context.d_row_inv_norms_x));
        context.d_row_inv_norms_x = nullptr;
    }
    if (context.d_row_inv_norms_xnext != nullptr) {
        CHECK_CUDA(cudaFree(context.d_row_inv_norms_xnext));
        context.d_row_inv_norms_xnext = nullptr;
    }
    context.row_norm_capacity = 0;
    if (required_rows > 0) {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&context.d_row_inv_norms_x), required_rows * sizeof(double)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&context.d_row_inv_norms_xnext), required_rows * sizeof(double)));
        context.row_norm_capacity = required_rows;
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
    const int total_updates = std::max(0, max_iters - 1);
    const unsigned int anchor_blocks = static_cast<unsigned int>((static_cast<unsigned int>(l) + 255U) / 256U);
    const unsigned int diff_blocks = static_cast<unsigned int>((dense_elems + 255U) / 256U);
    const BatchRwrRestartConfig restart_config = get_batch_rwr_restart_config();

#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());

    const auto seed_build_start = Clock::now();
#endif
#ifdef DEBUG
    const auto seed_build_end = Clock::now();

    const bool reused_handles = context.cusparse != nullptr && context.cublas != nullptr;
    const bool reused_dense_capacity = context.dense_capacity >= dense_elems;
    const bool reused_dense_shape = context.mat_x != nullptr && context.mat_xnext != nullptr &&
                                    context.current_n == n && context.current_l == l;
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
    if (restart_config.use_sparse_restart) {
        CHECK_CUDA(cudaMemset(context.d_x, 0, dense_elems * sizeof(double)));
        if (anchor_blocks > 0) {
            write_restart_entries_kernel<<<anchor_blocks, 256>>>(
                context.d_x, context.d_anchors, n, l, static_cast<double>(restart_prob));
            CHECK_CUDA(cudaGetLastError());
        }
    } else {
        CHECK_CUDA(cudaMemset(context.d_e, 0, dense_elems * sizeof(double)));
        if (anchor_blocks > 0) {
            write_restart_entries_kernel<<<anchor_blocks, 256>>>(
                context.d_e, context.d_anchors, n, l, static_cast<double>(restart_prob));
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaMemcpy(context.d_x, context.d_e, dense_elems * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA(cudaMemset(context.d_xnext, 0, dense_elems * sizeof(double)));
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
    const BatchRwrSpmmConfig spmm_config = get_batch_rwr_spmm_config();
    const BatchRwrEarlyStopConfig early_stop_config = get_batch_rwr_early_stop_config();
    const bool enable_convergence_checks =
        early_stop_config.enabled && total_updates > 0 && diff_blocks > 0;

    size_t bufferSize = 0;
#ifdef DEBUG
    const auto preprocess_start = Clock::now();
#endif
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        context.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        context.mat_a,
        context.mat_x,
        &beta,
        context.mat_xnext,
        CUDA_R_64F,
        spmm_config.alg,
        &bufferSize));

#ifdef DEBUG
    const bool reused_workspace = context.buffer_capacity >= bufferSize;
#endif
    ensure_batch_rwr_workspace(context, bufferSize);
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
        context.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        context.mat_a,
        context.mat_x,
        &beta,
        context.mat_xnext,
        CUDA_R_64F,
        spmm_config.alg,
        context.d_buffer));
#ifdef DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto preprocess_end = Clock::now();

    const auto iter_loop_start = Clock::now();
    long long iter_spmm_time = 0;
    long long iter_add_time = 0;
    long long iter_swap_time = 0;
    long long iter_convergence_time = 0;
#endif
    if (enable_convergence_checks) {
        ensure_batch_rwr_diff_capacity(context, diff_blocks);
    }

    double* d_final = context.d_x;
    int iterations_run = 0;
    int convergence_checks = 0;
    double last_checked_max_diff = -1.0;

    for (int i = 0; i < total_updates; ++i) {
#ifdef DEBUG
        const auto spmm_start = Clock::now();
#endif
        CHECK_CUSPARSE(cusparseSpMM(
            context.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            context.mat_a,
            context.mat_x,
            &beta,
            context.mat_xnext,
            CUDA_R_64F,
            spmm_config.alg,
            context.d_buffer));

        double one = 1.0;
#ifdef DEBUG
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto spmm_end = Clock::now();
        iter_spmm_time += elapsed_ms(spmm_start, spmm_end);

        const auto add_start = spmm_end;
#endif
        if (restart_config.use_sparse_restart) {
            if (anchor_blocks > 0) {
                add_restart_entries_kernel<<<anchor_blocks, 256>>>(
                    context.d_xnext, context.d_anchors, n, l, static_cast<double>(restart_prob));
                CHECK_CUDA(cudaGetLastError());
            }
        } else {
            CHECK_CUBLAS(cublasDaxpy(
                context.cublas,
                n * l,
                &one,
                context.d_e,
                1,
                context.d_xnext,
                1));
        }

#ifdef DEBUG
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto add_end = Clock::now();
        iter_add_time += elapsed_ms(add_start, add_end);
#endif
        ++iterations_run;
        bool converged = false;
        const bool should_check_convergence =
            enable_convergence_checks &&
            iterations_run >= early_stop_config.min_iters &&
            ((iterations_run % early_stop_config.check_every) == 0 || i == total_updates - 1);

        if (should_check_convergence) {
#ifdef DEBUG
            const auto convergence_start = add_end;
#endif
            if (early_stop_config.use_row_l2_normalized_diff) {
                ensure_batch_rwr_row_norm_capacity(context, static_cast<size_t>(n));
                compute_row_inv_l2_norms_kernel<<<static_cast<unsigned int>(n), 256>>>(
                    context.d_xnext, static_cast<size_t>(n), static_cast<size_t>(l), context.d_row_inv_norms_xnext);
                compute_row_inv_l2_norms_kernel<<<static_cast<unsigned int>(n), 256>>>(
                    context.d_x, static_cast<size_t>(n), static_cast<size_t>(l), context.d_row_inv_norms_x);
                normalized_abs_diff_block_max_kernel<<<diff_blocks, 256>>>(
                    context.d_xnext,
                    context.d_x,
                    context.d_row_inv_norms_xnext,
                    context.d_row_inv_norms_x,
                    static_cast<size_t>(n),
                    dense_elems,
                    context.d_block_max);
            } else {
                abs_diff_block_max_kernel<<<diff_blocks, 256>>>(
                    context.d_xnext, context.d_x, dense_elems, context.d_block_max);
            }
            CHECK_CUDA(cudaGetLastError());
            int max_idx = 0;
            last_checked_max_diff = 0.0;
            CHECK_CUBLAS(cublasIdamax(context.cublas, static_cast<int>(diff_blocks), context.d_block_max, 1, &max_idx));
            if (max_idx > 0) {
                CHECK_CUDA(cudaMemcpy(&last_checked_max_diff,
                                      context.d_block_max + (max_idx - 1),
                                      sizeof(double),
                                      cudaMemcpyDeviceToHost));
            }
            ++convergence_checks;
            converged = last_checked_max_diff < early_stop_config.tol;
#ifdef DEBUG
            CHECK_CUDA(cudaDeviceSynchronize());
            const auto convergence_end = Clock::now();
            iter_convergence_time += elapsed_ms(convergence_start, convergence_end);
#endif
        }

        if (converged) {
            d_final = context.d_xnext;
            break;
        }

#ifdef DEBUG
        const auto swap_start =
            should_check_convergence ? Clock::now() : add_end;
#endif
        std::swap(context.d_x, context.d_xnext);
        CHECK_CUSPARSE(cusparseDnMatSetValues(context.mat_x, context.d_x));
        CHECK_CUSPARSE(cusparseDnMatSetValues(context.mat_xnext, context.d_xnext));
        d_final = context.d_x;
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
    CHECK_CUDA(cudaMemcpy(d_X_out, d_final, dense_elems * sizeof(double), cudaMemcpyDeviceToDevice));
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
        << ", dense_views=" << (reused_dense_shape ? "reused" : "rebuilt")
        << ", anchor_capacity=" << (reused_anchor_capacity ? "reused" : "resized") << "]\n"
        << "  seed uploads: " << elapsed_ms(upload_start, upload_end) << " ms\n"
        << "  sparse descriptor setup: " << elapsed_ms(spmat_setup_start, spmat_setup_end) << " ms"
        << " [mat_a=" << (recreated_mat_a ? "recreated" : "created") << "]\n"
        << "  spmm preprocess: " << elapsed_ms(preprocess_start, preprocess_end) << " ms"
        << " [workspace=" << (reused_workspace ? "reused" : "resized") << ", bytes=" << bufferSize << "]\n"
        << "  spmm alg: " << spmm_config.name
        << " [source=" << spmm_config.source
        << ", preprocess_hint=" << (spmm_config.preprocess_expected_to_help ? "expected_helpful" : "unknown_or_no_effect") << "]\n"
        << "  restart mode: " << restart_config.mode_name
        << " [source=" << restart_config.source << "]\n"
        << "  early stop: " << (early_stop_config.enabled ? "enabled" : "disabled")
        << " [source=" << early_stop_config.source
        << ", tol=" << early_stop_config.tol
        << ", min_iters=" << early_stop_config.min_iters
        << ", check_every=" << early_stop_config.check_every
        << ", mode=" << early_stop_config.mode_name << "]\n"
        << "  iteration loop: " << elapsed_ms(iter_loop_start, iter_loop_end) << " ms\n"
        << "    spmm: " << iter_spmm_time << " ms\n"
        << "    add E: " << iter_add_time << " ms\n"
        << "    convergence check: " << iter_convergence_time << " ms\n"
        << "    swap/rebind: " << iter_swap_time << " ms\n"
        << "    iterations executed: " << iterations_run << " / " << total_updates << "\n"
        << "    convergence checks: " << convergence_checks
        << (enable_convergence_checks ? "" : " (disabled)")
        << "\n"
        << "    last checked max diff: " << last_checked_max_diff
        << (enable_convergence_checks && last_checked_max_diff >= 0.0 && last_checked_max_diff < early_stop_config.tol ? " (stopped early)" : "")
        << "\n"
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
    if (context.d_anchors != nullptr) {
        CHECK_CUDA(cudaFree(context.d_anchors));
        context.d_anchors = nullptr;
    }
    if (context.d_block_max != nullptr) {
        CHECK_CUDA(cudaFree(context.d_block_max));
        context.d_block_max = nullptr;
    }
    if (context.d_row_inv_norms_x != nullptr) {
        CHECK_CUDA(cudaFree(context.d_row_inv_norms_x));
        context.d_row_inv_norms_x = nullptr;
    }
    if (context.d_row_inv_norms_xnext != nullptr) {
        CHECK_CUDA(cudaFree(context.d_row_inv_norms_xnext));
        context.d_row_inv_norms_xnext = nullptr;
    }
    if (context.d_buffer != nullptr) {
        CHECK_CUDA(cudaFree(context.d_buffer));
        context.d_buffer = nullptr;
    }
    context.dense_capacity = 0;
    context.anchor_capacity = 0;
    context.diff_capacity = 0;
    context.row_norm_capacity = 0;
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

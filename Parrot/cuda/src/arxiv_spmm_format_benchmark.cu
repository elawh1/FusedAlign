#include "cuda_helper.cuh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kVerifyRelTol = CUDA_USE_FLOAT ? 1e-5 : 1e-10;
constexpr double kVerifyAbsTol = CUDA_USE_FLOAT ? 1e-6 : 1e-12;

struct Timer {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    Timer() {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
    }

    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() {
        CHECK_CUDA(cudaEventRecord(start));
    }

    float end() {
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

struct BellMatrix {
    int rows = 0;
    int cols = 0;
    int block_size = 0;
    int ell_cols = 0;
    int blocks_per_row = 0;
    std::vector<int> col_ind;
    std::vector<element_type> vals;
};

struct BenchResult {
    double ms = 0.0;
    size_t workspace_bytes = 0;
};

void usage(const char* argv0) {
    std::printf("Usage: %s <arxiv_data_dir> [K=64] [niter=20] [graph=1] [max_ell_mb=8192]\n", argv0);
    std::printf("  Loads graph1.txt or graph2.txt using config.txt, row-normalizes, transposes,\n");
    std::printf("  then benchmarks cuSPARSE SpMM in CSR and Blocked-ELL formats.\n");
    std::printf("  Set max_ell_mb=0 to disable the Blocked-ELL sparse-storage guard.\n");
}

CudaCsrMatrix load_graph_like_main(const std::string& data_path, int graph_id, int& n_out) {
    int g1_num_nodes = 0;
    int g2_num_nodes = 0;
    bool use_attr = false;
    bool directed = false;
    int inner_iters = 0;
    int outer_iters = 0;
    load_config(data_path, g1_num_nodes, g2_num_nodes, use_attr, directed, inner_iters, outer_iters);

    const int n = graph_id == 2 ? g2_num_nodes : g1_num_nodes;
    const std::string graph_file = data_path + (graph_id == 2 ? "/graph2.txt" : "/graph1.txt");
    std::vector<element_type> edgelist = readFileToVector(graph_file, directed);
    const int avg_nnz_per_n = n > 0 ? static_cast<int>(edgelist.size()) / n : 0;

    std::printf("Loading %s (nodes=%d directed=%d element_type=%s)\n",
                graph_file.c_str(), n, static_cast<int>(directed), CUDA_ELEMENT_TYPE_NAME);

    CudaCsrMatrix graph = make_cuda_csr_from_edgelist(edgelist, n, n);
    CudaCsrMatrix normalized = row_normalize_graph_cuda(graph, avg_nnz_per_n);
    CudaCsrMatrix normalized_t = transpose_cuda_csr(normalized);
    destroy_cuda_csr(normalized);
    destroy_cuda_csr(graph);

    n_out = n;
    return normalized_t;
}

void fill_dense(std::vector<element_type>& h_dense) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (element_type& x : h_dense) {
        x = static_cast<element_type>(dist(rng));
    }
}

void copy_graph_to_host(const CudaCsrMatrix& graph,
                        std::vector<int>& h_row_offsets,
                        std::vector<int>& h_col_ind,
                        std::vector<element_type>& h_vals) {
    h_row_offsets.resize(static_cast<size_t>(graph.rows) + 1);
    h_col_ind.resize(static_cast<size_t>(graph.nnz));
    h_vals.resize(static_cast<size_t>(graph.nnz));
    CHECK_CUDA(cudaMemcpy(h_row_offsets.data(),
                          graph.d_row_offsets,
                          (static_cast<size_t>(graph.rows) + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_col_ind.data(),
                          graph.d_col_ind,
                          static_cast<size_t>(graph.nnz) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vals.data(),
                          graph.d_vals,
                          static_cast<size_t>(graph.nnz) * sizeof(element_type),
                          cudaMemcpyDeviceToHost));
}

BellMatrix csr_to_blocked_ell(int n,
                              int block_size,
                              const std::vector<int>& row_offsets,
                              const std::vector<int>& col_ind,
                              const std::vector<element_type>& vals) {
    BellMatrix bell;
    bell.rows = ((n + block_size - 1) / block_size) * block_size;
    bell.cols = bell.rows;
    bell.block_size = block_size;

    const int block_rows = bell.rows / block_size;
    int max_blocks_per_row = 0;
    std::vector<std::map<int, std::vector<element_type>>> rows(static_cast<size_t>(block_rows));

    for (int br = 0; br < block_rows; ++br) {
        const int row_begin = br * block_size;
        const int row_end = std::min(row_begin + block_size, n);
        auto& block_map = rows[static_cast<size_t>(br)];
        for (int row = row_begin; row < row_end; ++row) {
            const int local_row = row - row_begin;
            for (int p = row_offsets[static_cast<size_t>(row)];
                 p < row_offsets[static_cast<size_t>(row) + 1]; ++p) {
                const int col = col_ind[static_cast<size_t>(p)];
                const int block_col = col / block_size;
                const int local_col = col % block_size;
                auto& block = block_map[block_col];
                if (block.empty()) {
                    block.assign(static_cast<size_t>(block_size) * block_size,
                                 static_cast<element_type>(0));
                }
                block[static_cast<size_t>(local_row) * block_size + local_col] =
                    vals[static_cast<size_t>(p)];
            }
        }
        max_blocks_per_row = std::max(max_blocks_per_row, static_cast<int>(block_map.size()));
    }

    bell.blocks_per_row = std::max(1, max_blocks_per_row);
    bell.ell_cols = bell.blocks_per_row * block_size;
    bell.col_ind.assign(static_cast<size_t>(block_rows) * bell.blocks_per_row, -1);
    bell.vals.assign(static_cast<size_t>(bell.rows) * bell.ell_cols,
                     static_cast<element_type>(0));

    for (int br = 0; br < block_rows; ++br) {
        int slot = 0;
        for (const auto& entry : rows[static_cast<size_t>(br)]) {
            bell.col_ind[static_cast<size_t>(slot) * block_rows + br] = entry.first;
            for (int local_row = 0; local_row < block_size; ++local_row) {
                const int row = br * block_size + local_row;
                for (int local_col = 0; local_col < block_size; ++local_col) {
                    bell.vals[static_cast<size_t>(row) * bell.ell_cols +
                              slot * block_size + local_col] =
                        entry.second[static_cast<size_t>(local_row) * block_size + local_col];
                }
            }
            ++slot;
        }
    }

    return bell;
}

BenchResult bench_csr(cusparseHandle_t handle,
                      const CudaCsrMatrix& graph,
                      cusparseDnMatDescr_t mat_b,
                      cusparseDnMatDescr_t mat_c,
                      cusparseSpMMAlg_t alg,
                      int nwarm,
                      int niter) {
    CudaSpMatView mat_a = create_spmm_matrix_view(graph);
    const element_type alpha = static_cast<element_type>(1);
    const element_type beta = static_cast<element_type>(0);
    size_t buffer_size = 0;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           mat_a.descr,
                                           mat_b,
                                           &beta,
                                           mat_c,
                                           CUDA_R_TYPE,
                                           alg,
                                           &buffer_size));

    void* d_buffer = nullptr;
    if (buffer_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));
    }

    CHECK_CUSPARSE(cusparseSpMM_preprocess(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           mat_a.descr,
                                           mat_b,
                                           &beta,
                                           mat_c,
                                           CUDA_R_TYPE,
                                           alg,
                                           d_buffer));

    auto run = [&]() {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    mat_a.descr,
                                    mat_b,
                                    &beta,
                                    mat_c,
                                    CUDA_R_TYPE,
                                    alg,
                                    d_buffer));
    };

    for (int i = 0; i < nwarm; ++i) {
        run();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    Timer timer;
    timer.begin();
    for (int i = 0; i < niter; ++i) {
        run();
    }
    const float total_ms = timer.end();

    if (d_buffer != nullptr) {
        cudaFree(d_buffer);
    }
    destroy_spmm_matrix_view(mat_a);
    return {static_cast<double>(total_ms) / niter, buffer_size};
}

BenchResult bench_bell(cusparseHandle_t handle,
                       const BellMatrix& bell,
                       int k,
                       int* d_col_ind,
                       element_type* d_vals,
                       element_type* d_b,
                       element_type* d_c,
                       int nwarm,
                       int niter) {
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c = nullptr;

    CHECK_CUSPARSE(cusparseCreateBlockedEll(&mat_a,
                                            bell.rows,
                                            bell.cols,
                                            bell.block_size,
                                            bell.ell_cols,
                                            d_col_ind,
                                            d_vals,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            CUDA_R_TYPE));
    CHECK_CUSPARSE(cusparseCreateDnMat(&mat_b,
                                       bell.rows,
                                       k,
                                       bell.rows,
                                       d_b,
                                       CUDA_R_TYPE,
                                       CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&mat_c,
                                       bell.rows,
                                       k,
                                       bell.rows,
                                       d_c,
                                       CUDA_R_TYPE,
                                       CUSPARSE_ORDER_COL));

    const element_type alpha = static_cast<element_type>(1);
    const element_type beta = static_cast<element_type>(0);
    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           mat_a,
                                           mat_b,
                                           &beta,
                                           mat_c,
                                           CUDA_R_TYPE,
                                           CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
                                           &buffer_size));

    void* d_buffer = nullptr;
    if (buffer_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));
    }

    auto run = [&]() {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    mat_a,
                                    mat_b,
                                    &beta,
                                    mat_c,
                                    CUDA_R_TYPE,
                                    CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
                                    d_buffer));
    };

    for (int i = 0; i < nwarm; ++i) {
        run();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    Timer timer;
    timer.begin();
    for (int i = 0; i < niter; ++i) {
        run();
    }
    const float total_ms = timer.end();

    if (d_buffer != nullptr) {
        cudaFree(d_buffer);
    }
    cusparseDestroyDnMat(mat_c);
    cusparseDestroyDnMat(mat_b);
    cusparseDestroySpMat(mat_a);
    return {static_cast<double>(total_ms) / niter, buffer_size};
}

bool verify_padded(const element_type* d_ref,
                   int n_ref,
                   const element_type* d_cmp,
                   int n_pad,
                   int k,
                   double rel_tol,
                   double abs_tol) {
    std::vector<element_type> h_ref(static_cast<size_t>(n_ref) * k);
    std::vector<element_type> h_cmp(static_cast<size_t>(n_pad) * k);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(element_type),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cmp.data(), d_cmp, h_cmp.size() * sizeof(element_type),
                          cudaMemcpyDeviceToHost));

    double max_rel = 0.0;
    double max_abs = 0.0;
    int worst_row = 0;
    int worst_col = 0;
    bool ok = true;
    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < n_ref; ++row) {
            const double ref = static_cast<double>(h_ref[static_cast<size_t>(col) * n_ref + row]);
            const double cmp = static_cast<double>(h_cmp[static_cast<size_t>(col) * n_pad + row]);
            const double abs_err = std::abs(ref - cmp);
            const double rel_err = abs_err / std::max(1.0, std::abs(ref));
            if (rel_err > max_rel) {
                max_rel = rel_err;
                max_abs = abs_err;
                worst_row = row;
                worst_col = col;
            }
            if (abs_err > abs_tol + rel_tol * std::abs(ref)) {
                ok = false;
            }
        }
    }

    if (!ok) {
        std::printf("max_rel=%e max_abs=%e at row=%d col=%d ",
                    max_rel, max_abs, worst_row, worst_col);
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const std::string data_path = argv[1];
    const int k = argc > 2 ? std::atoi(argv[2]) : 64;
    const int niter = argc > 3 ? std::atoi(argv[3]) : 20;
    const int graph_id = argc > 4 ? std::atoi(argv[4]) : 1;
    const size_t max_ell_mb = argc > 5 ? static_cast<size_t>(std::atoll(argv[5])) : 8192;
    const size_t max_ell_bytes = max_ell_mb * 1000ULL * 1000ULL;
    const int nwarm = 5;

    if (k <= 0 || niter <= 0 || (graph_id != 1 && graph_id != 2)) {
        usage(argv[0]);
        return 1;
    }

    int n = 0;
    CudaCsrMatrix graph = load_graph_like_main(data_path, graph_id, n);

    std::vector<int> h_row_offsets;
    std::vector<int> h_col_ind;
    std::vector<element_type> h_vals;
    copy_graph_to_host(graph, h_row_offsets, h_col_ind, h_vals);

    const size_t dense_elems = static_cast<size_t>(n) * k;
    std::vector<element_type> h_b(dense_elems);
    fill_dense(h_b);

    element_type* d_b = nullptr;
    element_type* d_c_ref = nullptr;
    element_type* d_c_tmp = nullptr;
    CHECK_CUDA(cudaMalloc(&d_b, dense_elems * sizeof(element_type)));
    CHECK_CUDA(cudaMalloc(&d_c_ref, dense_elems * sizeof(element_type)));
    CHECK_CUDA(cudaMalloc(&d_c_tmp, dense_elems * sizeof(element_type)));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), dense_elems * sizeof(element_type), cudaMemcpyHostToDevice));

    cusparseHandle_t handle = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c_ref = nullptr;
    cusparseDnMatDescr_t mat_c_tmp = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateDnMat(&mat_b, n, k, n, d_b, CUDA_R_TYPE, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&mat_c_ref, n, k, n, d_c_ref, CUDA_R_TYPE, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&mat_c_tmp, n, k, n, d_c_tmp, CUDA_R_TYPE, CUSPARSE_ORDER_COL));

    std::printf("\n=== arxiv CSR vs Blocked-ELL SpMM ===\n");
    std::printf("graph%d row-normalized transpose: n=%d nnz=%d avg_nnz=%.2f K=%d niter=%d\n",
                graph_id, n, graph.nnz, static_cast<double>(graph.nnz) / std::max(1, n), k, niter);
    std::printf("max_ell_mb=%zu\n\n", max_ell_mb);
    std::printf("%-14s %10s %12s %12s %10s %s\n",
                "format", "ms/iter", "GFLOP/s", "workspaceMB", "check", "details");

    const double true_gflops = 2.0 * static_cast<double>(graph.nnz) * k / 1e9;

    const BenchResult csr_default = bench_csr(handle,
                                              graph,
                                              mat_b,
                                              mat_c_ref,
                                              CUSPARSE_SPMM_ALG_DEFAULT,
                                              nwarm,
                                              niter);
    std::printf("%-14s %10.3f %12.2f %12.1f %10s baseline\n",
                "CSR_DEFAULT",
                csr_default.ms,
                true_gflops / (csr_default.ms * 1e-3),
                csr_default.workspace_bytes / 1e6,
                "OK");

    struct CsrAlg {
        const char* name;
        cusparseSpMMAlg_t alg;
    };
    const CsrAlg csr_algs[] = {
        {"CSR_ALG1", CUSPARSE_SPMM_CSR_ALG1},
        {"CSR_ALG2", CUSPARSE_SPMM_CSR_ALG2},
        {"CSR_ALG3", CUSPARSE_SPMM_CSR_ALG3},
    };
    for (const CsrAlg& alg : csr_algs) {
        const BenchResult result = bench_csr(handle, graph, mat_b, mat_c_tmp,
                                             alg.alg, nwarm, niter);
        const bool ok = verify_padded(d_c_ref, n, d_c_tmp, n, k,
                                      kVerifyRelTol, kVerifyAbsTol);
        std::printf("%-14s %10.3f %12.2f %12.1f %10s\n",
                    alg.name,
                    result.ms,
                    true_gflops / (result.ms * 1e-3),
                    result.workspace_bytes / 1e6,
                    ok ? "OK" : "FAIL");
    }

    const int block_sizes[] = {1, 2, 4, 8, 16, 32};
    for (int block_size : block_sizes) {
        BellMatrix bell = csr_to_blocked_ell(n, block_size, h_row_offsets, h_col_ind, h_vals);
        const size_t ell_col_bytes = bell.col_ind.size() * sizeof(int);
        const size_t ell_val_bytes = bell.vals.size() * sizeof(element_type);
        const size_t ell_sparse_bytes = ell_col_bytes + ell_val_bytes;
        const double pad_factor = static_cast<double>(bell.vals.size()) / std::max(1, graph.nnz);

        char name[32];
        std::snprintf(name, sizeof(name), "BELL_%d", block_size);
        if (max_ell_mb > 0 && ell_sparse_bytes > max_ell_bytes) {
            std::printf("%-14s %10s %12s %12s %10s ell_cols=%d rows=%d pad_x=%.2f sparseMB=%.1f skipped\n",
                        name, "-", "-", "-", "SKIP", bell.ell_cols, bell.rows, pad_factor,
                        ell_sparse_bytes / 1e6);
            continue;
        }

        int* d_bell_col_ind = nullptr;
        element_type* d_bell_vals = nullptr;
        element_type* d_b_pad = nullptr;
        element_type* d_c_pad = nullptr;
        const size_t dense_pad_elems = static_cast<size_t>(bell.rows) * k;
        CHECK_CUDA(cudaMalloc(&d_bell_col_ind, ell_col_bytes));
        CHECK_CUDA(cudaMalloc(&d_bell_vals, ell_val_bytes));
        CHECK_CUDA(cudaMalloc(&d_b_pad, dense_pad_elems * sizeof(element_type)));
        CHECK_CUDA(cudaMalloc(&d_c_pad, dense_pad_elems * sizeof(element_type)));
        CHECK_CUDA(cudaMemcpy(d_bell_col_ind, bell.col_ind.data(), ell_col_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_bell_vals, bell.vals.data(), ell_val_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_b_pad, 0, dense_pad_elems * sizeof(element_type)));
        for (int col = 0; col < k; ++col) {
            CHECK_CUDA(cudaMemcpy(d_b_pad + static_cast<size_t>(col) * bell.rows,
                                  d_b + static_cast<size_t>(col) * n,
                                  n * sizeof(element_type),
                                  cudaMemcpyDeviceToDevice));
        }

        const BenchResult result = bench_bell(handle,
                                              bell,
                                              k,
                                              d_bell_col_ind,
                                              d_bell_vals,
                                              d_b_pad,
                                              d_c_pad,
                                              nwarm,
                                              niter);
        const bool ok = verify_padded(d_c_ref, n, d_c_pad, bell.rows, k,
                                      kVerifyRelTol, kVerifyAbsTol);
        const double padded_gflops = 2.0 * static_cast<double>(bell.vals.size()) * k / 1e9;

        std::printf("%-14s %10.3f %12.2f %12.1f %10s ell_cols=%d rows=%d pad_x=%.2f sparseMB=%.1f padded_GF/s=%.2f\n",
                    name,
                    result.ms,
                    true_gflops / (result.ms * 1e-3),
                    result.workspace_bytes / 1e6,
                    ok ? "OK" : "FAIL",
                    bell.ell_cols,
                    bell.rows,
                    pad_factor,
                    ell_sparse_bytes / 1e6,
                    padded_gflops / (result.ms * 1e-3));

        cudaFree(d_c_pad);
        cudaFree(d_b_pad);
        cudaFree(d_bell_vals);
        cudaFree(d_bell_col_ind);
    }

    cusparseDestroyDnMat(mat_c_tmp);
    cusparseDestroyDnMat(mat_c_ref);
    cusparseDestroyDnMat(mat_b);
    cusparseDestroy(handle);
    cudaFree(d_c_tmp);
    cudaFree(d_c_ref);
    cudaFree(d_b);
    destroy_cuda_csr(graph);
    return 0;
}

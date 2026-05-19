// Benchmark: CSR SpMM vs Blocked ELL SpMM (cuSPARSE)
// Usage: ./spmm_ell_bench [N=20000] [K=256] [avg_nnz=10] [niter=20]

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <map>
#include <string>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t _e = (call); \
    if (_e != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, (int)_e); \
        std::exit(1); \
    } \
} while (0)

// ── random CSR (N×N, avg_nnz non-zeros per row, sorted cols) ─────────────────
void gen_csr(int N, int avg_nnz, unsigned seed,
             std::vector<int>& row_ptr,
             std::vector<int>& col_ind,
             std::vector<float>& vals) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> vd(0.1f, 1.0f);
    row_ptr.resize(N + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < N; i++) {
        std::set<int> picked;
        picked.insert(i); // self-loop so no empty row
        std::uniform_int_distribution<int> cd(0, N - 1);
        while ((int)picked.size() < std::min(avg_nnz, N))
            picked.insert(cd(rng));
        for (int c : picked) {
            col_ind.push_back(c);
            vals.push_back(vd(rng));
        }
        row_ptr[i + 1] = (int)col_ind.size();
    }
}

// ── load graph from edge-list file (format: "src dst" as floats per line) ────
// Returns N (number of nodes). Builds a CSR adjacency (row-normalized weights=1).
int load_graph_csr(const char* path, bool directed,
                   std::vector<int>& row_ptr,
                   std::vector<int>& col_ind,
                   std::vector<float>& vals) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); std::exit(1); }

    std::vector<std::pair<int,int>> edges;
    float a, b;
    while (fscanf(f, "%f %f", &a, &b) == 2) {
        int u = (int)a, v = (int)b;
        edges.push_back({u, v});
        if (!directed && u != v) edges.push_back({v, u});
    }
    fclose(f);

    int N = 0;
    for (auto& [u, v] : edges) N = std::max(N, std::max(u, v) + 1);

    // sort and deduplicate
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    // build CSR
    row_ptr.assign(N + 1, 0);
    for (auto& [u, v] : edges) row_ptr[u + 1]++;
    for (int i = 0; i < N; i++) row_ptr[i + 1] += row_ptr[i];

    col_ind.resize(edges.size());
    vals.resize(edges.size(), 1.0f);
    std::vector<int> pos(row_ptr.begin(), row_ptr.begin() + N);
    for (auto& [u, v] : edges) col_ind[pos[u]++] = v;

    return N;
}

// ── CSR → Blocked ELL ────────────────────────────────────────────────────────
// N_pad must be a multiple of blk_size (cuSPARSE rows/cols requirement).
// ellColInd stores block-column indices (col / blk_size).
// ellValue layout is cuSPARSE Blocked-ELL row-major:
//   [block_row][row_in_block][ell_slot][col_in_block].
// Rows N..N_pad-1 are zero (padding). Padded ELL slots use col_ind=-1, val=0.
int csr_to_bell(int N, int N_pad, int blk_size,
                const std::vector<int>& row_ptr,
                const std::vector<int>& col_ind,
                const std::vector<float>& vals,
                std::vector<int>&   ell_col_ind,
                std::vector<float>& ell_vals) {
    const int block_rows = N_pad / blk_size;

    // Pass 1: find max block-columns in any block-row. ellCols is measured in
    // scalar columns of the Blocked-ELL value array, so each block slot adds
    // blk_size columns.
    int max_block_cols = 0;
    for (int br = 0; br < block_rows; br++) {
        std::set<int> blks;
        int row_begin = br * blk_size;
        int row_end = std::min(row_begin + blk_size, N);
        for (int i = row_begin; i < row_end; i++) {
            for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
                blks.insert(col_ind[p] / blk_size);
        }
        max_block_cols = std::max(max_block_cols, (int)blks.size());
    }
    int ell_cols = max_block_cols * blk_size;
    int ell_blocks = max_block_cols;

    // Allocate for N_pad rows; extra rows stay zero (padding).
    ell_col_ind.assign((size_t)block_rows * ell_blocks, -1);
    ell_vals.assign((size_t)N_pad * ell_cols, 0.0f);

    // Pass 2: fill block rows
    for (int br = 0; br < block_rows; br++) {
        std::map<int, std::vector<float>> blk_map;
        int row_begin = br * blk_size;
        int row_end = std::min(row_begin + blk_size, N);
        for (int i = row_begin; i < row_end; i++) {
            int local_row = i - row_begin;
            for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
                int c = col_ind[p];
                int bc = c / blk_size;
                int local_col = c % blk_size;
                if (!blk_map.count(bc))
                    blk_map[bc].assign((size_t)blk_size * blk_size, 0.0f);
                blk_map[bc][(size_t)local_row * blk_size + local_col] = vals[p];
            }
        }
        int slot = 0;
        for (auto& [bc, bv] : blk_map) {
            ell_col_ind[(size_t)br * ell_blocks + slot] = bc;
            for (int r = 0; r < blk_size; r++) {
                for (int c = 0; c < blk_size; c++) {
                    ell_vals[((size_t)row_begin + r) * ell_cols
                             + (size_t)slot * blk_size + c] =
                        bv[(size_t)r * blk_size + c];
                }
            }
            slot++;
        }
    }
    return ell_cols;
}

// ── CUDA event timing helpers ─────────────────────────────────────────────────
struct Timer {
    cudaEvent_t start, stop;
    Timer()  { CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop)); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { CHECK_CUDA(cudaEventRecord(start)); }
    float end()  { CHECK_CUDA(cudaEventRecord(stop));
                   CHECK_CUDA(cudaEventSynchronize(stop));
                   float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); return ms; }
};

// ── CSR SpMM benchmark ────────────────────────────────────────────────────────
// C (N×K, col-major) = A_csr (N×N) * B (N×K, col-major)
double bench_csr(cusparseHandle_t handle,
                 int N, int K, int nnz,
                 int* d_rowptr, int* d_colind, float* d_vals,
                 float* d_B, float* d_C,
                 cusparseSpMMAlg_t alg,
                 int nwarm, int niter) {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, N, N, nnz,
        d_rowptr, d_colind, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, N, K, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, N, K, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL));

    float alpha = 1.0f, beta = 0.0f;
    size_t buf_sz = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, &buf_sz));

    void* d_buf = nullptr;
    if (buf_sz) CHECK_CUDA(cudaMalloc(&d_buf, buf_sz));

    // preprocess if needed (required for CSR_ALG3; no-op for others)
    CHECK_CUSPARSE(cusparseSpMM_preprocess(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, d_buf));

    auto run = [&]() {
        CHECK_CUSPARSE(cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, d_buf));
    };

    for (int i = 0; i < nwarm; i++) run();
    CHECK_CUDA(cudaDeviceSynchronize());

    Timer t;
    t.begin();
    for (int i = 0; i < niter; i++) run();
    float total = t.end();

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    if (d_buf) cudaFree(d_buf);

    return (double)total / niter;
}

// ── Blocked ELL SpMM benchmark ────────────────────────────────────────────────
// N_pad: padded row/column count for ELL descriptor (must be multiple of blk_size).
// The benchmark matrix is square; columns are padded with all-zero entries so
// A_pad * B_pad has the same first N rows as A * B.
// d_B, d_C: dense buffers with N_pad rows (col-major, leading dim N_pad).
double bench_ell(cusparseHandle_t handle,
                 int N_pad, int K, int ell_cols, int blk_size,
                 int* d_ell_colinds, float* d_ell_vals,
                 float* d_B, float* d_C,
                 int nwarm, int niter) {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CHECK_CUSPARSE(cusparseCreateBlockedEll(&matA, N_pad, N_pad,
        blk_size, ell_cols,
        d_ell_colinds, d_ell_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, N_pad, K, N_pad, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, N_pad, K, N_pad, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL));

    float alpha = 1.0f, beta = 0.0f;
    size_t buf_sz = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &buf_sz));

    void* d_buf = nullptr;
    if (buf_sz) CHECK_CUDA(cudaMalloc(&d_buf, buf_sz));

    auto run = [&]() {
        CHECK_CUSPARSE(cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_BLOCKED_ELL_ALG1, d_buf));
    };

    for (int i = 0; i < nwarm; i++) run();
    CHECK_CUDA(cudaDeviceSynchronize());

    Timer t;
    t.begin();
    for (int i = 0; i < niter; i++) run();
    float total = t.end();

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    if (d_buf) cudaFree(d_buf);

    return (double)total / niter;
}

// ── verify two device vectors agree ──────────────────────────────────────────
bool verify(float* d_ref, float* d_cmp, size_t n, float tol = 1e-3f) {
    std::vector<float> h_ref(n), h_cmp(n);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cmp.data(), d_cmp, n * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float err = std::fabs(h_ref[i] - h_cmp[i]);
        float scale = std::max(1.0f, std::fabs(h_ref[i]));
        max_err = std::max(max_err, err / scale);
    }
    if (max_err > tol) {
        printf("  [MISMATCH] max_rel_err = %e\n", max_err);
        return false;
    }
    return true;
}

// ── verify first N rows of a padded col-major matrix against a reference ─────
// d_ref:  N_ref  x K, col-major (leading dim N_ref)
// d_cmp:  N_pad  x K, col-major (leading dim N_pad) — only first N rows checked
bool verify_padded(float* d_ref, int N_ref, float* d_cmp, int N_pad,
                   int N_rows, int K, float tol = 1e-3f) {
    std::vector<float> h_ref((size_t)N_ref * K), h_cmp((size_t)N_pad * K);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cmp.data(), d_cmp, h_cmp.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < N_rows; i++) {
            float ref = h_ref[(size_t)j * N_ref + i];
            float cmp = h_cmp[(size_t)j * N_pad + i];
            float err = std::fabs(ref - cmp);
            float scale = std::max(1.0f, std::fabs(ref));
            max_err = std::max(max_err, err / scale);
        }
    }
    if (max_err > tol) {
        printf("  [MISMATCH] max_rel_err = %e\n", max_err);
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Usage: ./spmm_ell_bench [graph_dir_or_N] [K=64] [avg_nnz=10] [niter=20]
//   graph_dir: path containing graph1.txt (edge list "src dst" as floats)
//              and config.txt with "directed=0|1"
//   If first arg is a number, uses random generation with that N.
int main(int argc, char** argv) {
    int niter = argc > 4 ? atoi(argv[4]) : 20;
    int nwarm = 5;
    int K     = argc > 2 ? atoi(argv[2]) : 64;

    printf("=== SpMM Benchmark: CSR vs Blocked ELL ===\n");

    std::vector<int>   h_rowptr, h_colind;
    std::vector<float> h_vals;
    int N = 0;

    bool use_graph = (argc > 1 && argv[1][0] != '\0' && !isdigit((unsigned char)argv[1][0]));
    if (use_graph) {
        // Load from graph directory
        char graph_path[512], config_path[512];
        snprintf(graph_path,  sizeof(graph_path),  "%s/graph1.txt", argv[1]);
        snprintf(config_path, sizeof(config_path), "%s/config.txt",  argv[1]);
        bool directed = true;
        FILE* cf = fopen(config_path, "r");
        if (cf) {
            char line[128];
            while (fgets(line, sizeof(line), cf)) {
                int v; if (sscanf(line, "directed=%d", &v) == 1) directed = (v != 0);
            }
            fclose(cf);
        }
        printf("Loading graph from %s (directed=%d)\n", graph_path, (int)directed);
        N = load_graph_csr(graph_path, directed, h_rowptr, h_colind, h_vals);
    } else {
        N           = argc > 1 ? atoi(argv[1]) : 20000;
        int avg_nnz = argc > 3 ? atoi(argv[3]) : 10;
        printf("Generating random graph N=%d avg_nnz=%d\n", N, avg_nnz);
        gen_csr(N, avg_nnz, 42, h_rowptr, h_colind, h_vals);
    }

    int nnz = (int)h_colind.size();
    printf("N=%d  K=%d  nnz=%d  avg_nnz=%.1f  niter=%d\n\n",
           N, K, nnz, (float)nnz / N, niter);

    // ── Dense matrices (col-major N×K) ───────────────────────────────────────
    size_t dense_sz = (size_t)N * K;
    std::vector<float> h_B(dense_sz), h_C(dense_sz, 0.0f);
    {
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto& x : h_B) x = d(rng);
    }

    // ── Upload CSR + dense matrices ──────────────────────────────────────────
    int   *d_rowptr, *d_colind;
    float *d_vals, *d_B, *d_C_ref, *d_C_cmp;
    CHECK_CUDA(cudaMalloc(&d_rowptr, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_vals,   nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B,      dense_sz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_ref,  dense_sz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_cmp,  dense_sz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowptr.data(), (N+1)*sizeof(int),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colind, h_colind.data(), nnz*sizeof(int),    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals,   h_vals.data(),   nnz*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B,      h_B.data(),      dense_sz*sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // ── CSR benchmarks ───────────────────────────────────────────────────────
    printf("\n%-20s  %10s  %12s\n", "Algorithm", "ms/iter", "GFLOP/s");
    printf("%-20s  %10s  %12s\n", "----------", "-------", "--------");

    double gflops_base = 2.0 * nnz * K / 1e9;

    // Run CSR_DEFAULT first, save gold reference in d_C_ref (never overwritten again)
    {
        double ms = bench_csr(handle, N, K, nnz,
                              d_rowptr, d_colind, d_vals,
                              d_B, d_C_ref, CUSPARSE_SPMM_ALG_DEFAULT, nwarm, niter);
        printf("%-20s  %10.3f  %12.1f\n", "CSR_DEFAULT", ms, gflops_base / (ms * 1e-3));
    }

    // Remaining CSR variants write to d_C_cmp so d_C_ref stays intact
    struct { const char* name; cusparseSpMMAlg_t alg; } csr_rest[] = {
        {"CSR_ALG1", CUSPARSE_SPMM_CSR_ALG1},
        {"CSR_ALG2", CUSPARSE_SPMM_CSR_ALG2},
        {"CSR_ALG3", CUSPARSE_SPMM_CSR_ALG3},
    };
    for (auto& v : csr_rest) {
        double ms = bench_csr(handle, N, K, nnz,
                              d_rowptr, d_colind, d_vals,
                              d_B, d_C_cmp, v.alg, nwarm, niter);
        bool ok = verify(d_C_ref, d_C_cmp, dense_sz);
        printf("%-20s  %10.3f  %12.1f  %s\n", v.name, ms, gflops_base / (ms * 1e-3),
               ok ? "OK" : "FAIL");
    }

    // ── Blocked ELL benchmarks (block sizes 1, 2, 4, 8) ─────────────────────
    // cuSPARSE Blocked-ELL requires both sparse rows and sparse columns to be
    // multiples of ellBlockSize. We pad the square N x N matrix to N_pad x N_pad
    // and zero-fill the extra dense B rows so the original result is unchanged.
    printf("\n");
    int blk_sizes[] = {1, 2, 4, 8};
    for (int blk : blk_sizes) {
        int N_pad = ((N + blk - 1) / blk) * blk;

        std::vector<int>   h_ell_colinds;
        std::vector<float> h_ell_vals;
        int ell_cols = csr_to_bell(N, N_pad, blk, h_rowptr, h_colind, h_vals,
                                   h_ell_colinds, h_ell_vals);

        size_t ell_ci_sz   = (size_t)(N_pad / blk) * (ell_cols / blk);
        size_t ell_v_sz    = (size_t)N_pad * ell_cols;
        size_t dense_pad   = (size_t)N_pad * K;

        int*   d_ell_ci;
        float* d_ell_v;
        float* d_B_pad;
        float* d_C_pad;
        CHECK_CUDA(cudaMalloc(&d_ell_ci, ell_ci_sz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_ell_v,  ell_v_sz  * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_pad,  dense_pad * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_pad,  dense_pad * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_ell_ci, h_ell_colinds.data(),
                              ell_ci_sz * sizeof(int),   cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_ell_v,  h_ell_vals.data(),
                              ell_v_sz  * sizeof(float), cudaMemcpyHostToDevice));

        // Build padded B (col-major, leading dim N_pad): copy N rows per column, zero the rest
        CHECK_CUDA(cudaMemset(d_B_pad, 0, dense_pad * sizeof(float)));
        for (int j = 0; j < K; j++)
            CHECK_CUDA(cudaMemcpy(d_B_pad + (size_t)j * N_pad,
                                  d_B     + (size_t)j * N,
                                  N * sizeof(float), cudaMemcpyDeviceToDevice));

        double ms = bench_ell(handle, N_pad, K, ell_cols, blk,
                              d_ell_ci, d_ell_v, d_B_pad, d_C_pad,
                              nwarm, niter);

        bool ok = verify_padded(d_C_ref, N, d_C_pad, N_pad, N, K);
        double mem_ell = (ell_ci_sz * sizeof(int) + ell_v_sz * sizeof(float)) / 1e6;
        double mem_csr = (nnz * (sizeof(int) + sizeof(float)) + (N+1)*sizeof(int)) / 1e6;

        char name[32];
        snprintf(name, sizeof(name), "ELL_blk%d", blk);
        printf("%-20s  %10.3f  %12.1f  %s  mem=%.0fMB (CSR=%.0fMB, ell_cols=%d, N_pad=%d)\n",
               name, ms, gflops_base / (ms * 1e-3),
               ok ? "OK" : "FAIL", mem_ell, mem_csr, ell_cols, N_pad);

        cudaFree(d_ell_ci);
        cudaFree(d_ell_v);
        cudaFree(d_B_pad);
        cudaFree(d_C_pad);
    }

    printf("\n");

    // ── Cleanup ──────────────────────────────────────────────────────────────
    cusparseDestroy(handle);
    cudaFree(d_rowptr); cudaFree(d_colind); cudaFree(d_vals);
    cudaFree(d_B); cudaFree(d_C_ref); cudaFree(d_C_cmp);
    return 0;
}

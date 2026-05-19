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

// ── CSR → Blocked ELL ────────────────────────────────────────────────────────
// ellColInd stores block-column indices (col / blk_size).
// ellValue  layout: [row][ell_slot][blk_size] (row-major across those dims).
// Padding: col_ind = 0 (block 0), values = 0.0.
int csr_to_bell(int N, int blk_size,
                const std::vector<int>& row_ptr,
                const std::vector<int>& col_ind,
                const std::vector<float>& vals,
                std::vector<int>&   ell_col_ind,
                std::vector<float>& ell_vals) {
    // Pass 1: find max blocks per row
    int ell_cols = 0;
    for (int i = 0; i < N; i++) {
        std::set<int> blks;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
            blks.insert(col_ind[p] / blk_size);
        ell_cols = std::max(ell_cols, (int)blks.size());
    }

    ell_col_ind.assign((size_t)N * ell_cols, 0);
    ell_vals.assign((size_t)N * ell_cols * blk_size, 0.0f);

    // Pass 2: fill
    for (int i = 0; i < N; i++) {
        // collect block_col → values[0..blk_size-1]
        std::map<int, std::vector<float>> blk_map;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
            int c = col_ind[p];
            int bc = c / blk_size;
            int off = c % blk_size;
            if (!blk_map.count(bc))
                blk_map[bc].assign(blk_size, 0.0f);
            blk_map[bc][off] = vals[p];
        }
        int slot = 0;
        for (auto& [bc, bv] : blk_map) {
            ell_col_ind[(size_t)i * ell_cols + slot] = bc;
            for (int k = 0; k < blk_size; k++)
                ell_vals[((size_t)i * ell_cols + slot) * blk_size + k] = bv[k];
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

    // preprocess if needed
    cusparseSpMM_preprocess(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, d_buf);

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
double bench_ell(cusparseHandle_t handle,
                 int N, int K, int ell_cols, int blk_size,
                 int* d_ell_colinds, float* d_ell_vals,
                 float* d_B, float* d_C,
                 int nwarm, int niter) {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CHECK_CUSPARSE(cusparseCreateBlockedEll(&matA, N, N,
        blk_size, ell_cols,
        d_ell_colinds, d_ell_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, N, K, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, N, K, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL));

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

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int N       = argc > 1 ? atoi(argv[1]) : 20000;
    int K       = argc > 2 ? atoi(argv[2]) : 256;
    int avg_nnz = argc > 3 ? atoi(argv[3]) : 10;
    int niter   = argc > 4 ? atoi(argv[4]) : 20;
    int nwarm   = 5;

    printf("=== SpMM Benchmark: CSR vs Blocked ELL ===\n");
    printf("N=%d  K=%d  avg_nnz=%d  niter=%d\n\n", N, K, avg_nnz, niter);

    // ── Build CSR on host ────────────────────────────────────────────────────
    std::vector<int>   h_rowptr, h_colind;
    std::vector<float> h_vals;
    gen_csr(N, avg_nnz, 42, h_rowptr, h_colind, h_vals);
    int nnz = (int)h_colind.size();
    printf("CSR: nnz=%d  actual_avg_nnz=%.1f\n", nnz, (float)nnz / N);

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
    struct { const char* name; cusparseSpMMAlg_t alg; } csr_variants[] = {
        {"CSR_DEFAULT", CUSPARSE_SPMM_ALG_DEFAULT},
        {"CSR_ALG1",    CUSPARSE_SPMM_CSR_ALG1},
        {"CSR_ALG2",    CUSPARSE_SPMM_CSR_ALG2},
        {"CSR_ALG3",    CUSPARSE_SPMM_CSR_ALG3},
    };

    printf("\n%-20s  %10s  %12s\n", "Algorithm", "ms/iter", "GFLOP/s");
    printf("%-20s  %10s  %12s\n", "----------", "-------", "--------");

    double gflops_base = 2.0 * nnz * K / 1e9;
    double ref_ms = 0.0;
    bool first_csr = true;

    for (auto& v : csr_variants) {
        CHECK_CUDA(cudaMemset(d_C_ref, 0, dense_sz * sizeof(float)));
        double ms = bench_csr(handle, N, K, nnz,
                              d_rowptr, d_colind, d_vals,
                              d_B, d_C_ref, v.alg, nwarm, niter);
        if (first_csr) { ref_ms = ms; first_csr = false; }
        printf("%-20s  %10.3f  %12.1f\n", v.name, ms, gflops_base / (ms * 1e-3));
    }

    // ── Blocked ELL benchmarks (block sizes 1, 2, 4, 8) ─────────────────────
    printf("\n");
    int blk_sizes[] = {1, 2, 4, 8};
    for (int blk : blk_sizes) {
        std::vector<int>   h_ell_colinds;
        std::vector<float> h_ell_vals;
        int ell_cols = csr_to_bell(N, blk, h_rowptr, h_colind, h_vals,
                                   h_ell_colinds, h_ell_vals);

        size_t ell_ci_sz = (size_t)N * ell_cols;
        size_t ell_v_sz  = (size_t)N * ell_cols * blk;

        int*   d_ell_ci;
        float* d_ell_v;
        CHECK_CUDA(cudaMalloc(&d_ell_ci, ell_ci_sz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_ell_v,  ell_v_sz  * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_ell_ci, h_ell_colinds.data(),
                              ell_ci_sz * sizeof(int),   cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_ell_v,  h_ell_vals.data(),
                              ell_v_sz  * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemset(d_C_cmp, 0, dense_sz * sizeof(float)));
        double ms = bench_ell(handle, N, K, ell_cols, blk,
                              d_ell_ci, d_ell_v, d_B, d_C_cmp,
                              nwarm, niter);

        // correctness vs CSR_DEFAULT
        bool ok = verify(d_C_ref, d_C_cmp, dense_sz);
        double mem_ell  = (ell_ci_sz * sizeof(int) + ell_v_sz * sizeof(float)) / 1e6;
        double mem_csr  = (nnz * (sizeof(int) + sizeof(float)) + (N+1)*sizeof(int)) / 1e6;

        char name[32];
        snprintf(name, sizeof(name), "ELL_blk%d", blk);
        printf("%-20s  %10.3f  %12.1f  %s  mem=%.0fMB (CSR=%.0fMB, ell_cols=%d)\n",
               name, ms, gflops_base / (ms * 1e-3),
               ok ? "OK" : "FAIL", mem_ell, mem_csr, ell_cols);

        cudaFree(d_ell_ci);
        cudaFree(d_ell_v);
    }

    printf("\n");

    // ── Cleanup ──────────────────────────────────────────────────────────────
    cusparseDestroy(handle);
    cudaFree(d_rowptr); cudaFree(d_colind); cudaFree(d_vals);
    cudaFree(d_B); cudaFree(d_C_ref); cudaFree(d_C_cmp);
    return 0;
}

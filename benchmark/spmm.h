#include <vector>
#include <omp.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <algorithm>
#include "matrix.h"

/**
 * SpMM: C = alpha * A * B + beta * C
 *
 * A : sparse matrix (MKL CSR, n1 x n1)
 * B : dense matrix  (row-major, n1 x n2)  -- called D_col in your code
 * C : dense matrix  (row-major, n1 x n2)  -- result
 *
 * Mirrors:
 *   mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, graph_csr, descr,
 *                   SPARSE_LAYOUT_ROW_MAJOR, B, n2, n2, beta, C, n2)
 *
 * Row-summation strategy:
 *   For each row i of A:
 *     C[i,:] = beta * C[i,:]
 *     For each nonzero (i, col_j, val_j) in row i:
 *         C[i,:] += val_j * B[col_j,:]
 */
void omp_spmm(double               alpha,
              sparse_matrix_t      A_mkl,   // MKL CSR handle
              size_t                  n1,      // number of rows/cols in A
              size_t                  n2,      // number of columns in B and C
              double               beta,
              const double*        B,       // dense input  [n1 x n2], row-major
              double*              C)       // dense output [n1 x n2], row-major
{

    auto time_0 = std::chrono::high_resolution_clock::now();
    // ----------------------------------------------------------------
    // 1.  Extract raw CSR arrays from the MKL sparse handle
    // ----------------------------------------------------------------
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;   // row_ptr begin,  size = rows
    MKL_INT *rows_end   = nullptr;   // row_ptr end,    size = rows
    MKL_INT *col_ind    = nullptr;   // column indices of nonzeros
    double  *vals       = nullptr;   // nonzero values

    check_mkl(
        mkl_sparse_d_export_csr(
            A_mkl,
            &indexing,
            &rows,
            &cols,
            &rows_start,
            &rows_end,
            &col_ind,
            &vals
        ),
        "export csr failed"
    );

    // MKL exports two arrays (row_start, row_end) where:
    //   row i has nonzeros in col_idx[ row_start[i] .. row_end[i]-1 ]
    // Both are 0-based when indexing == SPARSE_INDEX_BASE_ZERO.
    const int base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    auto time_1 = std::chrono::high_resolution_clock::now();

    // // ----------------------------------------------------------------
    // // 2. Precompute alpha * vals  (skip if alpha == 1.0)
    // // ----------------------------------------------------------------
    // MKL_INT nnz = rows_end[rows - 1] - base;   // total nonzeros
    // std::vector<double> scaled_vals;

    // const double* eff_vals = vals;              // pointer used in main loop

    // if (alpha != 1.0 && alpha != 0.0) {
    //     scaled_vals.resize(nnz);
    //     #pragma omp parallel for schedule(static) num_threads(48)
    //     for (MKL_INT p = 0; p < nnz; ++p)
    //         scaled_vals[p] = alpha * vals[p];
    //     eff_vals = scaled_vals.data();
    // }


    // ----------------------------------------------------------------
    // 2.  Parallel SpMM  (row-summation)
    // ----------------------------------------------------------------
    // #pragma omp parallel for schedule(dynamic, num_threads)
    #pragma omp parallel for schedule(guided)
    // #pragma omp parallel for schedule(static) num_threads(48)
    for (int i = 0; i < rows; ++i)
    {

        // if ( i == 0 & omp_get_thread_num() == 0) {
        //     int nthreads = omp_get_num_threads();
        //     printf("Number of threads in this region: %d\n", nthreads);
        // }
        // double* Ci = C + (size_t)i * n2;          // output row i
        double* __restrict__ Ci = C + (size_t)i * n2;

        // --- beta scaling --------------------------------------------
        if (beta == 0.0) {
            memset(Ci, 0, n2 * sizeof(double));
        } else if (beta != 1.0) {
            #pragma omp simd
            for (int k = 0; k < n2; ++k)
                Ci[k] *= beta;
        }

        if (alpha == 0.0) continue;

        // --- accumulate alpha * A[i,j] * B[j,:] for each nonzero -----
        const int row_s = rows_start[i] - base;    // start index in col_idx/values
        const int row_e = rows_end[i]   - base;    // one-past-end

        for (int p = row_s; p < row_e; ++p)
        {
            const int    j   = col_ind[p] - base;
            const double aij = alpha * vals[p];
            // const double aij = eff_vals[p];
            // const double* Bj = B + (size_t)j * n2;   // source row j of B
            const double* __restrict__ Bj = B + (size_t)j * n2;

            // Vectorised dot-add (compiler / AVX auto-vectorise this loop)
            #pragma omp simd
            for (int k = 0; k < n2; ++k)
                Ci[k] += aij * Bj[k];
        }
    }
    auto time_2 = std::chrono::high_resolution_clock::now();
    auto extract_csr = std::chrono::duration<double, std::milli>(time_1 - time_0).count();
    auto mult = std::chrono::duration<double, std::milli>(time_2 - time_1).count();
    // std::cout << "inside omp spmm  extract_csr " << extract_csr << " mult " << mult << " ms\n";

}

// ============================================================
// Row-Blocked SpMM  (K-dimension blocking)
//
//  Blocks the SHARED dimension K into panels of b_r rows.
//
//  For each K-panel q:
//    k_start = q * b_r
//    k_end   = min(k_start + b_r, n1)
//
//  STEP 1: Pack B[k_start:k_end, 0:n2] → B_panel  (b_r x n2, contiguous)
//  STEP 2: For each row i of A (parallel):
//            For each nonzero (i, k, v) where k in [k_start, k_end):
//              C[i, :] += v * B_panel[k - k_start, :]
//
//  C[i,:] is accumulated across all Q panels.
//  beta is applied BEFORE the first panel (q==0).
// ============================================================
void omp_spmm_kblock(double               alpha,
                     sparse_matrix_t      graph_csr,
                     MKL_INT              n1,         // rows/cols of A
                     int                  n2,         // cols of B and C
                     double               beta,
                     const double*        B,           // [n1 x n2] row-major
                     double*              C,           // [n1 x n2] row-major
                     int                  b_r,         // K-panel height (rows of B per stage)
                     int                  num_threads = 48)
{
    omp_set_num_threads(num_threads);

     // ----------------------------------------------------------------
    // 1. Export CSR
    // ----------------------------------------------------------------
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    b_r = std::min((MKL_INT)b_r, n1);
    b_r = std::max(b_r, 1);

    const int Q = (n1 + b_r - 1) / b_r;   // number of K-panels

    // ----------------------------------------------------------------
    // 2. Allocate B_panel buffer  [b_r x n2] — reused each stage
    //    Packed contiguous so that B_panel[local_k, :] is stride-n2
    //    but the entire panel fits in cache
    // ----------------------------------------------------------------
    std::vector<double> B_panel((size_t)b_r * n2);

    // ----------------------------------------------------------------
    // 3. Apply beta to C once before accumulation (q==0 equivalent)
    //    Do it upfront so the K-panel loop only accumulates (+=)
    // ----------------------------------------------------------------
    if (beta == 0.0) {
        memset(C, 0, (size_t)rows * n2 * sizeof(double));
    } else if (beta != 1.0) {
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (MKL_INT i = 0; i < rows; ++i) {
            double* __restrict__ Ci = C + (size_t)i * n2;
            #pragma omp simd
            for (int j = 0; j < n2; ++j)
                Ci[j] *= beta;
        }
    }

    if (alpha == 0.0) return;

    // ----------------------------------------------------------------
    // 4. Main K-panel loop
    // ----------------------------------------------------------------
    for (int q = 0; q < Q; ++q)
    {
        const MKL_INT k_start  = (MKL_INT)q * b_r;
        const MKL_INT k_end    = std::min(k_start + (MKL_INT)b_r, n1);
        const MKL_INT panel_h  = k_end - k_start;   // actual height (last panel)

        // ------------------------------------------------------------
        // STEP 1: Pack B[k_start:k_end, :] → B_panel[0:panel_h, :]
        //         B_panel[local_k, j] = B[k_start + local_k, j]
        //         Parallel copy — each thread owns a slice of rows
        // ------------------------------------------------------------
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (MKL_INT local_k = 0; local_k < panel_h; ++local_k)
        {
            const MKL_INT k = k_start + local_k;

            const double* __restrict__ Bk       = B        + (size_t)k       * n2;
            double*       __restrict__ Bk_panel = B_panel.data() + (size_t)local_k * n2;

            #pragma omp simd
            for (int j = 0; j < n2; ++j)
                Bk_panel[j] = Bk[j];
        }

        // implicit barrier — all threads done packing before compute

        // ------------------------------------------------------------
        // STEP 2: For each row i of A accumulate nonzeros whose
        //         column index k falls in [k_start, k_end)
        //
        //   C[i, :] += alpha * A[i,k] * B_panel[k - k_start, :]
        // ------------------------------------------------------------
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (MKL_INT i = 0; i < rows; ++i)
        {
            double* __restrict__ Ci = C + (size_t)i * n2;

            const MKL_INT row_s = rows_start[i] - base;
            const MKL_INT row_e = rows_end[i]   - base;

            for (MKL_INT p = row_s; p < row_e; ++p)
            {
                const MKL_INT k = col_ind[p] - base;

                // Only process nonzeros in current K-panel
                if (k < k_start || k >= k_end) continue;

                const MKL_INT local_k = k - k_start;
                const double  aik     = alpha * vals[p];

                // B_panel row for this nonzero — loaded ONCE in STEP 1
                const double* __restrict__ Bk_panel =
                    B_panel.data() + (size_t)local_k * n2;

                #pragma omp simd
                for (int j = 0; j < n2; ++j)
                    Ci[j] += aik * Bk_panel[j];
            }
        }
        // implicit barrier before packing next panel
    }
}





#include <vector>
#include <omp.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <algorithm>

// ============================================================
// 2D Tiled SpMM — block BOTH i (rows of A/C) AND k (cols of A / rows of B)
//
//  Tile (ib, kb):
//    i_start = ib * b_r,  i_end = min(i_start + b_r, n1)
//    k_start = kb * b_k,  k_end = min(k_start + b_k, n1)
//
//  For this tile:
//    C[i_start:i_end, :]  stays in cache for all k in [k_start, k_end)
//    B[k_start:k_end, :]  stays in cache for all i in [i_start, i_end)
//    → each loaded ONCE per tile
//
//  Memory budget per tile (should fit L2/L3):
//    C tile: b_r * n2 * 8 bytes
//    B tile: b_k * n2 * 8 bytes
//    Total:  (b_r + b_k) * n2 * 8 bytes
// ============================================================
void omp_spmm_2d_tiled(double          alpha,
                       sparse_matrix_t graph_csr,
                       MKL_INT         n1,
                       int             n2,
                       double          beta,
                       const double*   B,            // [n1 x n2] row-major
                       double*         C,            // [n1 x n2] row-major
                       int             b_r,          // tile height: rows of A/C per tile
                       int             b_k,          // tile depth:  rows of B per tile (= cols of A)
                       int             num_threads = 48)
{
    omp_set_num_threads(num_threads);

    // ----------------------------------------------------------------
    // Export CSR
    // ----------------------------------------------------------------
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    b_r = std::min((int)n1, std::max(1, b_r));
    b_k = std::min((int)n1, std::max(1, b_k));

    const int num_tiles_i = (n1 + b_r - 1) / b_r;   // tiles along rows of A
    const int num_tiles_k = (n1 + b_k - 1) / b_k;   // tiles along cols of A

    // ----------------------------------------------------------------
    // Apply beta to C once upfront
    // ----------------------------------------------------------------
    if (beta == 0.0) {
        memset(C, 0, (size_t)n1 * n2 * sizeof(double));
    } else if (beta != 1.0) {
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (MKL_INT i = 0; i < n1; ++i) {
            double* __restrict__ Ci = C + (size_t)i * n2;
            #pragma omp simd
            for (int j = 0; j < n2; ++j) Ci[j] *= beta;
        }
    }

    if (alpha == 0.0) return;

    // ----------------------------------------------------------------
    // Main tiled loop
    //
    // Parallelise over i-tiles: each thread owns disjoint rows of C
    // → no race condition on C
    // → each thread streams through all k-tiles for its i-tile rows
    //
    // For each i-tile:
    //   For each k-tile:
    //     C[i_start:i_end, :] stays in cache across all k in k-tile  ← key!
    //     B[k_start:k_end, :] loaded once per k-tile                 ← key!
    // ----------------------------------------------------------------
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int ib = 0; ib < num_tiles_i; ++ib)
    {
        const MKL_INT i_start = (MKL_INT)ib * b_r;
        const MKL_INT i_end   = std::min(i_start + (MKL_INT)b_r, n1);

        // Iterate over k-tiles (cols of A / rows of B)
        for (int kb = 0; kb < num_tiles_k; ++kb)
        {
            const MKL_INT k_start = (MKL_INT)kb * b_k;
            const MKL_INT k_end   = std::min(k_start + (MKL_INT)b_k, n1);

            // ----------------------------------------------------
            // Process tile (ib, kb):
            //   For each row i in [i_start, i_end):
            //     C[i, :] is in L2 cache (b_r rows kept warm)
            //     For each nonzero (i, k, v) with k in [k_start, k_end):
            //       B[k, :] is in L2/L3 (b_k rows kept warm)
            //       C[i, :] += alpha * v * B[k, :]
            // ----------------------------------------------------
            for (MKL_INT i = i_start; i < i_end; ++i)
            {
                double* __restrict__ Ci = C + (size_t)i * n2;  // ← stays in L2

                const MKL_INT row_s = rows_start[i] - base;
                const MKL_INT row_e = rows_end[i]   - base;

                for (MKL_INT p = row_s; p < row_e; ++p)
                {
                    const MKL_INT k = col_ind[p] - base;

                    // Only nonzeros in the current k-tile
                    if (k < k_start || k >= k_end) continue;

                    const double  aik = alpha * vals[p];

                    // B[k,:] stays in L2/L3 while we process all i in i-tile
                    const double* __restrict__ Bk = B + (size_t)k * n2;

                    #pragma omp simd
                    for (int j = 0; j < n2; ++j)
                        Ci[j] += aik * Bk[j];
                }
            }
        }
    }
}

// ============================================================
// BLOCKED: jik loop order   (Algorithm 2 v1 — panel blocked)
//
//  Outer loop:  j = 0, b_c, 2*b_c, ...  (column panels of B/C)
//  Middle loop: i = 0 .. n1-1           (rows of A, parallelised)
//  Inner loop:  nonzeros in row i of A
//
//  Each stage only touches a NARROW strip of B and C (b_c cols),
//  so those strips fit in L2/L3 cache → far fewer cache misses.
// difference with omp_spmm_jik : here we read each column partition B inside the row for loop - this version is faster
// ============================================================
void omp_spmm_jik_v1(double alpha, sparse_matrix_t  graph_csr, MKL_INT n1, int  n2, double beta, const double*  B, double*  C, int b_c) {
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );
    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    b_c = std::min(b_c, n2);
    b_c = std::max(b_c, 1);
    // Q : number of blocks in dense matrices
    const int Q = (n2 + b_c - 1) / b_c;
    std::vector<double> times(Q);
    std::vector<double> times_th(Q * rows);
    for (int q = 0; q < Q; ++q)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        const int j_start = q * b_c;
        const int j_end   = std::min(j_start + b_c, n2);
        const int panel_w = j_end - j_start;

        #pragma omp parallel for schedule(guided, 64) 
        for (MKL_INT i = 0; i < rows; ++i)
        {
            auto t11 = std::chrono::high_resolution_clock::now();
            double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

            if (beta == 0.0)
                memset(Ci_panel, 0, panel_w * sizeof(double));
            else if (beta != 1.0) {
                #pragma omp simd
                for (int j = 0; j < panel_w; ++j)
                    Ci_panel[j] *= beta;
            }
            // beta == 1.0 → no scaling needed, C is already correct

            if (alpha == 0.0) continue;

            const MKL_INT row_s = rows_start[i] - base;
            const MKL_INT row_e = rows_end[i]   - base;

            for (MKL_INT p = row_s; p < row_e; ++p)
            {
                const MKL_INT k   = col_ind[p] - base;
                const double  aik = alpha * vals[p];

                const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

                #pragma omp simd
                for (int j = 0; j < panel_w; ++j)
                    Ci_panel[j] += aik * Bk_panel[j];
            }
            auto t22 = std::chrono::high_resolution_clock::now();
            double mss = std::chrono::duration<double, std::milli>(t22 - t11).count();
            times_th[q * rows + i] = mss;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        times[q] = ms;
        // printf("bc %d q %d time %e\n", b_c, q, ms);
    }
    measure_divergence(times);
    measure_divergence(times_th);
}

// ============================================================
// BLOCKED: jik loop order   (Algorithm 2 v1 — panel blocked)
//
//  Outer loop:  j = 0, b_c, 2*b_c, ...  (column panels of B/C)
//  Middle loop: i = 0 .. n1-1           (rows of A, parallelised)
//  Inner loop:  nonzeros in row i of A
//  Blocks of B and C are stored as block-wise storage with each block is stored row-wise
//  Each stage only touches a NARROW strip of B and C (b_c cols),
//  so those strips fit in L2/L3 cache → far fewer cache misses.
// difference with omp_spmm_jik : here we read each column partition B inside the row for loop - this version is faster
// ============================================================
void omp_spmm_jik_v3(double               alpha,
                  sparse_matrix_t      graph_csr,
                  MKL_INT              n1,
                  int                  n2,
                  double               beta,
                  const double*        B,
                  double*              C,
                  int                  b_c)
{
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    b_c = std::min(b_c, n2);
    b_c = std::max(b_c, 1);
    // Q : number of blocks in dense matrices
    const int Q = (n2 + b_c - 1) / b_c;
    for (int q = 0; q < Q; ++q)
    {
        const int j_start = q * b_c;
        const int j_end   = std::min(j_start + b_c, n2);
        const int panel_w = j_end - j_start;

        #pragma omp parallel for schedule(guided) 
        for (MKL_INT i = 0; i < rows; ++i)
        {
            // if ( q == 0 && i == 0 && omp_get_thread_num() == 0) {
            //     int nthreads = omp_get_num_threads();
            //     printf("Number of threads in this region: %d\n", nthreads);
            // }
            // double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;
            double* __restrict__ Ci_panel = C + j_start * (size_t) n1 +  (size_t)i * panel_w;

            if (beta == 0.0)
                memset(Ci_panel, 0, panel_w * sizeof(double));
            else if (beta != 1.0) {
                #pragma omp simd
                for (int j = 0; j < panel_w; ++j)
                    Ci_panel[j] *= beta;
            }
            // beta == 1.0 → no scaling needed, C is already correct

            if (alpha == 0.0) continue;

            const MKL_INT row_s = rows_start[i] - base;
            const MKL_INT row_e = rows_end[i]   - base;

            for (MKL_INT p = row_s; p < row_e; ++p)
            {
                const MKL_INT k   = col_ind[p] - base;
                const double  aik = alpha * vals[p];

                // const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;
                const double* __restrict__ Bk_panel = B + j_start * (size_t) n1 +  (size_t)k * panel_w;

                #pragma omp simd
                for (int j = 0; j < panel_w; ++j)
                    Ci_panel[j] += aik * Bk_panel[j];
            }
        }
    }
}


// //worst than omp_spmm_jik_v1
// void omp_spmm_jik_v2(double               alpha,
//                   sparse_matrix_t      graph_csr,
//                   MKL_INT              n1,
//                   int                  n2,
//                   double               beta,
//                   const double*        B,
//                   double*              C,
//                   int                  b_c)
// {
//     sparse_index_base_t indexing;
//     MKL_INT  rows, cols;
//     MKL_INT *rows_start = nullptr;
//     MKL_INT *rows_end   = nullptr;
//     MKL_INT *col_ind    = nullptr;
//     double  *vals       = nullptr;

//     check_mkl(
//         mkl_sparse_d_export_csr(
//             graph_csr, &indexing,
//             &rows, &cols,
//             &rows_start, &rows_end,
//             &col_ind, &vals
//         ),
//         "export csr failed"
//     );

//     const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

//     int nthreads;
//     // Must query inside parallel region to get actual thread count
//     #pragma omp parallel
//     {
//         #pragma omp single
//         nthreads = omp_get_num_threads();
//     }

//     MKL_INT segs_per_thread = 64;
//     MKL_INT total_segments = segs_per_thread * nthreads;
//     MKL_INT segment_size = rows / total_segments;
//     MKL_INT remainder = rows % total_segments;

//     b_c = std::min(b_c, n2);
//     b_c = std::max(b_c, 1);
//     // Q : number of blocks in dense matrices
//     const int Q = (n2 + b_c - 1) / b_c;
//     for (int q = 0; q < Q; ++q)
//     {
//         const int j_start = q * b_c;
//         const int j_end   = std::min(j_start + b_c, n2);
//         const int panel_w = j_end - j_start;

//         #pragma omp parallel
//         {
//             int tid = omp_get_thread_num();

//             for (int s = 0; s < segs_per_thread; ++s) {
//                 // --- Option A: Blocked (contiguous segments per thread) ---
//                 MKL_INT seg_id = tid * segs_per_thread + s;

//                 // --- Option B: Cyclic (interleaved segments per thread) ---
//                 // MKL_INT seg_id = s * nthreads + tid;

//                 // Handle remainder: first 'remainder' segments get +1 row
//                 MKL_INT start = seg_id * segment_size + (seg_id < remainder ? seg_id : remainder);
//                 MKL_INT end   = start + segment_size  + (seg_id < remainder ? 1 : 0);

//                 for (MKL_INT i = start; i < end; ++i) {
//                     double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

//                     if (beta == 0.0)
//                         memset(Ci_panel, 0, panel_w * sizeof(double));
//                     else if (beta != 1.0) {
//                         #pragma omp simd
//                         for (int j = 0; j < panel_w; ++j)
//                             Ci_panel[j] *= beta;
//                     }
//                     // beta == 1.0 → no scaling needed, C is already correct

//                     if (alpha == 0.0) continue;

//                     const MKL_INT row_s = rows_start[i] - base;
//                     const MKL_INT row_e = rows_end[i]   - base;

//                     for (MKL_INT p = row_s; p < row_e; ++p)
//                     {
//                         const MKL_INT k   = col_ind[p] - base;
//                         const double  aik = alpha * vals[p];

//                         const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

//                         #pragma omp simd
//                         for (int j = 0; j < panel_w; ++j)
//                             Ci_panel[j] += aik * Bk_panel[j];
//                     }
//                 }
//             }
//         }
//     }
// }


// ============================================================
// BLOCKED: jik loop order   (Algorithm 2 — panel blocked)
//
//  Outer loop:  j = 0, b_c, 2*b_c, ...  (column panels of B/C)
//  Middle loop: i = 0 .. n1-1           (rows of A, parallelised)
//  Inner loop:  nonzeros in row i of A
//
//  Each stage only touches a NARROW strip of B and C (b_c cols),
//  so those strips fit in L2/L3 cache → far fewer cache misses.
// ============================================================
void omp_spmm_jik(double               alpha,
                  sparse_matrix_t      graph_csr,
                  MKL_INT              n1,
                  int                  n2,       // total columns
                  double               beta,
                  const double*        B,        // [n1 x n2] row-major
                  double*              C,        // [n1 x n2] row-major
                  int                  b_c      // panel width  (e.g. 256)
                  )
{
    auto time_0 = std::chrono::high_resolution_clock::now();
    // -------------------------------------------------------
    // Export CSR once — reused across all panels
    // -------------------------------------------------------
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    auto time_1 = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------
    // Clamp b_c to [1, n2]
    // -------------------------------------------------------
    b_c = std::min(b_c, n2);
    b_c = std::max(b_c, 1);

    // Number of column panels
    const int Q = (n2 + b_c - 1) / b_c;

    // ----------------------------------------------------------------
    // Allocate B panel buffer  [n1 x b_c]  — contiguous, reused each stage
    // All rows of B for the current panel packed tightly together
    // ----------------------------------------------------------------
    std::vector<double> B_panel((size_t)n1 * b_c);
    auto load_b_col_block = 0.0;
    // -------------------------------------------------------
    // Outer loop over column panels  j = 0, b_c, 2*b_c, ...
    // -------------------------------------------------------
    for (int q = 0; q < Q; ++q)
    {
        const int j_start = q * b_c;                        // first col in panel
        const int j_end   = std::min(j_start + b_c, n2);   // one-past-last col
        const int panel_w = j_end - j_start;                // actual width (last panel may be smaller)

        
        auto t0 = std::chrono::high_resolution_clock::now();
        // ------------------------------------------------------------
        // STEP 1: Pack B[:, j_start:j_end] → B_panel[:, 0:panel_w]
        //         Each row of B: copy panel_w doubles
        //         Done in parallel — each thread handles its own rows
        // ------------------------------------------------------------
        // #pragma omp parallel for schedule(static) num_threads(num_threads)
        #pragma omp parallel for schedule(guided)
        for (MKL_INT k = 0; k < n1; ++k)
        {
            const double* __restrict__ Bk       = B       + (size_t)k * n2      + j_start;
            double*       __restrict__ Bk_panel = B_panel.data() + (size_t)k * b_c;

            // copy panel_w doubles — compiler will vectorise
            #pragma omp simd
            for (int j = 0; j < panel_w; ++j)
                Bk_panel[j] = Bk[j];
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        load_b_col_block += std::chrono::duration<double, std::milli>(t1 - t0).count();
        // All threads done packing before we compute


        // -------------------------------------------------------
        // Middle loop over rows of A  — parallelised
        // -------------------------------------------------------
        // #pragma omp parallel for schedule(static) num_threads(num_threads)
        #pragma omp parallel for schedule(guided)
        for (MKL_INT i = 0; i < rows; ++i)
        {
            // Pointer to the panel slice of output row i
            //   C[i, j_start : j_end]
            double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

            if (beta == 0.0)
                memset(Ci_panel, 0, panel_w * sizeof(double));
            else if (beta != 1.0) {
                #pragma omp simd
                for (int j = 0; j < panel_w; ++j)
                    Ci_panel[j] *= beta;
            }

            if (alpha == 0.0) continue;

            const MKL_INT row_s = rows_start[i] - base;
            const MKL_INT row_e = rows_end[i]   - base;

            // Inner loop over nonzeros in row i
            for (MKL_INT p = row_s; p < row_e; ++p)
            {
                const MKL_INT k   = col_ind[p] - base;
                const double  aik = alpha * vals[p];

                // B_panel is contiguous — no stride-n2 jumps
                // loaded ONCE in STEP 1, reused for all rows i here
                const double* __restrict__ Bk_panel = B_panel.data() + (size_t)k * b_c;

                // Vectorised accumulation over the panel
                #pragma omp simd
                for (int j = 0; j < panel_w; ++j)
                    Ci_panel[j] += aik * Bk_panel[j];
            }
        }
        // after this omp parallel for, all threads are done with panel q
    }
    auto time_2 = std::chrono::high_resolution_clock::now();
    auto extract_csr = std::chrono::duration<double, std::milli>(time_1 - time_0).count();
    auto mult = std::chrono::duration<double, std::milli>(time_2 - time_1).count();
    // std::cout << "inside omp blocked b_c= " << b_c << " extract cr " << extract_csr << " mult " << mult << " load B " << load_b_col_block << " ms\n";
}

struct BlockedCSR
{
    // Original MKL buffers — no copy
    const MKL_INT* col_ind;
    const double*  vals;
    MKL_INT        base;
    MKL_INT         nnz;
    // For each k-panel qk: rows_start_k[qk][i] and rows_end_k[qk][i]
    // Both point as indices into col_ind/vals
    int                              Qk;
    int                              b_k;
    std::vector<std::vector<MKL_INT>> rows_start_k;  // [Qk][rows]
    std::vector<std::vector<MKL_INT>> rows_end_k;    // [Qk][rows]
};

// ============================================================
// Build BlockedCSR from MKL handle — ONE pass over CSR
// Requires col_ind already sorted per row (MKL guarantees this)
// b_k is col seg size
// ============================================================
BlockedCSR build_blocked_csr(sparse_matrix_t graph_csr,
                              MKL_INT         n1,
                              int             b_k)
{
    BlockedCSR bcsr;

    // Export MKL CSR — no copy
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start_mkl = nullptr;
    MKL_INT *rows_end_mkl   = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start_mkl, &rows_end_mkl,
            const_cast<MKL_INT**>(&bcsr.col_ind),
            const_cast<double**> (&bcsr.vals)
        ),
        "export csr failed"
    );
    bcsr.base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    bcsr.b_k  = b_k;
    bcsr.Qk   = (n1 + b_k - 1) / b_k;
    bcsr.nnz = rows_end_mkl[rows-1] - rows_start_mkl[0];

    bcsr.rows_start_k.assign(bcsr.Qk, std::vector<MKL_INT>(rows));
    bcsr.rows_end_k  .assign(bcsr.Qk, std::vector<MKL_INT>(rows));

    const MKL_INT base = bcsr.base;

    // ----------------------------------------------------------------
    // ONE parallel pass over all rows
    // For each row i, scan col_ind once → fill all Qk panel pointers
    // ----------------------------------------------------------------
    // #pragma omp parallel for schedule(dynamic) 
    for (MKL_INT i = 0; i < rows; ++i)
    {
        const MKL_INT row_s = rows_start_mkl[i] - base;
        const MKL_INT row_e = rows_end_mkl[i]   - base;
        // if (i >= 0 && i < 5) printf("i %d row_s %d  row_e %d", i, row_s, row_e);
        // Walk through nonzeros once, assign to k-panels using lower_bound
        // Since col_ind is sorted, lower_bound per panel is O(log(nnz_i))
        // Total cost per row: O(Qk * log(nnz_i))  — done ONCE at setup
        for (int qk = 0; qk < bcsr.Qk; ++qk)
        {
            const MKL_INT k_start = (MKL_INT)qk * b_k + base;
            const MKL_INT k_end   = std::min(k_start + (MKL_INT)b_k,
                                             n1 + base);
            
            // lower_bound on sorted col_ind — no full scan
            bcsr.rows_start_k[qk][i] =
                std::lower_bound(bcsr.col_ind + row_s,
                                 bcsr.col_ind + row_e,
                                 k_start) - bcsr.col_ind;
            bcsr.rows_end_k[qk][i] =
                std::lower_bound(bcsr.col_ind + row_s,
                                 bcsr.col_ind + row_e,
                                 k_end) - bcsr.col_ind;
            // if (i >= 0 && i < 5) printf("i %d k_start %d  k_end %d   bcsr.rows_start_k[%d][%d] %d  bcsr.rows_end_k[%d][%d] %d\n\n", i, k_start, k_end, qk, i, bcsr.rows_start_k[qk][i], qk, i, bcsr.rows_end_k[qk][i]);
        }
    }

    return bcsr;
}

// ============================================================
// Build BlockedCSR from MKL handle — ONE pass over CSR
// Requires col_ind already sorted per row (MKL guarantees this)
// ============================================================
BlockedCSR build_2d_blocked_csr(sparse_matrix_t graph_csr,
                              MKL_INT         n1,
                              int             b_k,
                              int              b_r,
                              std::vector<MKL_INT>& block_starts
                              )
{
    BlockedCSR bcsr;

    // Export MKL CSR — no copy
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start_mkl = nullptr;
    MKL_INT *rows_end_mkl   = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start_mkl, &rows_end_mkl,
            const_cast<MKL_INT**>(&bcsr.col_ind),
            const_cast<double**> (&bcsr.vals)
        ),
        "export csr failed"
    );
    bcsr.base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    bcsr.b_k  = b_k;
    bcsr.Qk   = (n1 + b_k - 1) / b_k;
    bcsr.nnz = rows_end_mkl[rows-1] - rows_start_mkl[0];

    bcsr.rows_start_k.assign(bcsr.Qk, std::vector<MKL_INT>(rows));
    bcsr.rows_end_k  .assign(bcsr.Qk, std::vector<MKL_INT>(rows));

    const MKL_INT base = bcsr.base;

    // ----------------------------------------------------------------
    // ONE parallel pass over all rows
    // For each row i, scan col_ind once → fill all Qk panel pointers
    // ----------------------------------------------------------------
    // #pragma omp parallel for schedule(dynamic) 
    for (MKL_INT i = 0; i < rows; ++i)
    {
        const MKL_INT row_s = rows_start_mkl[i] - base;
        const MKL_INT row_e = rows_end_mkl[i]   - base;
        // if (i >= 0 && i < 5) printf("i %d row_s %d  row_e %d", i, row_s, row_e);
        // Walk through nonzeros once, assign to k-panels using lower_bound
        // Since col_ind is sorted, lower_bound per panel is O(log(nnz_i))
        // Total cost per row: O(Qk * log(nnz_i))  — done ONCE at setup
        for (int qk = 0; qk < bcsr.Qk; ++qk)
        {
            const MKL_INT k_start = (MKL_INT)qk * b_k + base;
            const MKL_INT k_end   = std::min(k_start + (MKL_INT)b_k,
                                             n1 + base);
            
            // lower_bound on sorted col_ind — no full scan
            bcsr.rows_start_k[qk][i] =
                std::lower_bound(bcsr.col_ind + row_s,
                                 bcsr.col_ind + row_e,
                                 k_start) - bcsr.col_ind;
            bcsr.rows_end_k[qk][i] =
                std::lower_bound(bcsr.col_ind + row_s,
                                 bcsr.col_ind + row_e,
                                 k_end) - bcsr.col_ind;
            // if (i >= 0 && i < 5) printf("i %d k_start %d  k_end %d   bcsr.rows_start_k[%d][%d] %d  bcsr.rows_end_k[%d][%d] %d\n\n", i, k_start, k_end, qk, i, bcsr.rows_start_k[qk][i], qk, i, bcsr.rows_end_k[qk][i]);
        }
    }


    MKL_INT total_nnz = bcsr.nnz;
    MKL_INT target_nnz = total_nnz / b_r;
    // std::vector<MKL_INT> block_starts;
    block_starts.push_back(0);
    MKL_INT cum_nnz = 0;
    MKL_INT block_idx = 0;
    for (MKL_INT row = 0; row < n1 && block_idx < b_r - 1; ++row) {
        cum_nnz += rows_start_mkl[row+1] - rows_start_mkl[row];
        if (cum_nnz >= (block_idx + 1) * target_nnz) {
            block_starts.push_back(row + 1);
            block_idx++;
        }
    }
    block_starts.push_back(rows); // last block to n
    while (block_starts.size() <= b_r)
        block_starts.push_back(rows);

    return bcsr;
}

// ============================================================
// SpMM using BlockedCSR — clean and direct
// Blocking columns on both csr matrix and dense matrices 
// b_k is number of blocks in csr matrix
// b_c is size of columns in each block of dense matrix     
// ============================================================
void omp_spmm_jik_blockedcsr(double  alpha, sparse_matrix_t graph_csr, MKL_INT  n1, int  n2, double beta, const double*  B, double*  C, int  b_c, int   b_k) {
    // auto tt0 = std::chrono::high_resolution_clock::now();
    BlockedCSR bcsr = build_blocked_csr(graph_csr, n1, b_k);
    // auto tt1 = std::chrono::high_resolution_clock::now();
    // double ms = std::chrono::duration<double, std::milli>(tt1 - tt0).count();
    // printf("time of building %e\n", ms);

    const MKL_INT rows  = n1;
    const int     Q     = (n2 + b_c - 1) / b_c;
    const int     Qk    = bcsr.Qk;
    const MKL_INT base  = bcsr.base;

    b_c = std::min(b_c, n2);
    // ----------------------------------------------------------------
    // j-panel loop (columns of B/C)
    // ----------------------------------------------------------------
    for (int q = 0; q < Q; ++q)
    {
        const int j_start = q * b_c;
        const int j_end   = std::min(j_start + b_c, n2);
        const int panel_w = j_end - j_start;

        // k-panel loop (columns of A = rows of B)
        for (int qk = 0; qk < Qk; ++qk)
        {
            const MKL_INT* __restrict__ rs = bcsr.rows_start_k[qk].data();
            const MKL_INT* __restrict__ re = bcsr.rows_end_k[qk].data();

            // B tile: bcsr.b_k rows × panel_w cols
            // stays in L2/L3 while all rows i are processed
            #pragma omp parallel for schedule(guided) 
            for (MKL_INT i = 0; i < rows; ++i)
            {
                // if ( q == 0 && qk == 0 && i == 0 && omp_get_thread_num() == 0) {
                //     int nthreads = omp_get_num_threads();
                //     printf("Number of threads in this region: %d\n", nthreads);
                // }
                double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

                // Apply beta only at first k-panel
                if (qk == 0) {
                    if (beta == 0.0)
                        memset(Ci_panel, 0, panel_w * sizeof(double));
                    else if (beta != 1.0) {
                        #pragma omp simd
                        for (int j = 0; j < panel_w; ++j)
                            Ci_panel[j] *= beta;
                    }
                }

                if (alpha == 0.0) continue;

                // Directly use precomputed start/end for this k-panel
                // NO scanning, NO branching on col_ind range
                const MKL_INT p_start = rs[i];
                const MKL_INT p_end   = re[i];

                for (MKL_INT p = p_start; p < p_end; ++p)
                {
                    // if( i < 2 )
                    //     printf(" p start %d p end %d\n", p_start, p_end);
                    const MKL_INT k   = bcsr.col_ind[p] - base;
                    const double  aik = alpha * bcsr.vals[p];
                    const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] += aik * Bk_panel[j];
                }
            }
        }
    }
}


// ============================================================
// SpMM using Blocking of CSR along rows and blocking dense matrices along column
// b_r is number of rows  in each block of csr matrix
// b_c is size of columns in each block of dense matrix 
// ============================================================
void omp_spmm_rik(double          alpha,
                  sparse_matrix_t graph_csr,
                  MKL_INT         n1,
                  int             n2,
                  double          beta,
                  const double*   B,
                  double*         C,
                  int             b_r,
                  int             b_c)
{

    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    b_r = std::min((int)n1, std::max(1, b_r));
    b_c = std::min(n2,      std::max(1, b_c));

    const int Qr = (n1 + b_r - 1) / b_r;
    const int Q  = (n2 + b_c - 1) / b_c;

    for (int qr = 0; qr < Qr; ++qr)
    {
        const MKL_INT i_start = (MKL_INT)qr * b_r;
        const MKL_INT i_end   = std::min(i_start + (MKL_INT)b_r, n1);

        
        for (int q = 0; q < Q; ++q)
        {
            const int j_start = q * b_c;
            const int j_end   = std::min(j_start + b_c, n2);
            const int panel_w = j_end - j_start;

            // rows_start/rows_end/col_ind accessed sequentially — no random access
            #pragma omp parallel for schedule(guided) 
            for (MKL_INT i = i_start; i < i_end; ++i)
            {
                // if ( qr == 0 && q == 0 && i == 0 && omp_get_thread_num() == 0) {
                //     int nthreads = omp_get_num_threads();
                //     printf("Number of threads in this region: %d\n", nthreads);
                // }
                double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

                if (beta == 0.0)
                    memset(Ci_panel, 0, panel_w * sizeof(double));
                else if (beta != 1.0) {
                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] *= beta;
                }

                if (alpha == 0.0) continue;

                const MKL_INT row_s = rows_start[i] - base;
                const MKL_INT row_e = rows_end[i]   - base;

                for (MKL_INT p = row_s; p < row_e; ++p)
                {
                    const MKL_INT k   = col_ind[p] - base;
                    const double  aik = alpha * vals[p];
                    const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] += aik * Bk_panel[j];
                }
            }
        }
    }
}

void omp_spmm_rik_v2(double          alpha,
                  sparse_matrix_t graph_csr,
                  MKL_INT         n1,
                  int             n2,
                  double          beta,
                  const double*   B,
                  double*         C,
                  int             b_r,
                  int             b_c)
{

    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    b_r = std::min((int)n1, std::max(1, b_r));
    b_c = std::min(n2,      std::max(1, b_c));

    const int Qr = (n1 + b_r - 1) / b_r;
    const int Q  = (n2 + b_c - 1) / b_c;

    std::vector<double> times(Q * Qr);
    
    for (int qr = 0; qr < Qr; ++qr)
    {
        const MKL_INT i_start = (MKL_INT)qr * b_r;
        const MKL_INT i_end   = std::min(i_start + (MKL_INT)b_r, n1);

        #pragma omp parallel for schedule(guided) 
        for (int q = 0; q < Q; ++q)
        {

            auto t1 = std::chrono::high_resolution_clock::now();
            const int j_start = q * b_c;
            const int j_end   = std::min(j_start + b_c, n2);
            const int panel_w = j_end - j_start;

            // rows_start/rows_end/col_ind accessed sequentially — no random access
            
            for (MKL_INT i = i_start; i < i_end; ++i)
            {
                // if ( qr == 0 && q == 0 && i == 0 && omp_get_thread_num() == 0) {
                //     int nthreads = omp_get_num_threads();
                //     printf("Number of threads in this region: %d\n", nthreads);
                // }
                double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

                if (beta == 0.0)
                    memset(Ci_panel, 0, panel_w * sizeof(double));
                else if (beta != 1.0) {
                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] *= beta;
                }

                if (alpha == 0.0) continue;

                const MKL_INT row_s = rows_start[i] - base;
                const MKL_INT row_e = rows_end[i]   - base;

                for (MKL_INT p = row_s; p < row_e; ++p)
                {
                    const MKL_INT k   = col_ind[p] - base;
                    const double  aik = alpha * vals[p];
                    const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] += aik * Bk_panel[j];
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            times[qr * Q + q] = ms;
        }
    }
    double max_time=times[0], min_time=times[0], avg_time, sum_time=0.0;
    for(int i=0; i<times.size(); i++) {
        max_time = std::max(max_time, times[i]);
        max_time = std::min(min_time, times[i]);
        sum_time += times[i];
    }
    avg_time = sum_time / times.size();

    printf("times: min %e max %e avg %e sum %e\n", min_time, max_time, avg_time, sum_time);
}

void omp_spmm_csr_rowblck_nnz(double  alpha, sparse_matrix_t graph_csr, MKL_INT n1, int n2, double beta, const double* B, double* C, int b_r, int b_c){
    std::cout << "bc " << b_c << " br " <<b_r << " n2 " << n2 << " n1 " << n1 << std::endl;
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );
    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    b_r = std::min((int)n1, std::max(1, b_r));
    b_c = std::min(n2,      std::max(1, b_c));
    const int Q  = (n2 + b_c - 1) / b_c;
    auto tt0 = std::chrono::high_resolution_clock::now();
    MKL_INT total_nnz = rows_start[rows-1] - rows_start[0];
    MKL_INT target_nnz = total_nnz / b_r;
    std::vector<MKL_INT> block_starts;
    block_starts.push_back(0);
    MKL_INT cum_nnz = 0;
    MKL_INT block_idx = 0;
    for (MKL_INT row = 0; row < rows && block_idx < b_r - 1; ++row) {
        cum_nnz += rows_start[row+1] - rows_start[row];
        if (cum_nnz >= (block_idx + 1) * target_nnz) {
            block_starts.push_back(row + 1);
            block_idx++;
        }
    }
    block_starts.push_back(rows); // last block to n
    // If we ended up with fewer blocks, pad with empty blocks
    while (block_starts.size() <= b_r)
        block_starts.push_back(rows);
    auto tt1 = std::chrono::high_resolution_clock::now();
    double mss = std::chrono::duration<double, std::milli>(tt1 - tt0).count();
    std::cout << "times to blocking into " << b_r << " row blocks is " << mss << std::endl;

    // std::vector<double> times(Q * b_r);
     
    #pragma omp parallel for collapse(2) schedule(guided) 
    for (int q = 0; q < Q; ++q){
        for (int qr = 0; qr < b_r; ++qr){
            // auto t1 = std::chrono::high_resolution_clock::now();

            const int j_start = q * b_c;
            const int j_end = std::min(j_start + b_c, n2);
            const int panel_w = j_end - j_start;
            const MKL_INT i_start = (MKL_INT) block_starts[qr] ;
            const MKL_INT i_end   = (MKL_INT) block_starts[qr+1] ;
            for (MKL_INT i = i_start; i < i_end; ++i)
            {
                double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

                if (beta == 0.0)
                    memset(Ci_panel, 0, panel_w * sizeof(double));
                else if (beta != 1.0) {
                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] *= beta;
                }

                if (alpha == 0.0) continue;

                const MKL_INT row_s = rows_start[i] - base;
                const MKL_INT row_e = rows_end[i]   - base;

                for (MKL_INT p = row_s; p < row_e; ++p)
                {
                    const MKL_INT k   = col_ind[p] - base;
                    const double  aik = alpha * vals[p];
                    const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] += aik * Bk_panel[j];
                }
            }
            // auto t2 = std::chrono::high_resolution_clock::now();
            // double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            // times[qr * Q + q] = ms;
        }
    }
    // double max_time=times[0], min_time=times[0], avg_time, sum_time=0.0;
    // for(int i=0; i<times.size(); i++) {
    //     max_time = std::max(max_time, times[i]);
    //     min_time = std::min(min_time, times[i]);
    //     sum_time += times[i];
    // }
    // avg_time = sum_time / times.size();

    // printf("times: min %e max %e avg %e sum %e\n", min_time, max_time, avg_time, sum_time);
    // measure_divergence(times);
}

void omp_spmm_csr_2d_blck_nnz(double  alpha, sparse_matrix_t graph_csr, MKL_INT n1, int n2, double beta, const double* B, double* C, int b_r, int b_c){

    b_r = std::min((int)n1, std::max(1, b_r));
    b_c = std::min(n2,      std::max(1, b_c));

    // const int Qr = (n1 + b_r - 1) / b_r;
    const int Q  = (n2 + b_c - 1) / b_c;

    auto tt0 = std::chrono::high_resolution_clock::now();
    std::vector<MKL_INT> block_starts;
    BlockedCSR bcsr = build_2d_blocked_csr(graph_csr, n1, n1/2, b_r, block_starts);
    const int     Qk    = bcsr.Qk;
    const MKL_INT base  = bcsr.base;
    auto tt1 = std::chrono::high_resolution_clock::now();
    double mss = std::chrono::duration<double, std::milli>(tt1 - tt0).count();
    // printf("times to blocking  %e Qk %d \n", mss, Qk);

    // std::vector<double> times(Q * b_r);
    // std::vector<double> times_2(Q * b_r * Qk );
     
    #pragma omp parallel for collapse(2) schedule(guided) 
    for (int q = 0; q < Q; ++q){
        for (int qr = 0; qr < b_r; ++qr){
            // auto t1 = std::chrono::high_resolution_clock::now();

            const int j_start = q * b_c;
            const int j_end   = std::min(j_start + b_c, n2);
            const int panel_w = j_end - j_start;
            const MKL_INT i_start = (MKL_INT) block_starts[qr] ;
            const MKL_INT i_end   = (MKL_INT) block_starts[qr+1] ;

            for (int qk = 0; qk < Qk; ++qk)
            {
                // auto t11 = std::chrono::high_resolution_clock::now();
                const MKL_INT* __restrict__ rs = bcsr.rows_start_k[qk].data();
                const MKL_INT* __restrict__ re = bcsr.rows_end_k[qk].data();
                for (MKL_INT i = i_start; i < i_end; ++i)
                {
                    double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

                    // if (beta == 0.0)
                    //     memset(Ci_panel, 0, panel_w * sizeof(double));
                    // else if (beta != 1.0) {
                    //     #pragma omp simd
                    //     for (int j = 0; j < panel_w; ++j)
                    //         Ci_panel[j] *= beta;
                    // }

                    // if (alpha == 0.0) continue;

                    const MKL_INT row_s = rs[i] - base;
                    const MKL_INT row_e = re[i] - base;

                    for (MKL_INT p = row_s; p < row_e; ++p)
                    {
                        const MKL_INT k   = bcsr.col_ind[p] - base;
                        const double  aik = alpha * bcsr.vals[p];
                        const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

                        #pragma omp simd
                        for (int j = 0; j < panel_w; ++j)
                            Ci_panel[j] += aik * Bk_panel[j];
                    }
                }
                // auto t22 = std::chrono::high_resolution_clock::now();
                // double mss = std::chrono::duration<double, std::milli>(t22 - t11).count();
                // times_2[qk * (Q * b_r) + qr * Q + q] = mss;
                // if (qr == 10 ) times_2[(Qk * Q) + 2 * q + qk] = mss;
            }
            
            // auto t2 = std::chrono::high_resolution_clock::now();
            // double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            // times[qr * Q + q] = ms;
        }
    }
    // double max_time=times[0], min_time=times[0], avg_time, sum_time=0.0;
    // for(int i=0; i<times.size(); i++) {
    //     max_time = std::max(max_time, times[i]);
    //     min_time = std::min(min_time, times[i]);
    //     sum_time += times[i];
    // }
    // avg_time = sum_time / times.size();

    // printf("times: min %e max %e avg %e sum %e\n", min_time, max_time, avg_time, sum_time);

    // max_time=times_2[0]; min_time=times_2[0]; sum_time=0.0;
    // for(int i=0; i<times_2.size(); i++) {
    //     min_time = std::min(min_time, times_2[i]);
    //     max_time = std::max(max_time, times_2[i]);
    //     sum_time += times_2[i];
    // }
    // avg_time = sum_time / times_2.size();
    // printf("times 2 : min %e max %e avg %e sum %e\n", min_time, max_time, avg_time, sum_time);
    // measure_divergence(times);
    // measure_divergence(times_2);

    // measure_divergence(runtimes);
}

void omp_spmm_csr_rowblck_nnz_nomklcsr(
    double  alpha,
    CSRMatrix& csr,     // Use your structure!
    MKL_INT n1, int n2,       // Output size n1 x n2
    double beta,
    const double* B,          // [n1 x n2]
    double* C,                // [n1 x n2]
    int b_r, int b_c)
{
    // Use the arrays from your CSRMatrix
    const MKL_INT* rows_start = csr.rowptr.data();
    const MKL_INT* col_ind    = csr.colidx.data();
    const double*  vals       = csr.values.data();
    const MKL_INT  rows = csr.nrows, cols = csr.ncols;
    const sparse_index_base_t indexing = csr.indexing;

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

    b_r = std::min((int)n1, std::max(1, b_r));
    b_c = std::min(n2,      std::max(1, b_c));

    const int Q = (n2 + b_c - 1) / b_c;    // # of col panels

    // Build NNZ-balanced row blocks
    MKL_INT total_nnz = rows_start[rows] - rows_start[0]; // (rowptr is size nrows+1)
    MKL_INT target_nnz = total_nnz / b_r;

    std::vector<MKL_INT> block_starts;
    block_starts.push_back(0);
    MKL_INT cum_nnz = 0;
    MKL_INT block_idx = 0;
    for (MKL_INT row = 0; row < rows && block_idx < b_r - 1; ++row) {
        cum_nnz += rows_start[row+1] - rows_start[row];
        if (cum_nnz >= (block_idx + 1) * target_nnz) {
            block_starts.push_back(row + 1);
            block_idx++;
        }
    }
    block_starts.push_back(rows);

    // If we ended up with fewer blocks, pad with empty blocks
    while ((int)block_starts.size() <= b_r)
        block_starts.push_back(rows);

    // Timing (for all tiles)
    std::vector<double> times(Q * b_r);

    // === Parallel over tiles ===
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int q = 0; q < Q; ++q) {
        for (int qr = 0; qr < b_r; ++qr) {
            auto t1 = std::chrono::high_resolution_clock::now();

            const MKL_INT i_start = block_starts[qr];
            const MKL_INT i_end   = block_starts[qr+1];

            const int j_start = q * b_c;
            const int j_end   = std::min(j_start + b_c, n2);
            const int panel_w = j_end - j_start;

            // Process only C(i,j) for i in [i_start, i_end), j in [j_start, j_end)

            for (MKL_INT i = i_start; i < i_end; ++i) {
                double* Ci_panel = C + (size_t)i * n2 + j_start; // ith row, col panel

                // Apply beta update (don't race: each thread gets unique rows/cols)
                if (beta == 0.0)
                    std::memset(Ci_panel, 0, panel_w * sizeof(double));
                else if (beta != 1.0) {
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] *= beta;
                }
                if (alpha == 0.0) continue;

                // For this output row, get the nonzero column indices and values
                MKL_INT row_s = rows_start[i] - base;
                MKL_INT row_e = rows_start[i+1] - base; // Only need rowptr[i+1]

                for (MKL_INT p = row_s; p < row_e; ++p) {
                    MKL_INT k = col_ind[p] - base;
                    double aik = alpha * vals[p];

                    const double* Bk_panel = B + (size_t)k * n2 + j_start; // row k, col panel

                    // Fused multiply-add for col panel
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] += aik * Bk_panel[j];
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            times[qr * Q + q] = ms;
        }
    }
    // Timing report (optional)
    double max_time = times[0], min_time = times[0], sum_time = 0.0;
    for (double ms : times) {
        if (ms > max_time) max_time = ms;
        if (ms < min_time) min_time = ms;
        sum_time += ms;
    }
    double avg_time = sum_time / times.size();
    printf("SpMM times: min %.6e ms, max %.6e ms, avg %.6e ms, sum %.6e\n",
           min_time, max_time, avg_time, sum_time);
}


////   TESTED NOT GOOD ////
void omp_spmm_rik_v3(double          alpha,
                  sparse_matrix_t graph_csr,
                  MKL_INT         n1,
                  int             n2,
                  double          beta,
                  const double*   B,
                  double*         C,
                  int             b_r,
                  int             b_c)
{

    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    check_mkl(
        mkl_sparse_d_export_csr(
            graph_csr, &indexing,
            &rows, &cols,
            &rows_start, &rows_end,
            &col_ind, &vals
        ),
        "export csr failed"
    );

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    b_r = std::min((int)n1, std::max(1, b_r));
    b_c = std::min(n2,      std::max(1, b_c));

    const int Qr = (n1 + b_r - 1) / b_r;
    const int Q  = (n2 + b_c - 1) / b_c;

    #pragma omp parallel for schedule(guided) 
    for (int qr = 0; qr < Qr; ++qr)
    {
        const MKL_INT i_start = (MKL_INT)qr * b_r;
        const MKL_INT i_end   = std::min(i_start + (MKL_INT)b_r, n1);

        
        for (int q = 0; q < Q; ++q)
        {
            const int j_start = q * b_c;
            const int j_end   = std::min(j_start + b_c, n2);
            const int panel_w = j_end - j_start;

            // rows_start/rows_end/col_ind accessed sequentially — no random access
            
            for (MKL_INT i = i_start; i < i_end; ++i)
            {
                // if ( qr == 0 && q == 0 && i == 0 && omp_get_thread_num() == 0) {
                //     int nthreads = omp_get_num_threads();
                //     printf("Number of threads in this region: %d\n", nthreads);
                // }
                double* __restrict__ Ci_panel = C + (size_t)i * n2 + j_start;

                if (beta == 0.0)
                    memset(Ci_panel, 0, panel_w * sizeof(double));
                else if (beta != 1.0) {
                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] *= beta;
                }

                if (alpha == 0.0) continue;

                const MKL_INT row_s = rows_start[i] - base;
                const MKL_INT row_e = rows_end[i]   - base;

                for (MKL_INT p = row_s; p < row_e; ++p)
                {
                    const MKL_INT k   = col_ind[p] - base;
                    const double  aik = alpha * vals[p];
                    const double* __restrict__ Bk_panel = B + (size_t)k * n2 + j_start;

                    #pragma omp simd
                    for (int j = 0; j < panel_w; ++j)
                        Ci_panel[j] += aik * Bk_panel[j];
                }
            }
        }
    }
}

std::vector<sparse_matrix_t> split_csr_block_transpose( const sparse_matrix_t csr_mat, MKL_INT n, MKL_INT k) {
    // Export CSR arrays from MKL handle
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *rows_start, *rows_end, *col_index;
    double *values;
    if (mkl_sparse_d_export_csr(csr_mat, &indexing, &rows, &cols,
          &rows_start, &rows_end, &col_index, &values) != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("Failed to export CSR from MKL handle");

    if (n != rows || n != cols)
        throw std::runtime_error("Provided n does not match matrix size");
    std::vector<MKL_INT> block_starts(k + 1); // block_starts[i] = start row of block i, block_starts[k] = n
    MKL_INT rows_per_block = n / k;
    MKL_INT leftover = n % k;
    MKL_INT cur_row = 0;
    for (int i = 0; i < k; ++i) {
        block_starts[i] = cur_row;
        MKL_INT this_block_rows = rows_per_block + (i < leftover ? 1 : 0);
        cur_row += this_block_rows;
    }
    block_starts[k] = n; // last block end
    std::vector<sparse_matrix_t> transposed_blocks;
    for (int i = 0; i < k; ++i) {
        MKL_INT block_start = block_starts[i];
        MKL_INT block_end   = block_starts[i+1];
        MKL_INT block_nrows = block_end - block_start;
        if (block_end > n)  throw std::runtime_error("Calculated block end beyond matrix size");

        // Calculate nnz in block
        // MKL_INT nnz_block = rows_end[block_end-1] - rows_start[block_start];
        MKL_INT nnz_block = rows_start[block_end] - rows_start[block_start];
        MKL_INT nnz_begin = rows_start[block_start];

        if (nnz_begin < 0 || nnz_begin + nnz_block > rows_end[n-1]) throw std::runtime_error("NNZ indices out of bounds");

        // Prepare block CSR arrays
        std::vector<MKL_INT> block_rowptr(block_nrows + 1);
        std::vector<MKL_INT> block_colidx(nnz_block);
        std::vector<double>  block_data(nnz_block);
        // Fill block_rowptr
        for (MKL_INT j = 0; j <= block_nrows; ++j) {
            if (block_start + j > n)    throw std::runtime_error("rowptr index out of bounds");
            block_rowptr[j] = rows_start[block_start + j] - rows_start[block_start];
        }
            

        // Copy colidx and data
        // MKL_INT nnz_begin = rows_start[block_start];
        for (MKL_INT j = 0; j < nnz_block; ++j) {
            block_colidx[j] = col_index[nnz_begin + j];
            block_data[j]   = values[nnz_begin + j];
        }
        // Create CSR MKL handle for the block
        sparse_matrix_t csr_block;
        if (mkl_sparse_d_create_csr(&csr_block, indexing,
            block_nrows, n,
            block_rowptr.data(), block_rowptr.data() + 1,
            block_colidx.data(), block_data.data()) != SPARSE_STATUS_SUCCESS)
            throw std::runtime_error("Failed to create CSR block");

        // mkl_sparse_order(csr_block); 
        mkl_sparse_optimize(csr_block); 

        // Convert to transpose CSR (not CSC!)
        sparse_matrix_t csr_transpose_block;
        if (mkl_sparse_convert_csr(
            csr_block, SPARSE_OPERATION_TRANSPOSE,
            &csr_transpose_block) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(csr_block);
            throw std::runtime_error("Failed to transpose block");
        }
        // Force MKL to make an independent copy
        matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t csr_transpose_copy;
        if (mkl_sparse_copy(csr_transpose_block, descr, &csr_transpose_copy) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(csr_block);
            mkl_sparse_destroy(csr_transpose_block);
            throw std::runtime_error("Failed to copy transposed block");
        }
        // mkl_sparse_optimize(csr_transpose_block); 
        mkl_sparse_optimize(csr_transpose_copy); 
        // Clean up block handle
        mkl_sparse_destroy(csr_transpose_block);
        mkl_sparse_destroy(csr_block);
        // Store the transposed CSR handle
        // transposed_blocks.push_back(csr_transpose_block);
        transposed_blocks.push_back(csr_transpose_copy); 
    }

    return transposed_blocks;
}

std::vector<sparse_matrix_t> split_csr_block_nnz_transpose( const sparse_matrix_t csr_mat, MKL_INT n, MKL_INT k, std::vector<MKL_INT>& block_starts) {
    // Export CSR arrays from MKL handle
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT* rows_start;
    MKL_INT* rows_end;
    MKL_INT* col_index;
    double* values;
    if (mkl_sparse_d_export_csr(csr_mat, &indexing, &rows, &cols,
                                &rows_start, &rows_end, &col_index, &values) != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("Failed to export CSR from MKL handle");

    if (n != rows || n != cols)
        throw std::runtime_error("Provided n does not match matrix size");

    // Compute total NNZ
    MKL_INT total_nnz = rows_start[n] - rows_start[0];
    MKL_INT target_nnz = total_nnz / k;
    MKL_INT leftover_nnz = total_nnz % k;

    // // Step 1: Create block boundaries for equal NNZ
    // std::vector<MKL_INT> block_starts;
    // block_starts.push_back(0);
    // MKL_INT cur_nnz = 0;
    // MKL_INT nnz_per_block;
    // MKL_INT block_count = 0;
    // for (MKL_INT i = 0, row = 0; block_count < k - 1 && row < n; ++block_count) {
    //     nnz_per_block = target_nnz + (block_count < leftover_nnz ? 1 : 0);
    //     cur_nnz = 0;
    //     MKL_INT block_row_start = block_starts.back();
    //     for (row = block_row_start; row < n; ++row) {
    //         MKL_INT row_nnz = rows_start[row + 1] - rows_start[row];
    //         if (cur_nnz + row_nnz > nnz_per_block && row > block_row_start) {
    //             // Avoid skipping too many rows, cut here
    //             break;
    //         }
    //         cur_nnz += row_nnz;
    //     }
    //     if (row == block_row_start) row = block_row_start + 1; // At least one row per block
    //     block_starts.push_back(row);
    // }
    // block_starts.push_back(n); // Ensure last block goes to the end

    // std::vector<MKL_INT> block_starts;
    block_starts.push_back(0);
    MKL_INT cum_nnz = 0;
    MKL_INT block_idx = 0;
    for (MKL_INT row = 0; row < n && block_idx < k - 1; ++row) {
        cum_nnz += rows_start[row+1] - rows_start[row];
        if (cum_nnz >= (block_idx + 1) * target_nnz) {
            block_starts.push_back(row + 1);
            block_idx++;
        }
    }
    block_starts.push_back(n); // last block to n


    // If we ended up with fewer blocks, pad with empty blocks
    while (block_starts.size() <= k)
        block_starts.push_back(n);

    std::vector<sparse_matrix_t> transposed_blocks;
    for (int bi = 0; bi < k; ++bi) {
        MKL_INT block_start = block_starts[bi];
        MKL_INT block_end = block_starts[bi + 1];

        // printf("block_start %d  block_end %d diff %d\n", block_start, block_end, block_end- block_start);
        MKL_INT block_nrows = block_end - block_start;
        if (block_end > n) throw std::runtime_error("Calculated block end beyond matrix size");

        MKL_INT nnz_begin = rows_start[block_start];
        MKL_INT nnz_block = rows_start[block_end] - rows_start[block_start];

        if (nnz_begin < 0 || nnz_begin + nnz_block > rows_start[n]) throw std::runtime_error("NNZ indices out of bounds");

        // Prepare block CSR arrays
        std::vector<MKL_INT> block_rowptr(block_nrows + 1);
        std::vector<MKL_INT> block_colidx(nnz_block);
        std::vector<double> block_data(nnz_block);

        // Fill block_rowptr
        for (MKL_INT j = 0; j <= block_nrows; ++j) {
            if (block_start + j > n) throw std::runtime_error("rowptr index out of bounds");
            block_rowptr[j] = rows_start[block_start + j] - rows_start[block_start];
        }
        // Copy colidx and data
        for (MKL_INT j = 0; j < nnz_block; ++j) {
            block_colidx[j] = col_index[nnz_begin + j];
            block_data[j] = values[nnz_begin + j];
        }
        // Create CSR MKL handle for the block
        sparse_matrix_t csr_block;
        if (mkl_sparse_d_create_csr(&csr_block, indexing,
            block_nrows, n,
            block_rowptr.data(), block_rowptr.data() + 1,
            block_colidx.data(), block_data.data()) != SPARSE_STATUS_SUCCESS)
            throw std::runtime_error("Failed to create CSR block");

        mkl_sparse_optimize(csr_block);

        // Convert to transpose CSR
        sparse_matrix_t csr_transpose_block;
        if (mkl_sparse_convert_csr(
            csr_block, SPARSE_OPERATION_TRANSPOSE,
            &csr_transpose_block) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(csr_block);
            throw std::runtime_error("Failed to transpose block");
        }
        // Force MKL to make an independent copy
        matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t csr_transpose_copy;
        if (mkl_sparse_copy(csr_transpose_block, descr, &csr_transpose_copy) != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(csr_block);
            mkl_sparse_destroy(csr_transpose_block);
            throw std::runtime_error("Failed to copy transposed block");
        }
        mkl_sparse_optimize(csr_transpose_copy);
        mkl_sparse_destroy(csr_transpose_block);
        mkl_sparse_destroy(csr_block);

        transposed_blocks.push_back(csr_transpose_copy);
    }

    return transposed_blocks;
}

// void omp_spmm_csr_trans_blocks_jik_v1(
//     double alpha,
//     const std::vector<sparse_matrix_t>& csc_blocks,
//     MKL_INT n1,            // total number of columns in sparse blocks
//     int n2,                // total number of columns in dense matrix B
//     double beta,
//     const double* B,       // (n1 x n2) matrix
//     double* C,             // Output (n1 x n2) matrix
//     int b_c                // Block width in B/C
// )
// {
//     b_c = std::min(b_c, n2);
//     b_c = std::max(b_c, 1);
//     const int Q = (n2 + b_c - 1) / b_c;

//     for (int q = 0; q < Q; ++q) {
//         int j_start = q * b_c;
//         int j_end = std::min(j_start + b_c, n2);
//         int panel_w = j_end - j_start;

//         MKL_INT row_offset = 0;
//         for (size_t b = 0; b < csc_blocks.size(); ++b) {
//             // Export CSC arrays for this block
//             sparse_index_base_t indexing;
//             MKL_INT block_rows, block_cols;
//             MKL_INT *col_start = nullptr;
//             MKL_INT *col_end = nullptr;
//             MKL_INT *row_ind = nullptr;
//             double *vals = nullptr;

//             check_mkl(
//                 mkl_sparse_d_export_csc(
//                     csc_blocks[b], &indexing,
//                     &block_rows, &block_cols,
//                     &col_start, &col_end, &row_ind, &vals
//                 ),
//                 "export csc failed"
//             );

//             const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;

//             // For each column in sparse block (csc: column major)
//             #pragma omp parallel for schedule(guided)
//             for (MKL_INT k = 0; k < block_cols; ++k) {
//                 const MKL_INT col_s = col_start[k] - base;
//                 const MKL_INT col_e = col_end[k] - base;

//                 const double* restrict Bk_panel = B + (size_t)k * n2 + j_start;

//                 for (MKL_INT p = col_s; p < col_e; ++p) {
//                     MKL_INT i = row_ind[p] - base; // row index within block
//                     double aik = alpha * vals[p];

//                     double* restrict Ci_panel = C + (size_t)(i + row_offset) * n2 + j_start;

//                     if (beta == 0.0 && q == 0)
//                         memset(Ci_panel, 0, panel_w * sizeof(double));
//                     else if (beta != 1.0 && q == 0) {
//                         #pragma omp simd
//                         for (int jj = 0; jj < panel_w; ++jj)
//                             Ci_panel[jj] *= beta;
//                     }
//                     if (alpha == 0.0) continue;

//                     #pragma omp simd
//                     for (int jj = 0; jj < panel_w; ++jj)
//                         Ci_panel[jj] += aik * Bk_panel[jj];
//                 }
//             }
//             row_offset += block_rows;
//         }
//     }
// }

// Multiply with vector of CSR transposed blocks
void omp_spmm_csr_trans_blocks_jik_v1(
    double alpha,
    const std::vector<sparse_matrix_t>& csr_blocks, // Transposed (CSR) blocks
    MKL_INT n1,            // total number of columns in sparse blocks
    int n2,                // total number of columns in dense matrix B
    double beta,
    const double* B,       // (n1 x n2) matrix
    double* C,             // Output (n1 x n2) matrix
    int b_c,                 // Block width in B/C 
    const std::vector<MKL_INT>& block_starts
)
{

    // =============================================
    // Apply beta to ALL of C, exactly once, upfront
    // =============================================
    if (beta == 0.0) {
        memset(C, 0, (size_t)n1 * n2 * sizeof(double));
    } else if (beta != 1.0) {
        #pragma omp parallel for
        for (MKL_INT i = 0; i < n1; ++i) {
            #pragma omp simd
            for (int jj = 0; jj < n2; ++jj)
                C[i * n2 + jj] *= beta;
        }
    }
    // beta has been applied; from here on, just accumulate

    if (alpha == 0.0) return;

    b_c = std::min(b_c, n2);
    b_c = std::max(b_c, 1);
    const int Q = (n2 + b_c - 1) / b_c;

    std::vector<double> times(Q * csr_blocks.size());
    for (int q = 0; q < Q; ++q) {
        int j_start = q * b_c;
        int j_end = std::min(j_start + b_c, n2);
        int panel_w = j_end - j_start;
        #pragma omp parallel for schedule(dynamic)
        for (size_t b = 0; b < csr_blocks.size(); ++b) {
            auto t1 = std::chrono::high_resolution_clock::now();
            // Export CSR arrays for this block (transposed CSR)
            sparse_index_base_t indexing;
            MKL_INT block_rows, block_cols;
            MKL_INT *rows_start = nullptr;
            MKL_INT *rows_end = nullptr;
            MKL_INT *col_ind = nullptr;
            double *vals = nullptr;
            check_mkl(
                mkl_sparse_d_export_csr(
                    csr_blocks[b], &indexing,
                    &block_rows, &block_cols,
                    &rows_start, &rows_end,
                    &col_ind, &vals
                ),
                "export transposed CSR failed"
            );
            
            const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
            // We assume original sparse matrix is n1*n1
            // MKL_INT row_offset;
            // per_block = n1 / csr_blocks.size();
            // MKL_INT leftover = n1 % csr_blocks.size();
            // MKL_INT row_offset = 0;
            // if ( b > leftover ) row_offset = leftover * (rows_per_block + 1) + (b - leftover) * rows_per_block;
            // else row_offset = b * (rows_per_block + 1);
            MKL_INT row_offset = block_starts[b];
            // if (q==0) printf( "nnz %d\n", rows_end[block_rows-1]-rows_start[0]);
            // printf(" -- q %d b %d block_rows %d block_cols %d row_offset %d \n", q, b, block_rows, block_cols, row_offset);
            // Now, for each row in this CSR block (which is a column in original block)
            
            for (MKL_INT k = 0; k < block_rows; ++k) { // k = column in original block, row in transpose
                const MKL_INT row_s = rows_start[k] - base;
                const MKL_INT row_e = rows_end[k] - base;
                if (row_s < row_e) {
                    // printf("33 [ col block num q %d csr block num b %d k %d\n", q, b, k);
                    // const double* Bk_panel = B + (size_t)(k + row_offset) * n2 + j_start;
                    const double* Bk_panel = B + (size_t)(k) * n2 + j_start;
                    // printf("33 ] col block num q %d csr block num b %d k %d\n", q, b, k);
                    for (MKL_INT p = row_s; p < row_e; ++p) {
                        MKL_INT i = col_ind[p] - base; // i = row in original block, column in transpose
                        // printf("44 col block num q %d csr block num b %d k %d i %d\n", q, b, k, i);
                        double aik = alpha * vals[p];
                        MKL_INT global_i = i + row_offset;
                        // printf("__col block num q %d csr block num b %d block_rows %d block_cols %d k %d i %d global_i %d\n", q, b, block_rows, block_cols, k, i, global_i);
                        // double* Ci_panel = C + (size_t)i * n2 + j_start;
                        double* Ci_panel = C + (size_t)global_i * n2 + j_start;

                        // if (beta == 0.0 && q == 0)
                        //     memset(Ci_panel, 0, panel_w * sizeof(double));
                        // else if (beta != 1.0 && q == 0) {
                        //     #pragma omp simd
                        //     for (int jj = 0; jj < panel_w; ++jj)
                        //         Ci_panel[jj] *= beta;
                        // }
                        // if (alpha == 0.0) continue;

                        #pragma omp simd
                        for (int jj = 0; jj < panel_w; ++jj)
                            Ci_panel[jj] += aik * Bk_panel[jj];
                    }
                }
                
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            // printf("thread %d time %e\n", omp_get_thread_num(), ms);
            times[csr_blocks.size() * q + b] = ms;
        }
    }

    double max_time=times[0], min_time=times[0], avg_time, sum_time=0.0;
    for(int i=0; i<times.size(); i++) {
        max_time = std::max(max_time, times[i]);
        max_time = std::min(min_time, times[i]);
        sum_time += times[i];
    }
    avg_time = sum_time / times.size();

    printf("times: min %e max %e avg %e sum %e\n", min_time, max_time, avg_time, sum_time);
}


// A = graph_csr is n1 x n1 CSR sparse matrix
// B, C are n1 x n2, both column-major
// Computes: C = alpha * A * B + beta * C
//
// Important:
//   - B and C must point to arrays of size at least n1*n2
//   - graph_csr must already be a valid MKL CSR matrix
//   - use MKL_INT consistently with how the matrix was built
//
void omp_spmm_col_omp(double alpha, sparse_matrix_t graph_csr, MKL_INT n1, int n2, double beta, const double* B, double* C, omp_sched_t sched, int chunk) {
// void omp_spmm_col_omp(double alpha, sparse_matrix_t graph_csr, MKL_INT n1, int n2, double beta, const double* B, double* C) {
    if (!graph_csr || !B || !C || n1 <= 0 || n2 <= 0) return;

    // Export CSR data from MKL handle
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *rows_start = nullptr, *rows_end = nullptr;
    MKL_INT *col_indx   = nullptr;
    double  *values     = nullptr;

    sparse_status_t st = mkl_sparse_d_export_csr( graph_csr, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);

    if (st != SPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "mkl_sparse_d_export_csr failed: %d\n", (int)st); return;}

    if (rows != n1 || cols != n1) {
        std::fprintf(stderr, "Dimension mismatch: graph_csr is %lld x %lld, expected %lld x %lld\n",
                     (long long)rows, (long long)cols,
                     (long long)n1,  (long long)n1);
        return;
    }

    const size_t total = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    if (beta == 0.0) {
        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < total; ++idx) {
            C[idx] = 0.0;
        }
    } else if (beta != 1.0) {
        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < total; ++idx) {
            C[idx] *= beta;
        }
    }
    omp_set_schedule(sched, chunk);
    #pragma omp parallel for schedule(runtime)
    for (int j = 0; j < n2; ++j) {
        const double* __restrict x = B + static_cast<size_t>(j) * n1;
        double*       __restrict y = C + static_cast<size_t>(j) * n1;

        for (MKL_INT i = 0; i < n1; ++i) {
            double sum = 0.0;

            // CSR row i
            for (MKL_INT p = rows_start[i]; p < rows_end[i]; ++p) {
                sum += values[p] * x[col_indx[p]];
            }

            y[i] += alpha * sum;
        } 
    }
}

void omp_spmm_col_omp_blck(double alpha, sparse_matrix_t graph_csr, MKL_INT n1, int n2, double beta, const double* B, double* C, int b_r, int b_c) {
    if (!graph_csr || !B || !C || n1 <= 0 || n2 <= 0) return;
    if (b_r <= 0) b_r = 1;
    if (b_c <= 0) b_c = 1;
    if (b_r > n1) b_r = (int)n1;
    if (b_c > n2) b_c = n2;

    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT* rows_start = nullptr;
    MKL_INT* rows_end   = nullptr;
    MKL_INT* col_indx   = nullptr;
    double*  values     = nullptr;

    sparse_status_t st = mkl_sparse_d_export_csr( graph_csr, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);

    if (st != SPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "mkl_sparse_d_export_csr failed: %d\n", (int)st); return;}

    if (rows != n1 || cols != n1) {
        std::fprintf(stderr,
                     "Dimension mismatch: graph_csr is %lld x %lld, expected %lld x %lld\n",
                     (long long)rows, (long long)cols,
                     (long long)n1, (long long)n1);
        return;
    }

    auto ceil_div_int = [](long long a, long long b) -> long long {
        return (a + b - 1) / b;
    };

    const MKL_INT row_blk_size = (MKL_INT)ceil_div_int((long long)n1, (long long)b_r);
    const int     col_blk_size = (int)ceil_div_int((long long)n2, (long long)b_c);

    for (int bc = 0; bc < b_c; ++bc) {
        const int cbeg = bc * col_blk_size;
        const int cend = std::min(cbeg + col_blk_size, n2);

        if (cbeg >= cend) continue;

        for (int br = 0; br < b_r; ++br) {
            const MKL_INT rbeg = br * row_blk_size;
            const MKL_INT rend = std::min<MKL_INT>(rbeg + row_blk_size, n1);

            if (rbeg >= rend) continue;

            // Parallelize INSIDE this block over output columns
            #pragma omp parallel for schedule(static)
            for (int j = cbeg; j < cend; ++j) {
                const double* __restrict Bj = B + (size_t)j * n1;
                double* __restrict Cj       = C + (size_t)j * n1;

                // scale this column segment of the tile by beta
                if (beta == 0.0) {
                    for (MKL_INT i = rbeg; i < rend; ++i) {
                        Cj[i] = 0.0;
                    }
                } else if (beta != 1.0) {
                    for (MKL_INT i = rbeg; i < rend; ++i) {
                        Cj[i] *= beta;
                    }
                }

                // compute this output column segment
                for (MKL_INT i = rbeg; i < rend; ++i) {
                    double sum = 0.0;
                    for (MKL_INT p = rows_start[i]; p < rows_end[i]; ++p) {
                        sum += values[p] * Bj[col_indx[p]];
                    }

                    Cj[i] += alpha * sum;
                }
            }
        }
    }
}

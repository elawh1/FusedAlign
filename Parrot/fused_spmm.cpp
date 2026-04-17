#include "fused_spmm.h"

// Given: A (n1 x n1), B (n1 x n2), C (n2 x n2), D (n1 x n2), block_size
// All dense matrices are column-major.
// A and C are sparse_matrix_t (MKL CSR).
// B and D are std::vector<double>.
// For each block of columns in n2, generate E_col (n1 x 1) and D_col (n1 x 1)
std::vector<double> blockwise_colsum_matmul( sparse_matrix_t A, sparse_matrix_t C, const std::vector<double>& B, size_t n1, size_t n2, int n_blocks, double scalar) {
    int numThreads = mkl_get_max_threads();
    printf("in blockwise_colsum_matmul nthreads %d\n", numThreads);
    std::vector<double> D( (size_t) n1*n2, 0.0);
    // Extract CSR views for A and C
    sparse_index_base_t A_indexing;
    MKL_INT A_rows, A_cols;
    MKL_INT *A_row_start = nullptr, *A_row_end = nullptr;
    MKL_INT *A_col_indx = nullptr;
    double *A_values = nullptr;
    sparse_status_t st = mkl_sparse_d_export_csr( A, &A_indexing, &A_rows, &A_cols, &A_row_start, &A_row_end, &A_col_indx, &A_values);
    if (st != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("A sparse mkl_sparse_d_export_csr failed");
    if (A_rows != n1 || A_cols != n1)
        throw std::runtime_error("A dimension mismatch in sparse_to_dense");
    sparse_index_base_t C_indexing;
    MKL_INT C_rows, C_cols;
    MKL_INT *C_row_start = nullptr, *C_row_end = nullptr;
    MKL_INT *C_col_indx = nullptr;
    double *C_values = nullptr;
    st = mkl_sparse_d_export_csr( C, &C_indexing, &C_rows, &C_cols, &C_row_start, &C_row_end, &C_col_indx, &C_values);
    if (st != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("C sparse mkl_sparse_d_export_csr failed");
    if (C_rows != n2 || C_cols != n2)
        throw std::runtime_error("C dimension mismatch in sparse_to_dense");


    // Determine index offset (0-based or 1-based)
    MKL_INT A_offset = (A_indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    MKL_INT C_offset = (C_indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    // Compute block size
    size_t block_size = (n2 + n_blocks - 1) / n_blocks;
  
    // Temporary storage for one column of E
    // E = B * C
    // std::vector<double> E_col(n1, 0.0);
    // Temporary storage for A * E_col (one column of partial D)
    // std::vector<double> D_col(n1, 0.0);

    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    // Loop over blocks
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        //col_start and col_end are global indices
        size_t col_start = block_idx * block_size;
        size_t col_end = std::min(col_start + block_size, n2);
        size_t current_block_size = col_end - col_start;
        // printf("- block idx %d col_start %d col_end %d current_block_size %d\n", block_idx, col_start, col_end, current_block_size);
        if (current_block_size == 0) break;
        
        // B_block: pointer to B columns [col_start : col_end)
        const double* B_block = &B[col_start * n1];
        // For each column j in C_block (which corresponds to columns [col_start:col_end) of C)
        // We need to process C_block^T, meaning we need column j of C^T = row j of C
        // for (size_t local_j = 0; local_j < current_block_size; ++local_j) {
        //     size_t global_j = col_start + local_j;
        #pragma omp parallel for
        for (size_t E_global_j = 0; E_global_j < n2; ++E_global_j) {
            // size_t global_j = col_start + local_j;
            
            // Clear E_col for this column
            // for (size_t i = 0; i < n1; ++i) {
            //     E_col[i] = 0.0;
            // }
            std::vector<double> E_col(n1, 0.0);
            
            // Compute E_col = B_block * C_block^T[:, local_j]
            // C_block^T[:, local_j] means column local_j of C_block^T
            // which is row global_j of C
            
            // Iterate over non-zeros in every row of C
            MKL_INT C_row_start_idx = C_row_start[E_global_j] - C_offset;
            MKL_INT C_row_end_idx = C_row_end[E_global_j] - C_offset;
            for (MKL_INT idx = C_row_start_idx; idx < C_row_end_idx; ++idx) {
                MKL_INT k = C_col_indx[idx] - C_offset; // column index in C
                // printf("-- block_idx %d E_global_j %d global col idx of B %d \n", block_idx, E_global_j, k);
                // Check if column k is within current B_block range
                if ((size_t)k >= col_start && (size_t)k < col_end) {
                    size_t local_k = (size_t)k - col_start;
                    
                    // Add B_block[:, local_k] to E_col (c_val is always 1.0)
                    // printf("--- global col idx of B %d B col idx local %d\n", k, local_k * n1);
                    const double* B_col = &B_block[local_k * n1];
                    for (size_t i = 0; i < n1; ++i) {
                        E_col[i] += B_col[i];
                    }
                }
                else if ((size_t)k >= col_end) {
                    // Since CSR columns are sorted, we can break early
                    break;
                }
            } 
            std::vector<double> D_col(n1, 0.0);
            // // Now compute D_col = A * E_col
            // Sparse matrix-vector multiplication: D_col = A * E_col
            for (size_t i = 0; i < n1; ++i) {
                MKL_INT a_row_start_idx = A_row_start[i] - A_offset;
                MKL_INT a_row_end_idx = A_row_end[i] - A_offset;
                
                double sum = 0.0;
                for (MKL_INT idx = a_row_start_idx; idx < a_row_end_idx; ++idx) {
                    MKL_INT j = A_col_indx[idx] - A_offset;
                    // double a_val = A_values[idx];
                    sum += E_col[j];
                }
                D_col[i] = sum;
            }
            // Accumulate D_col into column global_j of D
            // D is column-major: D(i, global_j) = D[global_j * n1 + i]
            // printf("-in E_global_j %d D : ", E_global_j);
            for (size_t i = 0; i < n1; ++i) {
                D[E_global_j * n1 + i] += scalar * D_col[i];
            }   
            // printf("\n");
        }
            
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return D;
 
}

// Given: A (n1 x n1), B (n1 x n2), C (n2 x n2), D (n1 x n2), block_size
// All dense matrices are column-major.
// A and C are sparse_matrix_t (MKL CSR).
// B and D are std::vector<double>.
// For each block of columns in n2, generate E_col (n1 x 1) and D_col (n1 x 1)
std::vector<double> blockwise_colsum_matmul_2( sparse_matrix_t A, sparse_matrix_t C, const std::vector<double>& B, size_t n1, size_t n2, int n_blocks) {

    std::vector<double> D( (size_t) n1*n2, 0.0);

    // Extract CSR views for A and C
    sparse_index_base_t A_indexing;
    MKL_INT A_rows, A_cols;
    MKL_INT *A_row_start = nullptr, *A_row_end = nullptr;
    MKL_INT *A_col_indx = nullptr;
    double *A_values = nullptr;
    sparse_status_t st = mkl_sparse_d_export_csr( A, &A_indexing, &A_rows, &A_cols, &A_row_start, &A_row_end, &A_col_indx, &A_values);
    if (st != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("A sparse mkl_sparse_d_export_csr failed");
    if (A_rows != n1 || A_cols != n1)
        throw std::runtime_error("A dimension mismatch in sparse_to_dense");

    sparse_index_base_t C_indexing;
    MKL_INT C_rows, C_cols;
    MKL_INT *C_row_start = nullptr, *C_row_end = nullptr;
    MKL_INT *C_col_indx = nullptr;
    double *C_values = nullptr;
    st = mkl_sparse_d_export_csr( C, &C_indexing, &C_rows, &C_cols, &C_row_start, &C_row_end, &C_col_indx, &C_values);
    if (st != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("C sparse mkl_sparse_d_export_csr failed");
    if (C_rows != n2 || C_cols != n2)
        throw std::runtime_error("C dimension mismatch in sparse_to_dense");


    // Determine index offset (0-based or 1-based)
    MKL_INT A_offset = (A_indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    MKL_INT C_offset = (C_indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
  
    size_t block_size = (n2 + n_blocks - 1) / n_blocks;

    // E_block = B_blcok * C -> D_blcok += E_block
    // Loop over blocks
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        //col_start and col_end are global indices
        size_t col_start = block_idx * block_size;
        size_t col_end = std::min(col_start + block_size, n2);
        size_t current_block_size = col_end - col_start;
        // printf("- block idx %d col_start %d col_end %d current_block_size %d\n", block_idx, col_start, col_end, current_block_size);
        if (current_block_size == 0) break;
        
        const double* B_block = &B[col_start * n1];
        
        std::vector<double> E_block(n1 * n2, 0.0);
        std::vector<double> D_block(n1 * n2, 0.0);
        // #pragma omp parallel for
        for (size_t C_row = 0; C_row < n2; ++C_row) {
            for (MKL_INT idx = C_row_start[C_row] - C_offset; idx <  C_row_end[C_row] - C_offset; ++idx) {
                MKL_INT C_col = C_col_indx[idx] - C_offset;
                if ((size_t)C_col >= col_start && (size_t)C_col < col_end) {
                    size_t local_block_col = (size_t)C_col - col_start;
                    const double* B_col = &B_block[local_block_col * n1];
                    //add corresponding colum of B for nonzero
                    for (size_t i = 0; i < n1; ++i) {
                        E_block[n1 * C_row + i] += B_col[i];
                    }
                }
                else if ((size_t)C_col >= col_end) {
                    // Since CSR columns are sorted, we can break early
                    break;
                }
            }
        }

        int numThreads = omp_get_max_threads();
        mkl_set_num_threads(numThreads);
        omp_set_num_threads(1);
        //SPMM  A*E_block
        matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL; descr.mode = SPARSE_FILL_MODE_FULL; descr.diag = SPARSE_DIAG_NON_UNIT;
        sparse_status_t st = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, SPARSE_LAYOUT_COLUMN_MAJOR, E_block.data(), n2, n1, 0.0, D_block.data(), n1);
        if (st != SPARSE_STATUS_SUCCESS) {  throw std::runtime_error("mkl_sparse_d_mm failed for A * E_block");}
        mkl_set_num_threads(1);
        omp_set_num_threads(numThreads);

        // Accumulate D_block into D
        for (size_t i = 0; i < n1 * n2; ++i) {
            // printf(" %e ", D_block[i]);
            D[i] += D_block[i];
        }   
              
    }
    
    return D;
 
}

std::vector<double> blockwise_outputcols_matmul( sparse_matrix_t A,  sparse_matrix_t C,  const std::vector<double>& B, 
    size_t n1, size_t n2, int n_blocks, double scalar) {

    int numThreads = mkl_get_max_threads();
    printf("in blockwise_outputcols_matmul nthreads %d\n", numThreads);
    if (B.size() != n1 * n2) throw std::runtime_error("B size mismatch");
    if (n_blocks <= 0)  throw std::runtime_error("n_blocks must be positive");

    std::vector<double> D(n1 * n2, 0.0);

    // Export C CSR so we can manually build E_block = B * C[J,:]^T
    sparse_index_base_t C_indexing;
    MKL_INT C_rows, C_cols;
    MKL_INT *C_row_start = nullptr, *C_row_end = nullptr;
    MKL_INT *C_col_indx  = nullptr;
    double  *C_values    = nullptr;

    sparse_status_t st = mkl_sparse_d_export_csr(
        C, &C_indexing, &C_rows, &C_cols,
        &C_row_start, &C_row_end, &C_col_indx, &C_values
    );
    if (st != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_export_csr(C) failed");
    }
    if ((size_t)C_rows != n2 || (size_t)C_cols != n2) {
        throw std::runtime_error("C dimension mismatch");
    }
    
    const MKL_INT C_offset = (C_indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    // Optional but often helpful
    mkl_sparse_optimize(A);

    const size_t block_size = (n2 + n_blocks - 1) / n_blocks;

    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const size_t col_start = (size_t)block_idx * block_size;
        const size_t col_end   = std::min(col_start + block_size, n2);
        const size_t bs = col_end - col_start;
    
        if( col_end - col_start > 0 ) {
            // E_block and D_block are n1 x bs, column-major
            std::vector<double> E_block(n1 * bs, 0.0);
            std::vector<double> D_block(n1 * bs, 0.0);

            // Build E_block(:, local_j) = sum_k C(global_j,k) * B(:,k)
            for (size_t local_j = 0; local_j < bs; ++local_j) {
                const size_t global_j = col_start + local_j;
                double* E_col = E_block.data() + local_j * n1;

                const MKL_INT row_beg = C_row_start[global_j] - C_offset;
                const MKL_INT row_end = C_row_end[global_j]   - C_offset;

                for (MKL_INT idx = row_beg; idx < row_end; ++idx) {
                    const MKL_INT k_mkl = C_col_indx[idx] - C_offset;
                    const size_t  k     = (size_t)k_mkl;

                    const double c_val = (C_values != nullptr) ? C_values[idx] : 1.0;
                    const double* B_col = B.data() + k * n1;   // column-major B(:,k)

                    if (c_val == 1.0) {
                        for (size_t i = 0; i < n1; ++i) {
                            E_col[i] += B_col[i];
                        }
                    } else {
                        for (size_t i = 0; i < n1; ++i) {
                            E_col[i] += c_val * B_col[i];
                        }
                    }
                }
            }

            double* D_block_idx = D.data() + col_start * n1;
            
            // D_block = A * E_block
            // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, SPARSE_LAYOUT_COLUMN_MAJOR, E_block.data(),
            //     (MKL_INT)bs, (MKL_INT)n1, 0.0, D_block.data(), (MKL_INT)n1 );
            st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, scalar, A, descr, SPARSE_LAYOUT_COLUMN_MAJOR, E_block.data(),
                (MKL_INT)bs, (MKL_INT)n1, 0.0, D_block_idx, (MKL_INT)n1 );
            if (st != SPARSE_STATUS_SUCCESS) {
                throw std::runtime_error("mkl_sparse_d_mm(A, E_block) failed");
            }
            // Store D_block into D(:, col_start:col_end)
            // std::copy(D_block.begin(), D_block.end(), D.begin() + col_start * n1);
        }
        
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    // printf("D_block_idx worked\n");
    return D;
}



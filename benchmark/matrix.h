#include <vector>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <cmath>
#include <iostream>

struct CSRMatrixView {
    const MKL_INT* rowptr;
    const MKL_INT* colidx;
    const double*  values;
    MKL_INT nrows = 0, ncols = 0;
    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
};

struct CSRMatrix {
    // Use types compatible with MKL (for generality)
    std::vector<MKL_INT> rowptr;   // size nrows+1
    std::vector<MKL_INT> colidx;   // size nnz
    std::vector<double>   values;  // size nnz
    MKL_INT nrows = 0, ncols = 0;
    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;

    // Optional: a simple constructor
    CSRMatrix(MKL_INT nrows_=0, MKL_INT ncols_=0,
              sparse_index_base_t indexing_ = SPARSE_INDEX_BASE_ZERO)
        : nrows(nrows_), ncols(ncols_), indexing(indexing_) {}

    // Optional: Construct from raw data
    CSRMatrix(MKL_INT nrows_, MKL_INT ncols_,
              const MKL_INT* rowptr_, const MKL_INT* colidx_,
              const double* values_, size_t nnz,
              sparse_index_base_t indexing_ = SPARSE_INDEX_BASE_ZERO)
        : nrows(nrows_), ncols(ncols_), indexing(indexing_),
          rowptr(rowptr_, rowptr_+nrows_+1),
          colidx(colidx_, colidx_+nnz),
          values(values_, values_+nnz)
    {}
};

// D_row: row-major n1 x n2 matrix
// bc:    block column size
// Returns: vector with column blocks stored contiguously (each block row-wise)
std::vector<double> reorder_to_col_blocks(const std::vector<double>& D_row,
                                          int n1, int n2, int bc)
{
    std::vector<double> D_blocked(n1 * n2);

    // Number of full blocks + possibly one partial block
    int num_blocks = (n2 + bc - 1) / bc;

    // Precompute the starting offset of each block in the output
    // Block b has dimensions n1 x bc_b, where bc_b = min(bc, n2 - b*bc)
    // Output offset for block b = sum of (n1 * bc_j) for j < b = n1 * b * bc
    // (except the last block may be smaller, but earlier blocks are all n1*bc)

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < num_blocks; ++b)
    {
        int col_start = b * bc;
        int col_end   = std::min(col_start + bc, n2);
        int bc_actual = col_end - col_start;  // width of this block

        // Output offset: all previous blocks have size n1 * bc,
        // except we need exact offset
        // offset = n1 * col_start  (since sum of previous block widths = col_start)
        size_t out_offset = (size_t)n1 * col_start;

        // Copy row by row
        for (int i = 0; i < n1; ++i)
        {
            // Source: row i, columns [col_start, col_end) in row-major layout
            const double* src = &D_row[(size_t)i * n2 + col_start];

            // Destination: block b, row i within the block
            double* dst = &D_blocked[out_offset + (size_t)i * bc_actual];

            std::memcpy(dst, src, bc_actual * sizeof(double));
        }
    }

    return D_blocked;
}

std::vector<double> restore_from_col_blocks(const std::vector<double>& D_blocked, int n1, int n2, int bc)
{
    std::vector<double> D_row(n1 * n2);
    int num_blocks = (n2 + bc - 1) / bc;
    
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < num_blocks; ++b)
    {
        int col_start = b * bc;
        int col_end   = std::min(col_start + bc, n2);
        int bc_actual = col_end - col_start;

        size_t in_offset = (size_t)n1 * col_start;

        for (int i = 0; i < n1; ++i)
        {
            const double* src = &D_blocked[in_offset + (size_t)i * bc_actual];
            double* dst       = &D_row[(size_t)i * n2 + col_start];

            std::memcpy(dst, src, bc_actual * sizeof(double));
        }
    }

    return D_row;
}

double measure_divergence(const std::vector<double>& runtimes) {
    if (runtimes.empty()) {
        std::cout << "Input vector is empty.\n";
        return 0.0;
    }

    // Mean
    double sum = 0.0;
    double max_time = runtimes[0];
    double min_time = runtimes[0];

    for (double t : runtimes) {
        sum += t;
        if (t > max_time) max_time = t;
        if (t < min_time) min_time = t;
    }
    double mean = sum / runtimes.size();

    // Standard deviation
    double sq_sum = 0.0;
    for (double t : runtimes) {
        sq_sum += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(sq_sum / runtimes.size());

    // Coefficient of variation
    double coeff_of_variation = (mean != 0) ? (stddev / mean) : 0.0;

    // Ratios
    double max_to_mean_ratio = (mean != 0) ? (max_time / mean) : 0.0;
    double max_to_min_ratio = (min_time != 0) ? (max_time / min_time) : 0.0;

    // Print results
    std::cout << "Mean             : " << mean << "\n"
              << "Stddev           : " << stddev << "\n"
              << "Coeff Variation  : " << coeff_of_variation << "\n"
              << "Max/Mean ratio   : " << max_to_mean_ratio << "\n"
              << "Max/Min ratio    : " << max_to_min_ratio << "\n";

    // Return coefficient of variation (or change this to return any other metric)
    return coeff_of_variation;
}

// row_ptr: size n1+1
// col_ind: size nnz (number of nonzeros)
// n1: matrix is n1 x n1
// bc: number of column blocks
std::vector<std::vector<double>> nnz_block_percent( const MKL_INT* row_ptr, const MKL_INT* col_ind, int n1, int bc) {
    std::vector<std::vector<double>> result(n1, std::vector<double>(bc, 0.0));

    int block_size = (n1 + bc - 1) / bc; // ceil division so all columns are covered

    for(int i = 0; i < n1; ++i) {
        MKL_INT row_nnz = row_ptr[i+1] - row_ptr[i];
        // If you want count, use std::vector<int> and skip percentage step
        if(row_nnz == 0) continue;
        // Count nnz per block in this row
        std::vector<int> block_counts(bc, 0);
        for(int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
            MKL_INT col = col_ind[k];
            int block = col / block_size;
            if(block >= bc) block = bc - 1; // safety for last block
            block_counts[block]++;
        }
        // Fill percentage per block (change to count if desired)
        for(int b = 0; b < bc; ++b) {
            result[i][b] = 100.0 * block_counts[b] / row_nnz; // as percentage
            // result[i][b] = block_counts[b]; // uncomment for count instead of percentage
        }
    }
    for(int i=0; i<n1; i++) {
        std::cout <<  "nnz " << row_ptr[i+1] - row_ptr[i] << "  ";
        for(int j=0; j<bc; j++) {
            std::cout << (int) result[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return result;
}
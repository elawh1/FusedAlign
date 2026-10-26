#include "fused_spmm.h"

namespace {

std::vector<double> mkl_sparse_chain_colmajor(sparse_matrix_t A,
                                              sparse_matrix_t C,
                                              const std::vector<double>& B,
                                              size_t n1,
                                              size_t n2,
                                              double scalar) {
    if (B.size() != n1 * n2) throw std::runtime_error("B size mismatch");

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    std::vector<double> temp(n1 * n2, 0.0);
    std::vector<double> out(n1 * n2, 0.0);

    // B is column-major n1 x n2. The same storage is row-major n2 x n1 for B^T.
    // This computes temp_storage = C * B^T, interpreted as (B * C^T) in column-major n1 x n2.
    sparse_status_t st = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                         1.0,
                                         C,
                                         descr,
                                         SPARSE_LAYOUT_ROW_MAJOR,
                                         B.data(),
                                         static_cast<MKL_INT>(n1),
                                         static_cast<MKL_INT>(n1),
                                         0.0,
                                         temp.data(),
                                         static_cast<MKL_INT>(n1));
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm failed for C * B^T");

    st = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                         scalar,
                         A,
                         descr,
                         SPARSE_LAYOUT_COLUMN_MAJOR,
                         temp.data(),
                         static_cast<MKL_INT>(n2),
                         static_cast<MKL_INT>(n1),
                         0.0,
                         out.data(),
                         static_cast<MKL_INT>(n1));
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm failed for A * temp");

    return out;
}

}  // namespace

std::vector<double> blockwise_colsum_matmul(sparse_matrix_t A,
                                            sparse_matrix_t C,
                                            const std::vector<double>& B,
                                            size_t n1,
                                            size_t n2,
                                            int n_blocks,
                                            double scalar) {
    (void)n_blocks;
    return mkl_sparse_chain_colmajor(A, C, B, n1, n2, scalar);
}

std::vector<double> blockwise_colsum_matmul_2(sparse_matrix_t A,
                                              sparse_matrix_t C,
                                              const std::vector<double>& B,
                                              size_t n1,
                                              size_t n2,
                                              int n_blocks) {
    (void)n_blocks;
    return mkl_sparse_chain_colmajor(A, C, B, n1, n2, 1.0);
}

std::vector<double> blockwise_outputcols_matmul(sparse_matrix_t A,
                                                sparse_matrix_t C,
                                                const std::vector<double>& B,
                                                size_t n1,
                                                size_t n2,
                                                int n_blocks,
                                                double scalar) {
    (void)n_blocks;
    return mkl_sparse_chain_colmajor(A, C, B, n1, n2, scalar);
}

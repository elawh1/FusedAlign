#include "helper.h"
#include "helper2.h"

std::vector<double> sparse_to_dense_colmajor(sparse_matrix_t A, MKL_INT nrows, MKL_INT ncols) ;

//Create digonal sparse matrix from a vector of diagonal elements
MKLCSC make_diag_csc(const std::vector<double>& diag, MKL_INT m, MKL_INT n, bool invert);

std::vector<double> l2_norm(std::vector<double> rwr, int n, int d) ;

void l2_norm_inplace(std::vector<double>& rwr, size_t n, size_t d);
void get_normalized_neg_exp_dist (std::vector<double> rwr1, std::vector<double>  rwr2, int n1, int n2, int l, std::vector<double>& cross_rwr_dist) ;

void masked_sddmm_intra_cost (std::vector<double>&  emb, sparse_matrix_t mask, int n, int L, sparse_matrix_t& result) ;
std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<double>& H_vec) ;

sparse_matrix_t row_normalize_graph( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) ;

sparse_matrix_t row_normalize_graph_new( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) ;

void get_batch_rwr_scores(sparse_matrix_t graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, 
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) ;

void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, 
                    std::vector<MKL_INT>>& anchors, double val) ;

// Then set H[row, col] = 1 for each anchor
void init_H_with_anchors(std::vector<double>& H, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors);

// Then set H[row, col] = log(1e-6) for each anchor
std::vector<double> init_Hlog_with_anchors(MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors);

bool out_cross_dist_converge( std::vector<double>& out_cross_dist, std::vector<double>& out_cross_dist_old, double tol) ;

std::vector<double> get_product_rwr_mat_fused( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, 
    int n1, int n2, int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    double rwr_restart_prob, double gamma, int max_iters, double tol) {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto time_0 = std::chrono::high_resolution_clock::now();
    sparse_matrix_t graph1_normalized = row_normalize_graph( graph1_py, n1, avg_nnz1_per_n1);
    sparse_matrix_t graph2_normalized = row_normalize_graph( graph2_py, n2, avg_nnz2_per_n2);
    auto time_1 = std::chrono::high_resolution_clock::now();
    std::vector<double> out_cross_dist(n1 * n2, 0.0);
    std::vector<double> out_cross_dist_old(n1 * n2, 0.0);
    // std::vector<double> temp(n1 * n2);
    // std::vector<double> CC(n1 * n2);
    auto time_2 = std::chrono::high_resolution_clock::now();
    printf("in fused cross dist print sparse matrices\n");
    // print_sparse_matrix_info(graph1_normalized, "graph1_normalized");
    // print_sparse_matrix_info(graph2_normalized, "graph2_normalized");
    for (int i = 0; i < max_iters; ++i) {
        auto time_2_0 = std::chrono::high_resolution_clock::now();
        if (i > 0)  out_cross_dist_old = out_cross_dist;
        auto time_2_1 = std::chrono::high_resolution_clock::now();
        out_cross_dist = blockwise_outputcols_matmul( graph1_normalized, graph2_normalized, out_cross_dist, n1, n2, 1,  (1-rwr_restart_prob));
        auto time_2_2 = std::chrono::high_resolution_clock::now();
        mkl_domatadd('C', 'N', 'N', n1, n2, (1+rwr_restart_prob), in_cross_dist.data(), n1, 1.0, out_cross_dist.data(), n1, out_cross_dist.data(), n1);
        /*
        // CC = in_cross_dist;
        mkl_domatcopy('C', 'T', n1, n2, 1.0, in_cross_dist.data(), n1, CC.data(), n2);

        // 2 spmm correct working version
        auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        mkl_domatcopy('C', 'T', n1, n2, 1.0, temp.data(), n1, out_cross_dist.data(), n2);
        double coeff = (1 - rwr_restart_prob) * gamma;
        auto time_2_0_1 = std::chrono::high_resolution_clock::now();
        st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        auto time_2_0_2 = std::chrono::high_resolution_clock::now();
        mkl_domatcopy('C', 'T', n2, n1, 1.0, CC.data(), n2, out_cross_dist.data(), n1);
        */
        // out_cross_dist = CC;
        // fused spmm 
        // out_cross_dist = blockwise_colsum_matmul( graph1_normalized, graph2_normalized, out_cross_dist, n1, n2, 10);
       
        auto time_2_3 = std::chrono::high_resolution_clock::now();
        //compute max of absolute values of (out_cross_dist - out_cross_dist_old)
        if (out_cross_dist_converge(out_cross_dist, out_cross_dist_old, tol)) break;
        auto time_2_4 = std::chrono::high_resolution_clock::now();
        std::cout << "fused cross dist " << i <<
        " --  copy new to old " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_1-time_2_0).count() << 
        " -- fused spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_2-time_2_1).count() <<
        " -- add to att dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_3-time_2_2).count() <<
        " -- converge compute " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_4-time_2_3).count() << std::endl;
    }
    auto time_3 = std::chrono::high_resolution_clock::now();
    
    //out_cross_dist = (1 - self.gamma) * out_cross_dist
    cblas_dscal(n1 * n2, 1.0 - gamma, out_cross_dist.data(), 1);
    auto time_4 = std::chrono::high_resolution_clock::now();

    std::cout << "in get_product_rwr_mat row normalize " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   << 
    " init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   << 
    " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()   << 
    " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count()   
    << std::endl;
    
    return out_cross_dist;
}

void get_product_rwr_mat( sparse_matrix_t graph1_normalized, sparse_matrix_t graph2_normalized, const std::vector<double>& in_cross_dist, 
    int n1, int n2, double rwr_restart_prob, double gamma, std::vector<double>& out_cross_dist, int max_iters, double tol) {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto time_0 = std::chrono::high_resolution_clock::now();
   
    auto time_1 = std::chrono::high_resolution_clock::now();
    // std::vector<double> out_cross_dist(n1 * n2, 0.0);
    std::vector<double> out_cross_dist_old(n1 * n2, 0.0);
    std::vector<double> temp(n1 * n2);
    std::vector<double> CC(n1 * n2);
    auto time_2 = std::chrono::high_resolution_clock::now();
    // int colsize = 4096*2;
    // prepare_sparse_mm_once(graph1_normalized, descr, colsize, max_iters);
    // prepare_sparse_mm_once(graph2_normalized, descr, colsize, max_iters);


    //Reorder the graphs with metis
    // idx_t num_blocks = 4;
    // PermutedMKLCSR RG1 = reorder_with_metis(graph1_normalized, num_blocks);
    // sparse_matrix_t g1_permuted = RG1.A;
    // PermutedMKLCSR RG2 = reorder_with_metis(graph2_normalized, num_blocks);
    // sparse_matrix_t g2_permuted = RG2.A;

    //Reorder the graphs with mkl
    // int num_blocks = 1;
    // for (int cand : {8, 4, 2}) {
    //     if (n1 % cand == 0) {
    //         num_blocks = cand;
    //         break;
    //     }
    // }
    // sparse_matrix_t g1_BSR = nullptr;
    // sparse_status_t st = mkl_sparse_convert_bsr(graph1_normalized, num_blocks,  SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &g1_BSR);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_convert_bsr failed");
    // num_blocks = 1;
    // for (int cand : {8, 4, 2}) {
    //     if (n2 % cand == 0 ) {
    //         num_blocks = cand;
    //         break;
    //     }
    // }
    // sparse_matrix_t g2_BSR = nullptr;
    // st = mkl_sparse_convert_bsr(graph2_normalized, num_blocks,  SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &g2_BSR);
    // printf("status = %d\n", st);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_convert_bsr failed");
    
    auto time_2_new = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_iters; ++i) {
        auto time_2_0 = std::chrono::high_resolution_clock::now();

        /* anothe way to compute /  SPPM and add at end  -----
        // out_cross_dist_old = out_cross_dist;
        // auto time_2_0_1 = std::chrono::high_resolution_clock::now();
        // //second term computation out_cross_dist is column major too
        // //‌ *** Version one with spmm **** //
        // //temp is n1xn2 temp = G1.cross_dist
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        // double coeff = (1 - rwr_restart_prob) * gamma;
        // auto time_2_0_2 = std::chrono::high_resolution_clock::now();
        // //we need temp2 = temp^T
        // mkl_domatcopy('C', 'T', n1, n2, 1.0, temp.data(), n1, out_cross_dist.data(), n2);
        // auto time_2_0_3 = std::chrono::high_resolution_clock::now();
        // //temp3 is n2xn1  temp3 = G2.(temp^T) = G2.(temp2)
        // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, 0.0, temp.data(), n2);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        // auto time_2_0_4 = std::chrono::high_resolution_clock::now();
        // //adding with scaling first element and transposing second element
        // //out_cross_dist:  n1 x n2
        // mkl_domatadd('C', 'N', 'T', n1, n2, (1+rwr_restart_prob), in_cross_dist.data(), n1, 1.0, temp.data(), n2, out_cross_dist.data(), n1);
        */

        out_cross_dist_old = out_cross_dist;
        mkl_domatcopy('C', 'T', n1, n2, 1.0, in_cross_dist.data(), n1, CC.data(), n2);
        // 2 spmm correct working version
        auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, g1_BSR, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // auto st = blocked_mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1, colsize);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        mkl_domatcopy('C', 'T', n1, n2, 1.0, temp.data(), n1, out_cross_dist.data(), n2);
        double coeff = (1 - rwr_restart_prob) * gamma;
        auto time_2_0_1 = std::chrono::high_resolution_clock::now();
        st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2);
        // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, g2_BSR, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2);
        // st = blocked_mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2, colsize);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        auto time_2_0_2 = std::chrono::high_resolution_clock::now();
        mkl_domatcopy('C', 'T', n2, n1, 1.0, CC.data(), n2, out_cross_dist.data(), n1);
        
        // out_cross_dist = CC;
        // fused spmm 
        // out_cross_dist = blockwise_colsum_matmul( graph1_normalized, graph2_normalized, out_cross_dist, n1, n2, 10);
        //compute max of absolute values of (out_cross_dist - out_cross_dist_old)
        if (out_cross_dist_converge(out_cross_dist, out_cross_dist_old, tol)) break;
        auto time_2_2 = std::chrono::high_resolution_clock::now();
    }
    auto time_3 = std::chrono::high_resolution_clock::now();
    
    //out_cross_dist = (1 - self.gamma) * out_cross_dist
    cblas_dscal(n1 * n2, 1.0 - gamma, out_cross_dist.data(), 1);
    auto time_4 = std::chrono::high_resolution_clock::now();

    std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   << 
    " permute matrices " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_new-time_2).count()   << 
    " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2_new).count()   << 
    " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count()  << std::endl;
}

void get_product_rwr_mat_new( sparse_matrix_t graph1_normalized, sparse_matrix_t graph2_normalized, 
    const std::vector<double>& in_cross_dist, MKL_INT n1, MKL_INT n2, double rwr_restart_prob, double gamma, 
    std::vector<double>& out_cross_dist, int max_iters, double tol) {
    size_t N = (size_t)n1 * (size_t)n2;
    if (out_cross_dist.size() != N || in_cross_dist.size() != N)    throw std::runtime_error("in cross dist computation size mismatch");
    // const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
    // size_t off = 0;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto time_0 = std::chrono::high_resolution_clock::now();
    std::vector<double> out_cross_dist_old(N, 0.0);
    std::vector<double> temp(N);
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto time_spmm1 = 0, time_spmm2 = 0, time_add = 0, time_convergence = 0;
    for (int i = 0; i < max_iters; ++i) {
        // We want graph1_normalized * out_cross_dist * graph2_normalized^T. In firts spmm instead of out_cross_dist * graph2_normalized^T 
        // we compute ( graph2_normalized * out_cross_dist^T )^T and we read and store dense ones row major - but since all dense matrices are col major it is interpreted as transpose
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph2_normalized, descr, SPARSE_LAYOUT_ROW_MAJOR, out_cross_dist_old.data(), n1, n2, 0.0, temp.data(), n1);
        auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph2_normalized, descr, SPARSE_LAYOUT_ROW_MAJOR, out_cross_dist_old.data(), n1, n1, 0.0, temp.data(), n1);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        time_spmm1 += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        double coeff = (1.0 - rwr_restart_prob) * gamma;
        st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, temp.data(), n2, n1, 0.0, out_cross_dist.data(), n1);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        time_spmm2 += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        cblas_daxpy(N, (1.0 + rwr_restart_prob), in_cross_dist.data(), 1, out_cross_dist.data(), 1);
        // off = 0;
        // while (off < N) {
        //     size_t len = std::min(CHUNK, N - off);
        //     cblas_daxpy((MKL_INT)len, (1.0 + rwr_restart_prob), in_cross_dist.data() + off, 1,  out_cross_dist.data() + off, 1);
        //     off += len;
        // }
        auto time_1_3 = std::chrono::high_resolution_clock::now();
        time_add += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2).count();
        if (out_cross_dist_converge(out_cross_dist, out_cross_dist_old, tol)) break;
        auto time_1_4 = std::chrono::high_resolution_clock::now();
        time_convergence += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_4 - time_1_3).count();
        std::swap(out_cross_dist, out_cross_dist_old);
    }
    auto time_2 = std::chrono::high_resolution_clock::now();
    std::swap(out_cross_dist, out_cross_dist_old);
    //out_cross_dist = (1 - self.gamma) * out_cross_dist
    // cblas_dscal(N, 1.0 - gamma, out_cross_dist.data(), 1);
    // off = 0;
    // while (off < N) {
    //     size_t len = std::min(CHUNK, N - off);
    //     cblas_dscal((MKL_INT)len, 1.0 - gamma, out_cross_dist.data() + off, 1);
    //     off += len;
    // }
    dscal_chunked(N, 1.0 - gamma, out_cross_dist.data());
    auto time_3 = std::chrono::high_resolution_clock::now();
    std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   << 
    " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   << 
    " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()  << std::endl;
    std::cout << "in for loop 1st spmm " << time_spmm1  << " 2nd spmm " << time_spmm2  << " add " << time_add  << " convergence " << time_convergence  << std::endl;
}

void minaa_v3_dense( std::vector<double>& Cost, const std::vector<double>& b, int n_row, int n_col, double epsilon, std::vector<double>& out_vec ) ;
void row_avg_square ( sparse_matrix_t& intra_dist, int n, std::vector<double>& row_avg_sqr) ;

//     const size_t N = (size_t)n1 * (size_t)n2;
//     if (out_cross_dist.size() != N || in_cross_dist.size() != N)    throw std::runtime_error("size mismatch");

//     std::vector<double> out_cross_dist_old(N, 0.0);
//     std::vector<double> temp(N);

//     auto daxpy_chunked = [](size_t N, double alpha, const double* x, double* y) {
//         const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
//         size_t off = 0;
//         while (off < N) {
//             size_t len = std::min(CHUNK, N - off);
//             cblas_daxpy((MKL_INT)len, alpha, x + off, 1, y + off, 1);
//             off += len;
//         }
//     };

//     auto dscal_chunked = [](size_t N, double alpha, double* x) {
//         const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
//         size_t off = 0;
//         while (off < N) {
//             size_t len = std::min(CHUNK, N - off);
//             cblas_dscal((MKL_INT)len, alpha, x + off, 1);
//             off += len;
//         }
//     };

//     const double coeff = (1.0 - rwr_restart_prob) * gamma;

//     for (int it = 0; it < max_iters; ++it) {
//         auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph2_normalized, descr, SPARSE_LAYOUT_ROW_MAJOR, out_cross_dist_old.data(),
//             n1,  n1, 0.0, temp.data(), n1); 
//         if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("first mkl_sparse_d_mm failed");
//         st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
//             temp.data(), n2,  n1,  0.0, out_cross_dist.data(), n1);  
//         if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("second mkl_sparse_d_mm failed");
//         daxpy_chunked(N, 1.0 + rwr_restart_prob, in_cross_dist.data(), out_cross_dist.data());
//         if (out_cross_dist_converge(out_cross_dist, out_cross_dist_old, tol))    break;
//         std::swap(out_cross_dist, out_cross_dist_old);
//     }
//     // if loop broke before swap, current result may already be in out_cross_dist
//     // if loop ended normally after swap pattern, final swap restores expected output
//     if (out_cross_dist.data() == out_cross_dist_old.data()) {
//         // impossible for std::vector, kept only for clarity
//     }

//     dscal_chunked(N, 1.0 - gamma, out_cross_dist.data());
// }

void minaa_v3_dense( std::vector<double>& Cost, const std::vector<double>& b, int n_row, int n_col, double epsilon, std::vector<double>& out_vec ) {
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (int j = 0; j < n_col; ++j) {
        double* col = &Cost[(size_t)j * n_row];
        double min_j = (col[0] -= b[0]);
        for (int i = 1; i < n_row; ++i) {
            col[i] -= b[i];
            min_j = std::min(min_j, col[i]);
        }
        double s_j = 0.0;
        for (int i = 0; i < n_row; i++) {
            double shifted = col[i] - min_j;
            col[i] += b[i];
            s_j += std::exp(-(shifted / epsilon));
        }
        out_vec[j] = min_j - epsilon * std::log(s_j / n_row);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}

void euclidean_distance_mat_vec( const std::vector<double>& X, const std::vector<double>& y, int n, int d, std::vector<double>& out_vec ) {
    #pragma omp parallel for 
    for (int i = 0; i < n; ++i) {
        out_vec[i]  = 0.0;
        const double* row = &X[(size_t)i * d];
        for (int j = 0; j < d; ++j) {
            double diff = row[j] - y[j];
            out_vec[i] += diff * diff;
        }
    }
}

void row_avg_square_dense(const std::vector<double>& matrix, size_t n, size_t d, std::vector<double>& row_sum_sqr) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        double square_sum = 0.0;
        const double* row = &matrix[i * d]; 
        for (size_t j = 0; j < d; ++j) {
            double val = row[j];
            square_sum += (val * val) / d;
        }
        row_sum_sqr[i] = square_sum;
    }
}

void row_avg_square ( sparse_matrix_t& intra_dist, int n, std::vector<double>& row_avg_sqr) {
    // Export CSR data
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start = nullptr, *row_end = nullptr;
    MKL_INT *col_indx = nullptr;
    double *values = nullptr;
    sparse_status_t st = mkl_sparse_d_export_csr( intra_dist, &indexing, &rows, &cols, &row_start, &row_end, &col_indx, &values);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("mkl_sparse_d_export_csr failed");

    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows; ++i) {
        double square_sum = 0.0;
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            square_sum += values[idx] * values[idx];
        }
        row_avg_sqr[i] = (double) square_sum / n;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}

void compute_L_fixed( sparse_matrix_t& intra_dist1, sparse_matrix_t& intra_dist2, int n1, int n2, std::vector<double>& L_fixed) {
    // 1) Square intra_dist1 and intra_dist2 elementwise
    std::vector<double> row_avg_square_dist1(n1), row_avg_square_dist2(n2);
    row_avg_square(intra_dist1, n1, row_avg_square_dist1);
    row_avg_square(intra_dist2, n2, row_avg_square_dist2);
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for schedule(static)
    for( int j=0; j < n2; ++j) {
        double dist2_term =  row_avg_square_dist2[j];
        for(int i=0; i < n1; ++i) {
            size_t idx = (size_t) j * (size_t) n1 + (size_t) i;
            L_fixed[idx] = dist2_term + row_avg_square_dist1[i];
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}


void compute_Cost_fixed( sparse_matrix_t& intra_dist1, sparse_matrix_t& intra_dist2, std::vector<double>& cross_dist, const std::vector<double>& H_log, 
        int n1, int n2, double lambda_e, double lambda_a ) ;

void compute_RW_term ( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, std::vector<double>& RW_term) ;
void compute_A2T_term( sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& S, int n1, int n2, std::vector<double>& A2T_term) ;

void compute_C_L( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, 
        sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& L_fixed,
         const std::vector<double>& cross_dist, const std::vector<double>& H_log, double lambda_e, 
         double lambda_a, double lambda_n, std::vector<double>& C);

void compute_C_L_fused( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, 
        sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& L_fixed,
         const std::vector<double>& cross_dist, const std::vector<double>& H_log, double lambda_e, 
         double lambda_a, double lambda_n, std::vector<double>& C) ;


void compute_cost_fused( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2,  sparse_matrix_t intra_dist1, 
        sparse_matrix_t intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C) ;


void compute_cost_fused_2( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2,  sparse_matrix_t intra_dist1, 
        sparse_matrix_t intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C) ;

double update_S_mat (std::vector<double>& S, const std::vector<double>& S_old, const std::vector<double>& Cost, const std::vector<double>& a_row, const std::vector<double>& b_col, int n1, int n2, double epsilon) ;

std::vector<double> con_prox_pt_opt( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2, 
    std::vector<double>& cross_dist, sparse_matrix_t intra_dist1_sparse, sparse_matrix_t intra_dist2_sparse, int inner_iters, int outer_iters, double lambda_n, double lambda_a, 
    double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs, bool sparsify);
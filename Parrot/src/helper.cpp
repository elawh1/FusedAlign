#include "helper.h"
#include "helper2.h"

// ============================================================
// BLOCKED: jik loop order   (Algorithm 2 v1 — panel blocked)
//  Outer loop:  j = 0, b_c, 2*b_c, ...  (column panels of B/C)
//  Middle loop: i = 0 .. n1-1           (rows of A, parallelised)
//  Inner loop:  nonzeros in row i of A
//  Each stage only touches a NARROW strip of B and C (b_c cols),
//  so those strips fit in L2/L3 cache → far fewer cache misses.
// difference with omp_spmm_jik : here we read each column partition B inside the row for loop - this version is faster
// ============================================================
void omp_spmm_jik_v1(double alpha, sparse_matrix_t graph_csr, MKL_INT n1, int n2, double beta, const double* B, double* C, int b_c){
    sparse_index_base_t indexing;
    MKL_INT  rows, cols;
    MKL_INT *rows_start = nullptr;
    MKL_INT *rows_end   = nullptr;
    MKL_INT *col_ind    = nullptr;
    double  *vals       = nullptr;

    auto st = mkl_sparse_d_export_csr( graph_csr, &indexing, &rows, &cols, &rows_start, &rows_end, &col_ind, &vals);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("export csr failed");

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    b_c = std::min(b_c, n2);
    b_c = std::max(b_c, 1);
    // Q : number of blocks in dense matrices
    const int Q = (n2 + b_c - 1) / b_c;

    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);

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
        }
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
}

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

    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);

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
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
}

std::vector<double> sparse_to_dense_colmajor(sparse_matrix_t A, MKL_INT nrows, MKL_INT ncols) {
    // Output: column-major dense matrix
    std::vector<double> dense(nrows * ncols, 0.0);

    // Export CSR data
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start = nullptr, *row_end = nullptr;
    MKL_INT *col_indx = nullptr;
    double *values = nullptr;

    sparse_status_t st = mkl_sparse_d_export_csr( A, &indexing, &rows, &cols, &row_start, &row_end, &col_indx, &values);
    if (st != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("mkl_sparse_d_export_csr failed");
    if (rows != nrows || cols != ncols)
        throw std::runtime_error("dimension mismatch in sparse_to_dense");

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;

    // Fill dense (column-major)
    for (MKL_INT i = 0; i < nrows; ++i) {
        for (MKL_INT k = row_start[i] - base; k < row_end[i] - base; ++k) {
            MKL_INT j = col_indx[k] - base;
            dense[i + j * nrows] = values[k];   // column-major indexing
        }
    }

    return dense;
}

//Create digonal sparse matrix from a vector of diagonal elements
MKLCSC make_diag_csc(const std::vector<double>& diag, MKL_INT m, MKL_INT n, bool invert) {
    MKL_INT nnz = diag.size();
    MKLCSC A;    
    A.col_start.resize(n);             
    A.col_end.resize(n);             
    A.row_ind.resize(nnz);             
    A.vals.resize(nnz);  

    for(MKL_INT k=0; k < nnz; k++) {
        A.col_start[k] = k;  
        A.col_end[k] = k+1;  
        A.row_ind[k] = k;
        if (invert) {
            if (diag[k] == 0.0) throw std::runtime_error("zero diagonal cannot be inverted");
            A.vals[k] = 1.0 / diag[k];
        } else {
            A.vals[k] = diag[k];
        }
    }
    
    auto st = mkl_sparse_d_create_csc(&A.spMat, SPARSE_INDEX_BASE_ZERO, m, n, A.col_start.data(), A.col_end.data(), A.row_ind.data(), A.vals.data());
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_create_csc failed");
    mkl_sparse_optimize(A.spMat);
	return A;
}

std::vector<double> l2_norm(std::vector<double> rwr, int n, int d) {
    std::vector<double> norms(n);
    // Row-wise L2 norm: row i = { rwr[i + j*n] for j=0..d-1 }
    for (int i = 0; i < n; i++)
        norms[i] = cblas_dnrm2(d, &rwr[i], n);   // N=d, incx = n
    // Invert norms
    for (int i = 0; i < n; i++)
        if (norms[i] > 0.0) norms[i] = 1.0 / norms[i];
    // Scale each row by 1/norm[i]
    for (int i = 0; i < n; i++)
        cblas_dscal(d, norms[i], &rwr[i], n);   // N=d, incx = n
    return rwr;
}

void l2_norm_inplace(std::vector<double>& rwr, size_t n, size_t d) {
    std::vector<double> norms(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double norm = cblas_dnrm2((MKL_INT)d, &rwr[i], (MKL_INT)n);
        norms[i] = (norm > 0.0) ? 1.0 / norm : 0.0;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        if (norms[i] != 0.0)
            cblas_dscal((MKL_INT)d, norms[i], &rwr[i], (MKL_INT)n);
    }
}

void get_normalized_neg_exp_dist (std::vector<double> rwr1, std::vector<double>  rwr2, int n1, int n2, int l, std::vector<double>& cross_rwr_dist) {
    // l2_norm_inplace(rwr1, n1, l);
    // l2_norm_inplace(rwr2, n2, l);
    // cross_rwr_dist = exp( - rwr1_norm * rwr2_norm^T )
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, (MKL_INT)n1, (MKL_INT)n2, (MKL_INT)l, -1.0, rwr1.data(), (MKL_INT)n1, 
            rwr2.data(), (MKL_INT)n2, 0.0, cross_rwr_dist.data(), (MKL_INT)n1);
    size_t total = (size_t)n1 * (size_t)n2;
    vdExp((MKL_INT)total, cross_rwr_dist.data(), cross_rwr_dist.data());
    // if (total > (size_t)std::numeric_limits<MKL_INT>::max()) {
    //     size_t offset = 0;
    //     const size_t chunk = (size_t)std::numeric_limits<MKL_INT>::max();
    //     while (offset < total) {
    //         size_t len = std::min(chunk, total - offset);
    //         vdExp((MKL_INT)len, cross_rwr_dist.data() + offset, cross_rwr_dist.data() + offset);
    //         offset += len;
    //     }
    // } else {
    //     vdExp((MKL_INT)total, cross_rwr_dist.data(), cross_rwr_dist.data());
    // }
}

void masked_sddmm_intra_cost (std::vector<double>&  emb, sparse_matrix_t mask, int n, int L, sparse_matrix_t& result) {
    // l2_norm_inplace(emb, n, L);

    // sparse_matrix_t result;
    matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto st = mkl_sparse_copy(mask, descr, &result);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("Copy failed");
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start, *row_end, *col_ind;
    double *result_vals;
    st = mkl_sparse_d_export_csr( result, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, &result_vals);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("Export CSR failed");

    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    // #pragma omp parallel for schedule(static)
    // for (MKL_INT i = 0; i < rows; ++i) {
    //     for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
    //         MKL_INT j = col_ind[idx];
    //         double product = 0.0;
    //         for (int l = 0; l < L; ++l) {
    //             const double* col = emb.data() + (size_t)l * rows;
    //             product += col[i] * col[j];
    //         }
    //         product = std::exp(-1.0 * product);
    //         result_vals[idx] = product ;     // inner product of two emb
    //     }
    // }
    if (rows != n || cols != n)
        throw std::runtime_error("mask dimensions do not match n");

    if (emb.size() != (size_t)n * (size_t)L)
        throw std::runtime_error("emb size mismatch");

    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            MKL_INT j = col_ind[idx];

            double product = 0.0;
            for (MKL_INT l = 0; l < L; ++l) {
                const double* col = emb.data() + (size_t)l * (size_t)rows;
                product += col[i] * col[j];
            }

            result_vals[idx] = std::exp(-product);
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}

std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<double>& H_vec) {
    std::vector<MKL_INT> anchor_g1;
    std::vector<MKL_INT> anchor_g2;
    for(size_t i=0; i< H_vec.size()/2; i++) {
        anchor_g1.push_back((MKL_INT) H_vec[i*2]);
        anchor_g2.push_back((MKL_INT) H_vec[i*2+1]);
    }
    return {anchor_g1, anchor_g2};
}

sparse_matrix_t row_normalize_graph( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) {

    std::vector<double> all_ones(n, (double) 1.0);
    std::vector<double> graph_row_sum(n, (double) 0.0);
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in row_normalize_graph");
    //connect isolated nodes and row normalize A  
    std::vector<MKL_INT> isolated;
    bool isolated_node = false ;
    for (MKL_INT i = 0; i < n; ++i) {
        if (graph_row_sum[i] < 1e-10) {
            isolated.push_back(i);
            isolated_node = true;
        }   
    }
    DEBUG_ONLY(printf("ang nnz %d num of isolated %ld\n", avg_nnz_per_n, isolated.size()));

    sparse_matrix_t graph_connected = nullptr;
    if(isolated_node) {
        DEBUG_ONLY(printf("connecting isolated nodes\n"));

        std::vector<double> edge_list;
        // connect to #avg_nnz_per_n number of random nodes and make it symmetric
        edge_list.reserve(isolated.size() * avg_nnz_per_n * 4);
        std::srand(std::time(0));
        for (MKL_INT i : isolated) {
            std::set<MKL_INT> random_nodes = {};
            while(random_nodes.size() < avg_nnz_per_n) {
                MKL_INT random_num = std::rand() % n;
                if ( random_nodes.find(random_num) !=  random_nodes.end()) continue;
                if ( random_num == i) continue;
                if ( random_num >= n )          std::cerr << "random num generated out of bound\n" << std::endl; 
                random_nodes.insert(random_num);
                edge_list.push_back(static_cast<double>(i));
                edge_list.push_back(static_cast<double>(random_num));
                edge_list.push_back(static_cast<double>(random_num));
                edge_list.push_back(static_cast<double>(i));
            }
            // graph_row_sum[i] = static_cast<double>(avg_nnz_per_n);
        }
        
        DEBUG_ONLY(printf("edge_list size %d \n", edge_list.size()));
        // sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, n, n);
        sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, n, n);
        //add add_mat to original graph
        status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, graph, 1.0, add_mat, &graph_connected);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_add failed in get_batch_rwr_scores");

        mkl_sparse_destroy(add_mat);

        // //update graph_row_sum
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_connected, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    }
    else {
        graph_connected = graph;
    }
    
    auto time_1 = std::chrono::high_resolution_clock::now();

    for(int i=0; i<graph_row_sum.size(); i++)
        if (graph_row_sum[i] < 1e-10) DEBUG_ONLY(printf("111 graph_row_sum[%d] is zero \n", i));
    
    MKLCSC Dinv = make_diag_csc(graph_row_sum, n, n, true);
    sparse_matrix_t graph_norm = nullptr;    
    mkl_sparse_optimize(Dinv.spMat);
    mkl_sparse_optimize(graph_connected);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph_connected, &graph_norm);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_spmm failed in get_batch_rwr_scores");


    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_norm, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    for(int i=0; i<graph_row_sum.size(); i++)
        if (graph_row_sum[i] < 1e-10) DEBUG_ONLY(printf("222 graph_row_sum[%d] is zero \n", i));

    mkl_sparse_optimize(graph_norm);
    sparse_matrix_t graph_norm_csr = nullptr;
    status = mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_convert_csr failed in get_batch_rwr_scores");
    status = mkl_sparse_optimize(graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_optimize failed in get_batch_rwr_scores");  
    
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_norm_csr, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    for(int i=0; i<graph_row_sum.size(); i++)
        if (graph_row_sum[i] < 1e-10) DEBUG_ONLY(printf("333 graph_row_sum[%d] is zero \n", i));


    return graph_norm_csr;
}

sparse_matrix_t row_normalize_graph_new( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) {
    
    std::vector<double> all_ones(n, (double) 1.0);
    std::vector<double> graph_row_sum(n, (double) 0.0);
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in row_normalize_graph");
    //connect isolated nodes and row normalize A  
    std::vector<MKL_INT> isolated;
    bool isolated_node = false ;
    for (MKL_INT i = 0; i < n; ++i) {
        if (graph_row_sum[i] == 0.0) {
            isolated.push_back(i);
            isolated_node = true;
        }   
    }
    DEBUG_ONLY(printf("num of isolated %ld\n", isolated.size()));

    sparse_matrix_t graph_connected = nullptr;
    if(isolated_node) {
        DEBUG_ONLY(printf("connecting isolated nodes\n"));

        std::vector<double> edge_list;

        //Connect to all nodes
        edge_list.reserve(isolated.size() * n * 2);
        for (MKL_INT i : isolated) {
            for (MKL_INT j = 0; j < n; ++j) {
                edge_list.push_back(static_cast<double>(i));
                edge_list.push_back(static_cast<double>(j));
            }
            // update row sum directly
            graph_row_sum[i] = static_cast<double>(n);
        }

        // // connect to #avg_nnz_per_n number of random nodes
        // edge_list.reserve(isolated.size() * avg_nnz_per_n * 2);
        // std::srand(std::time(0));
        // for (MKL_INT i : isolated) {
        //     // printf("isolated node %d\n", i);
        //     std::set<MKL_INT> random_nodes = {};
        //     while(random_nodes.size() < avg_nnz_per_n) {
        //         MKL_INT random_num = std::rand() % n;
        //         if ( random_nodes.find(random_num) !=  random_nodes.end()) continue;
        //         if ( random_num >= n )          std::cerr << "random num generated out of bound\n" << std::endl; 
        //         random_nodes.insert(random_num);
        //         edge_list.push_back(static_cast<double>(i));
        //         edge_list.push_back(static_cast<double>(random_num));
        //     }
        //     // update row sum directly
        //     graph_row_sum[i] = static_cast<double>(avg_nnz_per_n);
        //     // printf("isolated node %d connected to %d nodes\n", i, edge_list.size());
        // }

        // build CSR matrix (values are implicitly 1.0 inside)
        // sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, n, n);
        sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, n, n);
        //add add_mat to original graph
        status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE,graph, 1.0, add_mat, &graph_connected);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_add failed in get_batch_rwr_scores");
        // mkl_sparse_optimize(graph_connected);
        mkl_sparse_destroy(add_mat);

        // //update graph_row_sum
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_connected, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    }
    else {
        graph_connected = graph;
    }
    // std::vector<double> graph_row_sum = graph_connect_isolated(graph);
    // sparse_matrix_t graph_connected;
    // std::vector<double> graph_row_sum(n, (double) 0.0);
    // graph_connect_isolated(graph, graph_row_sum, graph_connected);
    
    MKLCSC Dinv = make_diag_csc(graph_row_sum, n, n, true);
    sparse_matrix_t graph_norm = nullptr;
    mkl_sparse_optimize(Dinv.spMat);
    mkl_sparse_optimize(graph_connected);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph_connected, &graph_norm);
    // mkl_sparse_optimize(graph);
    // auto status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph, &graph_norm);
    if (status != SPARSE_STATUS_SUCCESS) { std::cerr << "error : mkl_sparse_spmm failed" << std::endl; }
    sparse_matrix_t graph_norm_csr;
    mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    
    return graph_norm_csr;
}

void get_batch_rwr_scores(sparse_matrix_t graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, 
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto time_0 = std::chrono::high_resolution_clock::now();
    int l = anchors.size();
    //one hot encoding E matrix
    std::vector<double> E(n * l, 0.0);
    for (int j = 0; j < l; ++j)         E[ anchors[j] + j * n ] = restart_prob;   // one-hot at (row=anchor[j], col=j)
    
    //these are column major
    std::vector<double> X_vals = E;
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto add_after_spmm_time = 0, spmm_time=0, swap_after_spmm_time=0;
    for(int i=0; i < max_iters-1; i++) {
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        // auto status = mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, (double) (1-restart_prob), graph_norm_csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X_vals.data(), l, n, 1.0, Xnext_vals.data(), n);
        // if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("error spmm");
        omp_spmm_col_omp(1.0, graph_norm_csr, n, l, 0.0, X_vals.data(), Xnext_vals.data());
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        
        // cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, l, n,(double) (1-restart_prob), graph_norm_csr_dense.data(), n, X_vals.data(), n, restart_prob, Xnext_vals.data(), n);
        cblas_daxpy(n * l, 1.0, E.data(), 1, Xnext_vals.data(), 1);
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        add_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        std::swap(X_vals, Xnext_vals);
        auto time_1_3 = std::chrono::high_resolution_clock::now();
        swap_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2).count();
    }
    std::swap(X_vals, Xnext_vals);
    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_batch_rwr_scores generate E " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " spmm time " << spmm_time << " add after spmm time " << add_after_spmm_time << " swap after spmm time " << swap_after_spmm_time << std::endl);
}

void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, 
                    std::vector<MKL_INT>>& anchors, double val) {
    const MKL_INT k = (MKL_INT)anchors.first.size();

    #pragma omp parallel for
    for (MKL_INT t = 0; t < k; ++t) {
        MKL_INT i = anchors.first[t];
        MKL_INT j = anchors.second[t];
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            printf("error anchor_cost_zero: out-of-range anchor (%d,%d)\n", i, j);
            continue;
        }
        size_t idx = (size_t)j * (size_t)nrows + (size_t)i;
        A[idx] = val;
    }
}

// Then set H[row, col] = 1 for each anchor
void init_H_with_anchors(std::vector<double>& H, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors) {
    const MKL_INT k = (MKL_INT) anchors.first.size();
    if (anchors.second.size() != (size_t)k) {
        printf("anchor size mismatch: %zu vs %zu\n",
               anchors.first.size(), anchors.second.size());
        return;
    }
    for (MKL_INT t = 0; t < k; ++t) {
        MKL_INT i = anchors.first[t];   // row index
        MKL_INT j = anchors.second[t];  // col index
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            printf("error init_H_with_anchors: out-of-range anchor (%d,%d)\n", i, j);
            continue;
        }
        MKL_INT idx = i + j * nrows;    // column-major: (row, col) -> index
        H[idx] = 1.0;
    }
}

// Then set H[row, col] = log(1e-6) for each anchor
std::vector<double> init_Hlog_with_anchors(MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors) {
    size_t N = nrows * ncols;
    std::vector<double> H(N, 0.0);
    const MKL_INT k = (MKL_INT) anchors.first.size();
    if (anchors.second.size() != (size_t)k) {
        throw std::invalid_argument("anchor size mismatch  inside init_Hlog_with_anchors\n");
    }
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (MKL_INT t = 0; t < k; ++t) {
        MKL_INT i = anchors.first[t];   // row index
        MKL_INT j = anchors.second[t];  // col index
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            printf("error init_H_with_anchors: out-of-range anchor (%d,%d)\n", i, j);
            continue;
        }
        size_t idx = (size_t) i + (size_t) j * (size_t) nrows;    // column-major: (row, col) -> index
        H[idx] = std::log(1e-6);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return H;
}

// std::vector<double> fix_isolated_nodes_rowmajor(const std::vector<double>& A, size_t nrows, size_t ncols)
// {
//     std::vector<double> e(ncols, 1.0);
//     std::vector<double> s(nrows, 0.0);

//     // Row-major GEMV: y = alpha*A*x + beta*y
//     cblas_dgemv(CblasRowMajor, CblasNoTrans, nrows, ncols, 1.0, A, ncols,e.data(), 1, 0.0, s.data(), 1);

//     // 2) Build mask m: m[i]=1 if row i is "isolated" (sum ~ 0), else 0
//     std::vector<double> m(nrows);
//     #pragma omp simd
//     for (MKL_INT i = 0; i < nrows; ++i) {
//         m[i] = (std::abs(s[i]) <= eps) ? 1.0 : 0.0;
//     }

//     // Quick exit if no isolated rows
//     // (Optional) if (cblas_idamax(nrows, m.data(), 1) == 0 && m[0] == 0.0) return;

//     // 3) A += m * e^T   (rank-1 update). For isolated rows (m[i]=1), this adds ones across the row.
//     // Non-isolated rows (m[i]=0) are unchanged.
//     cblas_dger(CblasRowMajor,
//                nrows, ncols,
//                1.0,                // alpha
//                m.data(), 1,        // x = m (nrows)
//                e.data(), 1,        // y = e (ncols)
//                A, ncols);          // A (row-major leading dim = ncols)
// }


bool out_cross_dist_converge( std::vector<double>& out_cross_dist, std::vector<double>& out_cross_dist_old, double tol) {
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    bool converged = true;
    #pragma omp parallel for reduction(&:converged)
    for (size_t i = 0; i < out_cross_dist.size(); ++i) {
        if (std::abs(out_cross_dist[i] - out_cross_dist_old[i]) >= tol)     converged = false;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return converged;
}

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
    DEBUG_ONLY(printf("in fused cross dist print sparse matrices\n"));
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
        DEBUG_ONLY(std::cout << "fused cross dist " << i <<
            " --  copy new to old " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_1-time_2_0).count() <<
            " -- fused spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_2-time_2_1).count() <<
            " -- add to att dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_3-time_2_2).count() <<
            " -- converge compute " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_4-time_2_3).count() << std::endl);
    }
    auto time_3 = std::chrono::high_resolution_clock::now();
    
    //out_cross_dist = (1 - self.gamma) * out_cross_dist
    cblas_dscal(n1 * n2, 1.0 - gamma, out_cross_dist.data(), 1);
    auto time_4 = std::chrono::high_resolution_clock::now();

    DEBUG_ONLY(std::cout << "in get_product_rwr_mat row normalize " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() <<
        " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() << std::endl);
    
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
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, g1_BSR, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // auto st = blocked_mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1, colsize);
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        omp_spmm_col_omp(1.0, graph1_normalized, n1, n2, 0.0, out_cross_dist.data(), temp.data());
        mkl_domatcopy('C', 'T', n1, n2, 1.0, temp.data(), n1, out_cross_dist.data(), n2);
        double coeff = (1 - rwr_restart_prob) * gamma;
        auto time_2_0_1 = std::chrono::high_resolution_clock::now();
        // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, g2_BSR, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2);
        // st = blocked_mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2, colsize);
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        omp_spmm_col_omp(1.0, graph1_normalized, n1, n2, 0.0, out_cross_dist.data(), temp.data());
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

    DEBUG_ONLY(std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " permute matrices " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_new-time_2).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2_new).count() <<
        " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() << std::endl);
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
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph2_normalized, descr, SPARSE_LAYOUT_ROW_MAJOR, out_cross_dist_old.data(), n1, n1, 0.0, temp.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        omp_spmm_jik_v1(1.0, graph2_normalized, n2, n1, 0.0, out_cross_dist_old.data(), temp.data(), 64);
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        time_spmm1 += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        double coeff = (1.0 - rwr_restart_prob) * gamma;
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, temp.data(), n2, n1, 0.0, out_cross_dist.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        omp_spmm_col_omp(1.0, graph1_normalized, n1, n2, 0.0, temp.data(), out_cross_dist.data());
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
    DEBUG_ONLY(std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << std::endl);
    DEBUG_ONLY(std::cout << "in for loop 1st spmm " << time_spmm1 << " 2nd spmm " << time_spmm2 << " add " << time_add << " convergence " << time_convergence << std::endl);
}

// void get_product_rwr_mat_new2( sparse_matrix_t graph1_normalized, sparse_matrix_t graph2_normalized, const std::vector<double>& in_cross_dist,
//     MKL_INT n1, MKL_INT n2, double rwr_restart_prob, double gamma, std::vector<double>& out_cross_dist, int max_iters = 50, double tol = 1e-2){
//     matrix_descr descr{};
//     descr.type = SPARSE_MATRIX_TYPE_GENERAL;

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
        int n1, int n2, double lambda_e, double lambda_a ) {
    auto time_0 = std::chrono::high_resolution_clock::now();
    // 1) Square intra_dist1 and intra_dist2 elementwise
    std::vector<double> row_avg_square_dist1(n1), row_avg_square_dist2(n2);
    row_avg_square(intra_dist1, n1, row_avg_square_dist1);
    row_avg_square(intra_dist2, n2, row_avg_square_dist2);
    auto time_1 = std::chrono::high_resolution_clock::now();
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        double dist2_term =  row_avg_square_dist2[j];
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i ;
            cross_dist[idx] = cross_dist[idx] + lambda_e * (dist2_term + row_avg_square_dist1[i]) - lambda_a * H_log[idx]; 
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in compute_Cost_fixed before loop  : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " after loop  : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << std::endl);
}

void compute_RW_term ( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, std::vector<double>& RW_term) {
    std::vector<double> buffer(n1*n2);
    struct matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, buffer.data(), n1);   //just for test - not correct in algo
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm (A1^T S) failed");
    mkl_domatcopy('C', 'T', n1, n2, 1.0, buffer.data(), n1, RW_term.data(), n2); //transpose
    mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,  1.0, row_norm_adj2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, RW_term.data(), n1, n2, 0.0, buffer.data(), n2);      //spmm
    mkl_domatcopy( 'C', 'T', n2, n1, 1.0, buffer.data(), n2, RW_term.data(), n1);
}

void compute_A2T_term( sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& S, int n1, int n2, std::vector<double>& A2T_term) {
    std::vector<double> buffer(n1*n2);
    struct matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    // // buffer = intra_dist1 @ S  
    auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, buffer.data(), n1);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
    // A2T_term = buffer ^ T
    mkl_domatcopy('C', 'T', n1, n2, 1.0, buffer.data(), n1, A2T_term.data(), n2);
    // buffer = intra_dist2 @ A2T_term  
    st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, A2T_term.data(), n1, n2, 0.0, buffer.data(), n2);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
    // A2T_term =  buffer ^ T
    mkl_domatcopy('C', 'T', n2, n1, (double) -2, buffer.data(), n2, A2T_term.data(), n1);
}

void compute_C_L( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, 
        sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& L_fixed,
         const std::vector<double>& cross_dist, const std::vector<double>& H_log, double lambda_e, 
         double lambda_a, double lambda_n, std::vector<double>& C) {
    std::vector<double> buffer(n1*n2);
    std::vector<double> buffer2(n1*n2);
    // std::vector<double> buffer3(n1*n2);   //is C 
    struct matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    // // buffer3 = intra_dist1 @ S  
    auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, C.data(), n1);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
    // buffer2 = buffer3 ^ T
    mkl_domatcopy('C', 'T', n1, n2, 1.0, C.data(), n1, buffer2.data(), n2);
    // buffer3 = intra_dist2 @ buffer2  
    st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, buffer2.data(), n1, n2, 0.0, C.data(), n2);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
    // buffer2 =  buffer3 ^ T
    mkl_domatcopy('C', 'T', n2, n1, (double) -2, C.data(), n2, buffer2.data(), n1);

    st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, C.data(), n1);   //just for test - not correct in algo
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm (A1^T S) failed");
    mkl_domatcopy('C', 'T', n1, n2, 1.0, C.data(), n1, buffer.data(), n2); //transpose
    mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,  1.0, row_norm_adj2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, buffer.data(), n1, n2, 0.0, C.data(), n2);      //spmm
    mkl_domatcopy( 'C', 'T', n2, n1, 1.0, C.data(), n2, buffer.data(), n1);

    cblas_daxpy(n1 * n2, 1.0, L_fixed.data(), 1, buffer2.data(), 1);  //L+L_fixed which L is buffer2

    // Start from cross_dist
    cblas_dcopy(n1 * n2, cross_dist.data(), 1, C.data(), 1);
    // buffer3 += lambda_e * L
    cblas_daxpy(n1 * n2, lambda_e, buffer2.data(), 1, C.data(), 1);

    vdLn(n1*n2, buffer.data(), buffer.data());
    // buffer3 -= lambda_n * log(buffer)
    cblas_daxpy(n1*n2, -lambda_n, buffer.data(), 1, C.data(), 1);
    // buffer3 -= lambda_a * log(H)  (H_log is precomputed)
    cblas_daxpy(n1*n2, -lambda_a, H_log.data(), 1, C.data(), 1);
}

void compute_C_L_fused( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2, 
        sparse_matrix_t intra_dist1, sparse_matrix_t intra_dist2, const std::vector<double>& L_fixed,
         const std::vector<double>& cross_dist, const std::vector<double>& H_log, double lambda_e, 
         double lambda_a, double lambda_n, std::vector<double>& C) {

    size_t matrix_size = (size_t)n1 * n2;
    std::vector<double> buffer(matrix_size);
    std::vector<double> buffer2(matrix_size);
    // std::vector<double> buffer3(n1*n2);   //is C 
    struct matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto time_0 = std::chrono::high_resolution_clock::now();
   
    // // buffer3 = intra_dist1 @ S  
    auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, C.data(), n1);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
    // buffer2 = buffer3 ^ T
    mkl_domatcopy('C', 'T', n1, n2, 1.0, C.data(), n1, buffer2.data(), n2);
    // buffer3 = intra_dist2 @ buffer2  
    st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, buffer2.data(), n1, n2, 0.0, C.data(), n2);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
    // buffer2 =  buffer3 ^ T
    mkl_domatcopy('C', 'T', n2, n1, (double) -2, C.data(), n2, buffer2.data(), n1);
    auto time_1 = std::chrono::high_resolution_clock::now();
   
    st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, C.data(), n1);   //just for test - not correct in algo
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm (A1^T S) failed");
    mkl_domatcopy('C', 'T', n1, n2, 1.0, C.data(), n1, buffer.data(), n2); //transpose
    mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,  1.0, row_norm_adj2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, buffer.data(), n1, n2, 0.0, C.data(), n2);      //spmm
    mkl_domatcopy( 'C', 'T', n2, n1, 1.0, C.data(), n2, buffer.data(), n1);
    vdLn((MKL_INT)((size_t)n1 * n2), buffer.data(), buffer.data());
    auto time_2 = std::chrono::high_resolution_clock::now();
    // const double log_floor = 1e-15;
    //C = cross_dist + lambda_e *  (L_fixed  + buffer2) - lambda_n * log(buffer) lambda_a * H_log
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i ;
            C[idx] = cross_dist[idx] + lambda_e * (L_fixed[idx] + buffer2[idx]) - lambda_n * buffer[idx] - lambda_a * H_log[idx]; 
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_3 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in compute_C_L_fused first 2spmm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " second 2spmm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " matrix iter : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << " ms" << std::endl);
}


void compute_cost_fused( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2,  sparse_matrix_t intra_dist1, 
        sparse_matrix_t intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C) {

    size_t matrix_size = (size_t)n1 * n2;
    std::vector<double> buffer(matrix_size);
    std::vector<double> buffer2(matrix_size);
    struct matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto time_0 = std::chrono::high_resolution_clock::now();
   
    // // buffer3 = intra_dist1 @ S  
    auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, C.data(), n1);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
    // buffer2 = buffer3 ^ T
    mkl_domatcopy('C', 'T', n1, n2, 1.0, C.data(), n1, buffer2.data(), n2);
    // buffer3 = intra_dist2 @ buffer2  
    st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, buffer2.data(), n1, n2, 0.0, C.data(), n2);
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
    // buffer2 =  buffer3 ^ T
    mkl_domatcopy('C', 'T', n2, n1, (double) -2, C.data(), n2, buffer2.data(), n1);
    auto time_1 = std::chrono::high_resolution_clock::now();
   
    st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, C.data(), n1);   //just for test - not correct in algo
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm (A1^T S) failed");
    mkl_domatcopy('C', 'T', n1, n2, 1.0, C.data(), n1, buffer.data(), n2); //transpose
    mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,  1.0, row_norm_adj2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, buffer.data(), n1, n2, 0.0, C.data(), n2);      //spmm
    mkl_domatcopy( 'C', 'T', n2, n1, 1.0, C.data(), n2, buffer.data(), n1);
    vdLn((MKL_INT)((size_t)n1 * n2), buffer.data(), buffer.data());
    auto time_2 = std::chrono::high_resolution_clock::now();
    // const double log_floor = 1e-15;
    //C = cross_dist + lambda_e *  (L_fixed  + buffer2) - lambda_n * log(buffer) lambda_a * H_log
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i ;
            C[idx] =  lambda_e * buffer2[idx] - lambda_n * buffer[idx] + C_fixed[idx];
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_3 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in compute_cost_fused first 2spmm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " second 2spmm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " matrix iter : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << " ms" << std::endl);
}


void compute_cost_fused_2( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2,  sparse_matrix_t intra_dist1, 
        sparse_matrix_t intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C) {

    size_t matrix_size = (size_t)n1 * n2;
    std::vector<double> buffer(matrix_size);
    std::vector<double> buffer2(matrix_size);
    struct matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto time_0 = std::chrono::high_resolution_clock::now();
    // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, intra_dist2, descr, SPARSE_LAYOUT_ROW_MAJOR, S.data(), n1, n1, 0.0, C.data(), n1);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
    omp_spmm_jik_v1(1.0, intra_dist2, n2, n1, 0.0, S.data(), C.data(), 64);
    // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, ((double) -2) * lambda_e, intra_dist1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), n2, n1, 0.0, buffer2.data(), n1);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
    omp_spmm_col_omp(((double) -2) * lambda_e, intra_dist1, n1, n2, 0.0, C.data(), buffer2.data());
    auto time_1 = std::chrono::high_resolution_clock::now();
   

    // Instead of computing 
    // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj2, descr, SPARSE_LAYOUT_ROW_MAJOR, S.data(), n1, n1, 0.0, C.data(), n1);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
    omp_spmm_jik_v1(1.0, row_norm_adj2, n2, n1, 0.0, S.data(), C.data(), 64);
    // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), n2, n1, 0.0, buffer.data(), n1);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
    omp_spmm_col_omp(1.0, row_norm_adj1, n1, n2, 0.0, C.data(), buffer.data());
    vdLn((MKL_INT)((size_t)n1 * n2), buffer.data(), buffer.data());
    auto time_2 = std::chrono::high_resolution_clock::now();
    // const double log_floor = 1e-15;
    //C = cross_dist + lambda_e *  (L_fixed  + buffer2) - lambda_n * log(buffer) lambda_a * H_log
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i ;
            C[idx] =  buffer2[idx] - lambda_n * buffer[idx] + C_fixed[idx];
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_3 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in compute_cost_fused_2 first 2spmm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " second 2spmm : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " matrix iter : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << " ms" << std::endl);
}


double update_S_mat (std::vector<double>& S, const std::vector<double>& S_old, const std::vector<double>& Cost, const std::vector<double>& a_row, const std::vector<double>& b_col, int n1, int n2, double epsilon) {

    unsigned int numThreads = mkl_get_max_threads();
    std::vector<double> sum_diss_S (numThreads, 0.0);
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        double a_val = a_row[j];
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t)j * n1 + i;
            S[idx] = (double) (0.95 * (double) std::exp(( a_val + b_col[i] - Cost[idx]) / epsilon) / (n1*n2)) + 0.05 * (double) (S_old[idx] );
            // sum_diss_S[omp_get_thread_num()] += std::abs(S[idx] - S_old[idx]);
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);

    // return  std::accumulate(sum_diss_S.begin(), sum_diss_S.end(), 0); 
    return  0.0;
}

std::vector<double> con_prox_pt_opt( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, int n1, int n2,int avg_nnz1_per_n1, int avg_nnz2_per_n2, 
    std::vector<double>& cross_dist, sparse_matrix_t intra_dist1_sparse, sparse_matrix_t intra_dist2_sparse, int inner_iters, int outer_iters, double lambda_n, double lambda_a, 
    double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>,std::vector<MKL_INT>>& anchor_pairs, bool sparsify) {

    auto time_0 = std::chrono::high_resolution_clock::now();
    double lambda_total = lambda_n + lambda_a + lambda_p;
    size_t N = (size_t)n1 * (size_t)n2;
    double lambda_e = lambda_e_in * N;
    std::vector<double> S(N, 1.0 / static_cast<double>(N));      // S = (1/(n1*n2)) * 1_n1 * 1_n2^T
    std::vector<double> S_old(N);
    std::vector<double> Cost_buff(N);
    std::vector<double> Cost(N);
    std::vector<double> Cost_fixed(N);
    std::vector<double> temp(N);
    std::vector<double> H_log = init_Hlog_with_anchors(n1, n2, anchor_pairs);
    std::vector<double> a_row(n2); 
    std::vector<double> b_col(n1, 1.0 / n2); 
    bool has_C_old = false;

    auto time_1 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "initialization in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() << " ms" << std::endl);

    //transpose row_norm_adj1 and row_norm_adj2
    sparse_matrix_t row_norm_adj1_tr = nullptr, row_norm_adj2_tr = nullptr; 
    auto st = mkl_sparse_convert_csr(row_norm_adj1, SPARSE_OPERATION_TRANSPOSE, &row_norm_adj1_tr);
    if (st != SPARSE_STATUS_SUCCESS) { throw std::runtime_error("row_norm_adj1 transpose failed");}
    st = mkl_sparse_convert_csr(row_norm_adj2, SPARSE_OPERATION_TRANSPOSE, &row_norm_adj2_tr);
    if (st != SPARSE_STATUS_SUCCESS) { throw std::runtime_error("row_norm_adj2 transpose failed");}
    // compute_L_fixed(intra_dist1_sparse, intra_dist2_sparse, n1, n2, L_fixed);
    // std::vector<double> L_fixed = compute_L_fixed_dense(intra_dist1, intra_dist2, r, c, one_vec_n1, one_vec_n2, n1, n2);
    // compute_Cost_fixed(intra_dist1_sparse, intra_dist2_sparse, cross_dist, H_log, L_fixed, n1, n2, lambda_e, lambda_a, Cost_fixed);
    // cross_dist = cross_dist + lambda_e * L_fixed - lambda_a * log(H)
    compute_Cost_fixed(intra_dist1_sparse, intra_dist2_sparse, cross_dist, H_log, n1, n2, lambda_e, lambda_a);
    //Free memory 
    H_log.clear();      H_log.shrink_to_fit();

    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "L_fixed computation in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << " ms" << std::endl << std::endl);
    auto L_compute_time=0, C_compute_time=0, Cost_compute_time=0, compute_minaa=0, compute_minbb=0, update_S=0, inner_loop=0;
    
    // printf("before outer iteration mkl nthreads %d omp nthreads %d\n", mkl_get_max_threads(), omp_get_num_threads());
    for (int it = 0; it < outer_iters; ++it) {
        DEBUG_ONLY(printf("----- outer iterration %d ----\n", it));
        auto time_iter_0 = std::chrono::high_resolution_clock::now();
        cblas_dcopy(N, S.data(), 1, S_old.data(), 1); 

        auto time_L_done = std::chrono::high_resolution_clock::now();
        // compute_C_L_fused( row_norm_adj1, row_norm_adj2, S, n1, n2, intra_dist1_sparse, intra_dist2_sparse, L_fixed, cross_dist, H_log, lambda_e, lambda_a, lambda_n, Cost);
        compute_cost_fused_2( row_norm_adj1_tr, row_norm_adj2, S, n1, n2, intra_dist1_sparse, intra_dist2_sparse, cross_dist, lambda_e, lambda_n, Cost);
        auto time_C_done = std::chrono::high_resolution_clock::now();
        C_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_C_done - time_L_done).count();

        // ---- C_old / W check ----
        // here Cost_buff is cost_old 
        if (!has_C_old) {
            cblas_dcopy(n1*n2, Cost.data(), 1, Cost_buff.data(), 1);
            has_C_old = true;
        } 
        else {
            double W_old = cblas_ddot(N, S.data(), 1, Cost_buff.data(), 1);
            double W     = cblas_ddot(N, S.data(), 1, Cost.data(), 1);
            if (W <= W_old)     cblas_dcopy(N, Cost.data(), 1, Cost_buff.data(), 1);
            else              cblas_dcopy(N, Cost_buff.data(), 1, Cost.data(), 1);
        }
        // ---- Cost = Cost - lambda_p * log(S) ----
        vdLn(N, S.data(), S.data());
        // cblas_dcopy(n1*n2, Cost.data(), 1, Cost.data(), 1);
        cblas_daxpy(N, -lambda_p, S.data(), 1, Cost.data(), 1);

        auto time_iter_1 = std::chrono::high_resolution_clock::now();
        Cost_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1 - time_C_done).count()  ;
        // ---- Sinkhorn inner iterations ----
        mkl_domatcopy('C', 'T', n1, n2, 1.0, Cost.data(), n1, temp.data(), n2);     //here temp is transpose of cost matrix
        for (int j = 0; j < inner_iters; ++j) {
            auto t_0 = std::chrono::high_resolution_clock::now() ;
            // a_row = minaa_v3_dense(Cost, b_col, n1, n2, lambda_total);
            minaa_v3_dense(Cost, b_col, n1, n2, lambda_total, a_row);
            auto t_1 = std::chrono::high_resolution_clock::now() ;
            compute_minaa += std::chrono::duration_cast<std::chrono::milliseconds>(t_1 - t_0).count()  ;
            minaa_v3_dense(temp, a_row, n2, n1, lambda_total, b_col);
            auto t_2 = std::chrono::high_resolution_clock::now() ;
            compute_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_2 - t_1).count()  ;
        }
        auto time_iter_2 = std::chrono::high_resolution_clock::now();        
        inner_loop += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_2 - time_iter_1).count()  ;

        // update S and returns diff_S = sum |S - S_old|
        double diff_S = update_S_mat(S, S_old, Cost, a_row, b_col, n1, n2, lambda_total);
        auto time_iter_3 = std::chrono::high_resolution_clock::now();        
        update_S += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_3 - time_iter_2).count()  ;
        
        //diff_S = sum |S - S_old|
        // using temp to store diff_S 
        // mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, S.data(), n1, -1.0, S_old.data(), n1, temp.data(), n1);
        // vdAbs(n1*n2, temp.data(), temp.data());
        // double diff_S = cblas_dasum(n1*n2, temp.data(), 1);
        DEBUG_ONLY(printf("inner iter %d and diff of S %.6f\n\n", it, diff_S));


        // if( it > 3 && sparsify ) {
        //     //sparsify S
        //     printf("S sparsified to 90\% percentile\n");
        //     sparsify_vector(S, percentile(S, 0.90));
        // }
        // printf("\n ----- end of iteration %d ----\n", it);
        // print_stats(S);
        // printf("\n ");

    }
    auto time_4 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "total time of 10 outer iterations : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_2).count() << " ms" << std::endl);
    DEBUG_ONLY(std::cout << " L_compute_time : " << L_compute_time << " C_compute_time : " << C_compute_time << " Cost_compute_time : " << Cost_compute_time <<
        " mina : " << compute_minaa << " minb : " << compute_minbb << " update_S : " << update_S << " ms" << std::endl);
    DEBUG_ONLY(std::cout << "done con_prox_pt_opt_init" << std::endl);
    return S;
}   

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

void l2_norm_inplace(std::vector<double>& rwr, size_t n, size_t d) {
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
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
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
}

void get_normalized_neg_exp_dist (const std::vector<double>& rwr1, const std::vector<double>&  rwr2, int n1, int n2, int l, std::vector<double>& cross_rwr_dist) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, (MKL_INT)n1, (MKL_INT)n2, (MKL_INT)l, -1.0, rwr1.data(), (MKL_INT)n1, 
            rwr2.data(), (MKL_INT)n2, 0.0, cross_rwr_dist.data(), (MKL_INT)n1);
    size_t total = (size_t)n1 * (size_t)n2;
    vdExp((MKL_INT)total, cross_rwr_dist.data(), cross_rwr_dist.data());
}

void masked_sddmm_intra_cost (std::vector<double>&  emb, sparse_matrix_t mask, int n, int L, sparse_matrix_t& result) {
    matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto st = mkl_sparse_copy(mask, descr, &result);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("Copy failed");
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start, *row_end, *col_ind;
    double *result_vals;
    st = mkl_sparse_d_export_csr( result, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, &result_vals);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("Export CSR failed");
    if (rows != n || cols != n)
        throw std::runtime_error("mask dimensions do not match n");

    if (emb.size() != (size_t)n * (size_t)L)
        throw std::runtime_error("emb size mismatch");
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for schedule(dynamic)
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

sparse_matrix_t row_normalize_graph_old( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) {

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
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_add failed in row_normalize_graph");

        mkl_sparse_destroy(add_mat);

        // //update graph_row_sum
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_connected, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in row_normalize_graph");
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
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_spmm failed in row_normalize_graph");


    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_norm, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in row_normalize_graph");
    for(int i=0; i<graph_row_sum.size(); i++)
        if (graph_row_sum[i] < 1e-10) DEBUG_ONLY(printf("222 graph_row_sum[%d] is zero \n", i));

    mkl_sparse_optimize(graph_norm);
    sparse_matrix_t graph_norm_csr = nullptr;
    status = mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_convert_csr failed in row_normalize_graph");
    status = mkl_sparse_optimize(graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_optimize failed in row_normalize_graph");  
    
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_norm_csr, descr, all_ones.data(), (double) 0.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in row_normalize_graph");
    for(int i=0; i<graph_row_sum.size(); i++)
        if (graph_row_sum[i] < 1e-10) DEBUG_ONLY(printf("333 graph_row_sum[%d] is zero \n", i));


    return graph_norm_csr;
}

sparse_matrix_t row_normalize_graph(sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) {
    
    // Export CSR once - compute row sums directly
    sparse_index_base_t indexing;
    MKL_INT rows, cols, *row_start, *row_end, *col_ind;
    double *values;
    mkl_sparse_d_export_csr(graph, &indexing, &rows, &cols, 
                            &row_start, &row_end, &col_ind, &values);
    
    // Direct row sum computation
    std::vector<double> graph_row_sum(n, 0.0);
    std::vector<MKL_INT> isolated;
    
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);  
    #pragma omp parallel for
    for (MKL_INT i = 0; i < n; ++i) {
        double sum = 0.0;
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            sum += values[idx];
        }
        graph_row_sum[i] = sum;
    }
    
    for (MKL_INT i = 0; i < n; ++i) {
        if (graph_row_sum[i] < 1e-10) isolated.push_back(i);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);  
    // Deep copy graph
    sparse_matrix_t graph_connected = nullptr;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_copy(graph, descr, &graph_connected);
    
    if (!isolated.empty()) {
        std::vector<double> edge_list;
        edge_list.reserve(isolated.size() * avg_nnz_per_n * 4);
        
        std::mt19937 rng(std::time(0));
        std::uniform_int_distribution<MKL_INT> dist(0, n - 1);
        
        for (MKL_INT i : isolated) {
            std::vector<MKL_INT> random_nodes;
            random_nodes.reserve(avg_nnz_per_n);
            
            while (random_nodes.size() < avg_nnz_per_n) {
                MKL_INT r = dist(rng);
                if (r != i && std::find(random_nodes.begin(), 
                    random_nodes.end(), r) == random_nodes.end()) {
                    random_nodes.push_back(r);
                    edge_list.push_back(i);
                    edge_list.push_back(r);
                    edge_list.push_back(r);
                    edge_list.push_back(i);
                }
            }
        }
        
        sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, n, n);
        sparse_matrix_t temp = nullptr;
        mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, graph_connected, 
                        1.0, add_mat, &temp);
        mkl_sparse_destroy(add_mat);
        mkl_sparse_destroy(graph_connected);
        graph_connected = temp;
    }
    
    // In-place normalization
    mkl_sparse_d_export_csr(graph_connected, &indexing, &rows, &cols,
                           &row_start, &row_end, &col_ind, &values);
    
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);                        
    // Recompute row sums if modified
    if (!isolated.empty()) {
        #pragma omp parallel for
        for (MKL_INT i = 0; i < n; ++i) {
            double sum = 0.0;
            for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
                sum += values[idx];
            }
            graph_row_sum[i] = sum;
        }
    }
    
    #pragma omp parallel for
    for (MKL_INT i = 0; i < n; ++i) {
        double inv_sum = (graph_row_sum[i] > 1e-10) ? 1.0 / graph_row_sum[i] : 0.0;
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            values[idx] *= inv_sum;
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);  
    
    mkl_sparse_optimize(graph_connected);
    return graph_connected;
}

void get_batch_rwr_scores(sparse_matrix_t graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, 
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) {
    auto time_0 = std::chrono::high_resolution_clock::now();
    int l = anchors.size();
    const size_t N = (size_t)n * (size_t)l;
    std::vector<double> X_vals(N, 0.0);
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);     
    #pragma omp parallel for
    for (int j = 0; j < l; ++j) {
        X_vals[(size_t)anchors[j] + (size_t)j * (size_t)n] = restart_prob;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);   
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto add_after_spmm_time = 0, spmm_time = 0, swap_after_spmm_time = 0;
    for (int i = 0; i < max_iters - 1; i++) {
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        omp_spmm_col_omp(1.0, graph_norm_csr, n, l, 0.0, X_vals.data(), Xnext_vals.data());
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        mkl_set_num_threads(1);
        omp_set_num_threads(numThreads);  
        #pragma omp parallel for
        for (int j = 0; j < l; ++j) {
            Xnext_vals[(size_t)anchors[j] + (size_t)j * (size_t)n] += restart_prob;
        }
        mkl_set_num_threads(numThreads);
        omp_set_num_threads(1);  
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        add_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        std::swap(X_vals, Xnext_vals);
        auto time_1_3 = std::chrono::high_resolution_clock::now();
        swap_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2).count();
    }

    std::swap(X_vals, Xnext_vals);
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1); 

    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_batch_rwr_scores generate E " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " spmm time " << spmm_time << " add after spmm time " << add_after_spmm_time <<
        " swap after spmm time " << swap_after_spmm_time << std::endl);
}

void get_batch_rwr_scores_rowmjr(sparse_matrix_t graph_norm_csr, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, 
        bool directed, float restart_prob, int max_iters, bool connect_isolated, std::vector<double>& Xnext_vals) {
    auto time_0 = std::chrono::high_resolution_clock::now();
    int l = anchors.size();
    const size_t N = (size_t)n * (size_t)l;
    std::vector<double> X_vals(N, 0.0);
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for
    for (int j = 0; j < l; ++j) {
        X_vals[(size_t)anchors[j] * (size_t)l + (size_t)j] = restart_prob;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto add_after_spmm_time = 0, spmm_time = 0, swap_after_spmm_time = 0;
    for (int i = 0; i < max_iters - 1; i++) {
        auto time_1_0 = std::chrono::high_resolution_clock::now();
        omp_spmm_jik_v1(1.0, graph_norm_csr, n, l, 0.0, X_vals.data(), Xnext_vals.data(), 64);
        auto time_1_1 = std::chrono::high_resolution_clock::now();
        spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_1 - time_1_0).count();
        mkl_set_num_threads(1);
        omp_set_num_threads(numThreads);
        #pragma omp parallel for
        for (int j = 0; j < l; ++j) {
            Xnext_vals[(size_t)anchors[j] * (size_t)l + (size_t)j] += restart_prob;
        }
        mkl_set_num_threads(numThreads);
        omp_set_num_threads(1);
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        add_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        std::swap(X_vals, Xnext_vals);
        auto time_1_3 = std::chrono::high_resolution_clock::now();
        swap_after_spmm_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_3 - time_1_2).count();
    }
    std::swap(X_vals, Xnext_vals);

    // std::vector<double> Xnext_vals_colmajor(N, 0.0);
    // mkl_set_num_threads(1);
    // omp_set_num_threads(numThreads);
    // #pragma omp parallel for
    // for (MKL_INT row = 0; row < n; ++row) {
    //     for (int col = 0; col < l; ++col) {
    //         Xnext_vals_colmajor[(size_t)row + (size_t)col * (size_t)n] =
    //             Xnext_vals[(size_t)row * (size_t)l + (size_t)col];
    //     }
    // }
    // Xnext_vals.swap(Xnext_vals_colmajor);
    // mkl_set_num_threads(numThreads);
    // omp_set_num_threads(1);

    mkl_dimatcopy('R', 'T', n, l, 1.0, Xnext_vals.data(), n, l);

    auto time_2 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_batch_rwr_scores_rowmjr generate E " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " spmm time " << spmm_time << " add after spmm time " << add_after_spmm_time <<
        " swap after spmm time " << swap_after_spmm_time << std::endl);
}

void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, const std::pair<std::vector<MKL_INT>, 
                    std::vector<MKL_INT>>& anchors, double val) {
    const MKL_INT k = (MKL_INT)anchors.first.size();
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);               
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
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);  
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
    double coeff = (1.0 - rwr_restart_prob) * gamma;
    double alpha = 1.0 + rwr_restart_prob;
    unsigned int numThreads = mkl_get_max_threads();
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

        // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, temp.data(), n2, n1, 0.0, out_cross_dist.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        omp_spmm_col_omp(coeff, graph1_normalized, n1, n2, 0.0, temp.data(), out_cross_dist.data());
        auto time_1_2 = std::chrono::high_resolution_clock::now();
        time_spmm2 += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_2 - time_1_1).count();
        bool converged = true;
        omp_set_num_threads(numThreads);
        mkl_set_num_threads(1);
        #pragma omp parallel for reduction(&:converged)
        for (int i = 0; i < N; ++i) {
            out_cross_dist[i] += alpha * in_cross_dist[i];
            converged &= (std::abs(out_cross_dist[i] - out_cross_dist_old[i]) < tol);
        }
        if (converged) break;
        omp_set_num_threads(1);
        mkl_set_num_threads(numThreads);
        auto time_1_4 = std::chrono::high_resolution_clock::now();
        time_convergence += std::chrono::duration_cast<std::chrono::milliseconds>(time_1_4 - time_1_2).count();
        std::swap(out_cross_dist, out_cross_dist_old);
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
    auto time_2 = std::chrono::high_resolution_clock::now();
    std::swap(out_cross_dist, out_cross_dist_old);
    dscal_chunked(N, 1.0 - gamma, out_cross_dist.data());
    auto time_3 = std::chrono::high_resolution_clock::now();
    DEBUG_ONLY(std::cout << "in get_product_rwr_mat init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
        " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
        " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << std::endl);
    DEBUG_ONLY(std::cout << "in for loop 1st spmm " << time_spmm1 << " 2nd spmm " << time_spmm2 << " convergence " << time_convergence << std::endl);
}

void minaa_v3_dense_old( std::vector<double>& Cost, const std::vector<double>& b, int n_row, int n_col, double epsilon, std::vector<double>& out_vec ) {
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

void minaa_v3_dense( std::vector<double>& Cost, const std::vector<double>& b, int n_row, int n_col, double epsilon, std::vector<double>& out_vec ) {
    const unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    const double inv_epsilon    = 1.0 / epsilon;
    const double inv_n_row      = 1.0 / static_cast<double>(n_row);
    const double* __restrict__ bptr = b.data();
    const int nT = static_cast<int>(numThreads);
    std::vector<double> scratch(static_cast<size_t>(nT) * n_row);
    
    #pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        double* __restrict__ local = scratch.data() + (size_t)tid * n_row;
        #pragma omp for schedule(dynamic, 4)
        for (int j = 0; j < n_col; ++j)
        {
            const double* __restrict__ col = Cost.data() + (size_t)j * n_row;
            double min_j = std::numeric_limits<double>::max();
            #pragma omp simd reduction(min : min_j)
            for (int i = 0; i < n_row; ++i)
            {
                const double v = col[i] - bptr[i];
                local[i] = v;          // store shifted values
                min_j = (v < min_j) ? v : min_j;
            }
            double s_j = 0.0;
            #pragma omp simd reduction(+ : s_j)
            for (int i = 0; i < n_row; ++i)
            {
                s_j += std::exp((min_j - local[i]) * inv_epsilon);
            }
            out_vec[j] = min_j - epsilon * (std::log(s_j) + std::log(inv_n_row));
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
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

void compute_cost_fused_2( sparse_matrix_t row_norm_adj1, sparse_matrix_t row_norm_adj2, const std::vector<double>& S, int n1, int n2,  sparse_matrix_t intra_dist1,
        sparse_matrix_t intra_dist2, const std::vector<double>& C_fixed, double lambda_e, double lambda_n, std::vector<double>& C,
        std::vector<double>& buffer, std::vector<double>& buffer2) {
    size_t matrix_size = (size_t)n1 * n2;
    if (buffer.size() != matrix_size) buffer.resize(matrix_size);
    if (buffer2.size() != matrix_size) buffer2.resize(matrix_size);

    auto time_0 = std::chrono::high_resolution_clock::now();
    omp_spmm_jik_v1(1.0, intra_dist2, n2, n1, 0.0, S.data(), buffer.data(), 64);
    omp_spmm_col_omp(((double) -2) * lambda_e, intra_dist1, n1, n2, 0.0, buffer.data(), buffer2.data());
    auto time_1 = std::chrono::high_resolution_clock::now();
    omp_spmm_jik_v1(1.0, row_norm_adj2, n2, n1, 0.0, S.data(), buffer.data(), 64);
    omp_spmm_col_omp(1.0, row_norm_adj1, n1, n2, 0.0, buffer.data(), C.data());
    auto time_2 = std::chrono::high_resolution_clock::now();
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            size_t idx = (size_t) j * n1 + i;
            C[idx] = buffer2[idx] - lambda_n * std::log(C[idx]) + C_fixed[idx];
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
    std::vector<double> temp2(N);
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
        // compute_cost_fused_2( row_norm_adj1_tr, row_norm_adj2, S, n1, n2, intra_dist1_sparse, intra_dist2_sparse, cross_dist, lambda_e, lambda_n, Cost);
        compute_cost_fused_2( row_norm_adj1_tr, row_norm_adj2, S, n1, n2, intra_dist1_sparse, intra_dist2_sparse, cross_dist, lambda_e, lambda_n, Cost, temp, temp2);
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

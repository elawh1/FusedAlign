#ifdef MKL_AVAILABLE
#include <mkl.h>
#include <omp.h>
#include <mkl_vml.h>
#endif
#include <cstdio>
#include <cinttypes>   
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <numeric>
#include <set>
#include "helper2.h"
#include "fused_spmm.h"

struct MKLCSC {
    sparse_matrix_t spMat = nullptr;
    std::vector<MKL_INT> col_start, col_end, row_ind;
    std::vector<double>  vals;
    ~MKLCSC() { if (spMat) mkl_sparse_destroy(spMat); }
};

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

std::vector<double> get_normalized_neg_exp_dist (std::vector<double>  rwr1, std::vector<double>  rwr2, int n1, int n2, int l) {
    std::vector<double> rwr1_norm = l2_norm( rwr1, n1, l);
    std::vector<double> rwr2_norm = l2_norm( rwr2, n2, l);
    std::vector<double> cross_rwr_dist(n1 * n2);
    // cross_rwr_dist = exp( - rwr1_norm * rwr2_norm^T )
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, l, -1.0, rwr1_norm.data(), n1, rwr2_norm.data(), n2, 0.0, cross_rwr_dist.data(), n1);
    vdExp(n1 * n2, cross_rwr_dist.data(), cross_rwr_dist.data());
    return cross_rwr_dist;
}

sparse_matrix_t masked_sddmm_intra_cost (std::vector<double>  emb, sparse_matrix_t mask, int n, int L) {
    std::vector<double> emb_norm = l2_norm( emb, n, L);
    
    sparse_matrix_t result;
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
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            MKL_INT j = col_ind[idx];
            // mask_vals[idx] = dense_mat[j * rows + i];
            // inner product of two emb
            double product = emb_norm[i] * emb_norm[j];
            for ( int l=1; l < L; ++l) {
                product += emb_norm[l * rows + i] * emb_norm[l * rows + j];
            }
            product = std::exp(-1.0 * product);
            result_vals[idx] = product ;     // inner product of two emb
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return result;
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

std::vector<double>  get_batch_rwr_scores(sparse_matrix_t graph, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, bool directed, float restart_prob, int max_iters, bool connect_isolated) {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto time_0 = std::chrono::high_resolution_clock::now();
    
    //connect isolated nodes and row normalize A  
    std::vector<double> all_ones(n, (double) 1.0);
    std::vector<double> graph_row_sum(n, (double) 0.0);
    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    
    // -----    TODO   -----  Connect isolated nodes
    std::vector<MKL_INT> isolated;
    bool isolated_node = false ;
    for (MKL_INT i = 0; i < n; ++i) {
        if (graph_row_sum[i] == 0.0) {
            isolated.push_back(i);
            isolated_node = true;
        }   
    }
    printf("num of isolated %ld\n", isolated.size());
    sparse_matrix_t graph_connected = nullptr;
    if(isolated_node) {
        printf("connecting isolated nodes\n");

        std::vector<double> edge_list;

        
        // // connect to all nodes
        // edge_list.reserve(isolated.size() * n * 2);
        // for (MKL_INT i : isolated) {
        //     for (MKL_INT j = 0; j < n; ++j) {
        //         edge_list.push_back(static_cast<double>(i));
        //         edge_list.push_back(static_cast<double>(j));
        //     }
        //     // update row sum directly
        //     graph_row_sum[i] = static_cast<double>(n);
        // }
        

        // connect to #avg_nnz_per_n number of random nodes
        edge_list.reserve(isolated.size() * avg_nnz_per_n * 2);
        std::srand(std::time(0));
        for (MKL_INT i : isolated) {
            // printf("isolated node %d\n", i);
            std::set<MKL_INT> random_nodes = {};
            while(random_nodes.size() < avg_nnz_per_n) {
                MKL_INT random_num = std::rand() % n;
                if ( random_nodes.find(random_num) !=  random_nodes.end()) continue;
                if ( random_num >= n )          std::cerr << "random num generated out of bound\n" << std::endl; 
                random_nodes.insert(random_num);
                edge_list.push_back(static_cast<double>(i));
                edge_list.push_back(static_cast<double>(random_num));
            }
            // update row sum directly
            graph_row_sum[i] = static_cast<double>(avg_nnz_per_n);
            // printf("isolated node %d connected to %d nodes\n", i, edge_list.size());
        }
        
        // printf("connect isolated nodes : num fo edges to add %d\n", edge_list.size()/2);
        // build CSR matrix (values are implicitly 1.0 inside)
        sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, n, n);
        //add add_mat to original graph
        status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE,graph, 1.0, add_mat, &graph_connected);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_add failed in get_batch_rwr_scores");

        mkl_sparse_destroy(add_mat);

        // //update graph_row_sum
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_connected, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    }
    else {
        graph_connected = graph;
    }
    
    // std::vector<double> graph_row_sum = graph_connect_isolated(graph);
    // sparse_matrix_t graph_connected;
    // std::vector<double> graph_row_sum(n, (double) 0.0);
    // graph_connect_isolated(graph, graph_row_sum, graph_connected);
    auto time_1 = std::chrono::high_resolution_clock::now();

    MKLCSC Dinv = make_diag_csc(graph_row_sum, n, n, true);
    sparse_matrix_t graph_norm = nullptr;    
    mkl_sparse_optimize(Dinv.spMat);
    mkl_sparse_optimize(graph_connected);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph_connected, &graph_norm);
    // auto status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph, &graph_norm);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_spmm failed in get_batch_rwr_scores");
    mkl_sparse_optimize(graph_norm);
    sparse_matrix_t graph_norm_csr = nullptr;
    status = mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_convert_csr failed in get_batch_rwr_scores");
    status = mkl_sparse_optimize(graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_optimize failed in get_batch_rwr_scores");  
    
    // sparse_matrix_t graph_norm_csr = graph_row_normalize_connect_isolated( graph);

    auto time_2 = std::chrono::high_resolution_clock::now();
    int l = anchors.size();
    //one hot encoding E matrix
    std::vector<double> E(n * l, 0.0);
    for (int j = 0; j < l; ++j) {
        E[ anchors[j] + j * n ] = 1.0;   // one-hot at (row=anchor[j], col=j)
    }
    //these are column major
    // std::vector<double> X_vals(n*l, 1.0);      // X(t), all ones
    std::vector<double> X_vals = E;
    std::vector<double> Xnext_vals(n*l, 0.0);
    // std::vector<double> graph_norm_csr_dense = sparse_to_dense_colmajor(graph_norm_csr, n, n);
    auto time_3 = std::chrono::high_resolution_clock::now();
    // for(int i=0; i < max_iters; i++) {
    for(int i=0; i < max_iters-1; i++) {
        // status = mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, (double) (1-restart_prob), graph_norm_csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X_vals.data(), l, n, 0.0, Xnext_vals.data(), n);
        // if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("error spmm");
        // cblas_daxpy(n * l, restart_prob, E.data(), 1, Xnext_vals.data(), 1);
        // std::swap(X_vals, Xnext_vals);
        Xnext_vals = E;
        auto status = mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, (double) (1-restart_prob), graph_norm_csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X_vals.data(), l, n, restart_prob, Xnext_vals.data(), n);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("error spmm");
        // cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, l, n,(double) (1-restart_prob), graph_norm_csr_dense.data(), n, X_vals.data(), n, restart_prob, Xnext_vals.data(), n);
        X_vals = Xnext_vals;
    }
    auto time_4= std::chrono::high_resolution_clock::now();
    std::cout << "connect isolated and normalize : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_0).count()  
    << " rwr before iterations: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()  
    << " rwr iteration=100: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count()  
    << " ms" << std::endl;

    //column major dense matrix nxl
	return Xnext_vals;
}

void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchors, double val) {
    const MKL_INT k = (MKL_INT) anchors.first.size();
    std::vector<MKL_INT> idx(k);
    for (MKL_INT t = 0; t < k; ++t) {
        if (anchors.first[t] < 0 || anchors.first[t] >= nrows || anchors.second[t] < 0 || anchors.second[t] >= ncols) {
            printf( "error anchor_cost_zero: out-of-range anchor (%d,%d)\n", anchors.first[t], anchors.second[t]);
        }
        idx[t] = anchors.first[t] + anchors.second[t] * nrows;   //if A stored column wise 
    }
    std::vector<double> z(k, val);
    cblas_dsctr(k, z.data(), idx.data(), A.data());
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
    std::vector<double> H( nrows * ncols, 0.0);
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
        MKL_INT idx = i + j * nrows;    // column-major: (row, col) -> index
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


sparse_matrix_t row_normalize_graph( sparse_matrix_t graph, int n, MKL_INT avg_nnz_per_n) {
    
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
    printf("num of isolated %ld\n", isolated.size());

    sparse_matrix_t graph_connected = nullptr;
    if(isolated_node) {
        printf("connecting isolated nodes\n");

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

bool out_cross_dist_converge( std::vector<double>& out_cross_dist, std::vector<double>& out_cross_dist_old, double tol) {
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    bool converged = true;
    #pragma omp parallel for reduction(&:converged)
    for (int i = 0; i < out_cross_dist.size(); ++i) {
        if (std::abs(out_cross_dist[i] - out_cross_dist_old[i]) >= tol)     converged = false;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return converged;
}

std::vector<double> get_product_rwr_mat_fused( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, 
    int n1, int n2, int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    double rwr_restart_prob, double gamma, int max_iters = 50, double tol = 1e-2) {
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

std::vector<double> get_product_rwr_mat( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, 
    int n1, int n2, int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    double rwr_restart_prob, double gamma, int max_iters = 50, double tol = 1e-2) {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto time_0 = std::chrono::high_resolution_clock::now();
    sparse_matrix_t graph1_normalized = row_normalize_graph( graph1_py, n1, avg_nnz1_per_n1);
    sparse_matrix_t graph2_normalized = row_normalize_graph( graph2_py, n2, avg_nnz2_per_n2);
    // sparse_matrix_t graph1_normalized = graph_row_normalize_connect_isolated( graph1_py);
    // sparse_matrix_t graph2_normalized = graph_row_normalize_connect_isolated( graph2_py);
    
    auto time_1 = std::chrono::high_resolution_clock::now();
    std::vector<double> out_cross_dist(n1 * n2, 0.0);
    std::vector<double> out_cross_dist_old(n1 * n2, 0.0);
    std::vector<double> temp(n1 * n2);
    std::vector<double> CC(n1 * n2);
    auto time_2 = std::chrono::high_resolution_clock::now();
    

    // print_sparse_matrix_info(graph1_py, "graph1_py");
    // print_sparse_matrix_info(graph2_py, "graph2_py");

    // print_sparse_matrix_info(graph1_normalized, "graph1_normalized");
    // print_sparse_matrix_info(graph2_normalized, "graph2_normalized");

    for (int i = 0; i < max_iters; ++i) {
        auto time_2_0 = std::chrono::high_resolution_clock::now();

        /*    SPPM and add at end  -----
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
        
        // out_cross_dist = CC;
        // fused spmm 
        // out_cross_dist = blockwise_colsum_matmul( graph1_normalized, graph2_normalized, out_cross_dist, n1, n2, 10);
        auto time_2_0_3 = std::chrono::high_resolution_clock::now();
        // std::cout << "*** new method " << i <<
        // " --  new spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_0_2-time_2_0_1).count() << 
        // " -- copy " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_0_3-time_2_0_2).count() << std::endl;
        auto time_2_1 = std::chrono::high_resolution_clock::now();
        //compute max of absolute values of (out_cross_dist - out_cross_dist_old)
        if (out_cross_dist_converge(out_cross_dist, out_cross_dist_old, tol)) break;
        auto time_2_2 = std::chrono::high_resolution_clock::now();
        // vdSub(n1 * n2, out_cross_dist_old.data(), out_cross_dist.data(), temp.data());   //temp has diff 
        // vdAbs(n1*n2, temp.data(), out_cross_dist_old.data());   // out_cross_dist_old holds the abs of diff
        // double max_diff = *std::max_element(out_cross_dist_old.begin(), out_cross_dist_old.end());
        // if (max_diff < tol) break;

        // std::cout << "in get_product_rwr_mat for loop " << i <<
        // " spmm ops " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_1-time_2_0).count() << 
        // " out_cross_dist_converge " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_2-time_2_1).count() 
        // << std::endl;

        // std::cout << "*** in spmm ops in loop " << i <<
        // " -- cpy out_cross_dist_old " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_0_1-time_2_0).count() << 
        // " -- 1st spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_0_2-time_2_0_1).count() <<
        // " -- transpose " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_0_3-time_2_0_2).count() <<
        // " -- 2nd spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_0_4-time_2_0_3).count() <<
        // " -- transpose and scalar" << std::chrono::duration_cast<std::chrono::milliseconds>(time_2_1-time_2_0_4).count() << std::endl;
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

std::vector<double> minaa_v3_dense( std::vector<double> &H, const std::vector<double>& b, int n1, int n2, double epsilon) {
    std::vector<double> res(n2);
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (int j = 0; j < n2; ++j) {
        double* col = &H[(size_t)j * n1];
        double min_j = (col[0] -= b[0]);
        for (int i = 1; i < n1; ++i) {
            col[i] -= b[i];
            min_j = std::min(min_j, col[i]);
        }
        double s_j = 0.0;
        for (int i = 0; i < n1; i++) {
            double shifted = col[i] - min_j;
            col[i] += b[i];
            s_j += std::exp(-(shifted / epsilon));
        }
        res[j] = min_j - epsilon * std::log(s_j / n1);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return res;
}

std::vector<double> row_avg_square ( sparse_matrix_t& intra_dist, int n) {
    std::vector<double> row_avg_sqr(n);
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
        if (i == 0) printf(" ---- row_avg_square  --- threads = %d\n", omp_get_num_threads());
        double square_sum = 0.0;
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            square_sum += values[idx] * values[idx];
        }
        row_avg_sqr[i] = (double) square_sum / n;
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return row_avg_sqr;
}

std::vector<double> compute_L_fixed( sparse_matrix_t& intra_dist1, sparse_matrix_t& intra_dist2, int n1, int n2) {
    // 1) Square intra_dist1 and intra_dist2 elementwise
    std::vector<double> row_avg_square_dist1 = row_avg_square(intra_dist1, n1);
    std::vector<double> row_avg_square_dist2 = row_avg_square(intra_dist2, n2);
    unsigned int numThreads = mkl_get_max_threads();
    printf("inside compute_L_fixed nthreads %d\n", numThreads);
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    std::vector<double> L_fixed(n1 * n2);
    #pragma omp parallel for schedule(static)
    for( int j=0; j < n2; ++j) {
        double dist2_term =  row_avg_square_dist2[j];
        for(int i=0; i < n1; ++i) {
            L_fixed[j * n1 + i] = dist2_term + row_avg_square_dist1[i];
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    printf("before exitign compute_L_fixed nthreads %d\n", mkl_get_max_threads());
    return L_fixed; // n1 x  n2, col-major
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


void update_S_mat (std::vector<double>& S, std::vector<double>& S_old, const std::vector<double>& Cost, const std::vector<double>& a_row, const std::vector<double>& b_col, int n1, int n2, double epsilon) {
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (int j = 0; j < n2; ++j) {
        double a_val = a_row[j];
        for (int i = 0; i < n1; ++i) {
            S[j * n1 + i] = (double) (0.95 * (double) std::exp(( a_val + b_col[i] - Cost[j * n1 + i]) / epsilon) / (n1*n2)) + 0.05 * (double) (S_old[j * n1 + i] );
        } 
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
}

std::vector<double> con_prox_pt_opt( const std::string data_path, sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, int n1, int n2,
    int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    std::vector<double> cross_dist, sparse_matrix_t intra_dist1_sparse, sparse_matrix_t intra_dist2_sparse, int inner_iters, 
    int outer_iters, double lambda_n, double lambda_a, double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>, 
    std::vector<MKL_INT>>& anchor_pairs, bool sparsify) {

    auto time_0 = std::chrono::high_resolution_clock::now();
    double lambda_total = lambda_n + lambda_a + lambda_p;
    double lambda_e = lambda_e_in * n1 * n2;
    std::vector<double> S(n1 * n2, 1.0 / static_cast<double>(n1 * n2));      // S = (1/(n1*n2)) * 1_n1 * 1_n2^T
    std::vector<double> buffer(n1*n2);
    std::vector<double> S_old(n1*n2);
    std::vector<double> L(n1*n2, 0.0);
    std::vector<double> C(n1*n2);
    std::vector<double> C_old(n1*n2);
    std::vector<double> Cost(n1*n2);
    std::vector<double> Cost_trans(n1*n2);
    std::vector<double> S_log(n1*n2);
    std::vector<double> diff_abs(n1*n2);
    std::vector<double> RW_term(n1*n2);
    std::vector<double> H_log = init_Hlog_with_anchors(n1, n2, anchor_pairs);
    bool has_C_old = false;

    auto time_1 = std::chrono::high_resolution_clock::now();
    std::cout << "initialization in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   << " ms" << std::endl;
   
    sparse_matrix_t row_norm_adj1 = row_normalize_graph(graph1_py, n1, avg_nnz1_per_n1);
    sparse_matrix_t row_norm_adj2 = row_normalize_graph(graph2_py, n2, avg_nnz2_per_n2);
    // sparse_matrix_t row_norm_adj1 = graph_row_normalize_connect_isolated(graph1_py, n1);
    // sparse_matrix_t row_norm_adj2 = graph_row_normalize_connect_isolated(graph2_py, n2);
    
    //Sparse adj to dense
    // std::vector<double> dense_adj1 = sparse_to_dense_colmajor(row_norm_adj1, n1, n1);
    // std::vector<double> dense_adj2 = sparse_to_dense_colmajor(row_norm_adj2, n2, n2);

    auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "row normalizing graphs in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   << " ms" << std::endl;
    std::vector<double> L_fixed = compute_L_fixed(intra_dist1_sparse, intra_dist2_sparse, n1, n2);
    // std::vector<double> L_fixed = compute_L_fixed_dense(intra_dist1, intra_dist2, r, c, one_vec_n1, one_vec_n2, n1, n2);
    auto time_3 = std::chrono::high_resolution_clock::now();
    std::cout << "L_fixed computation in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()   << " ms" << std::endl;

    auto L_compute_time=0, C_compute_time=0, Cost_compute_time=0, before_minaa=0, compute_minaa=0, between_minaa_minbb=0, compute_minbb=0, update_S=0, inner_loop=0;
    printf("before outer iteration mkl nthreads %d omp nthreads %d\n", mkl_get_max_threads(), omp_get_num_threads());
    for (int it = 0; it < outer_iters; ++it) {
        printf("----- outer iterration %d ----\n", it);
        auto time_iter_0 = std::chrono::high_resolution_clock::now();
        cblas_dcopy(n1 * n2, S.data(), 1, S_old.data(), 1);
        // Computes  - 2 * intra_dist1 @ S @ intra_dist2^T
        // Computing L = L_fixed - 2 * A2T_term    since L is - 2 * A2T_term 
        // compute_A2T_term_dense(intra_dist1, intra_dist2, S, n1, n2, L);
        compute_A2T_term(intra_dist1_sparse, intra_dist2_sparse, S, n1, n2, L);
        // L = blockwise_outputcols_matmul( intra_dist1_sparse, intra_dist2_sparse, S, n1, n2, 10, -2.0);
        // L = blockwise_colsum_matmul( intra_dist1_sparse, intra_dist2_sparse, S, n1, n2, 10, -2.0);
        cblas_daxpy(n1 * n2, 1.0, L_fixed.data(), 1, L.data(), 1);
        auto time_L_done = std::chrono::high_resolution_clock::now();
        L_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_L_done - time_iter_0).count()  ;

        // ---- C = cross_dist + lambda_e * L - lambda_n * log(row_norm_adj1.T @ S @ row_norm_adj2) - lambda_a * log(H) ----
        // Start from cross_dist
        cblas_dcopy(n1 * n2, cross_dist.data(), 1, C.data(), 1);
        // C += lambda_e * L
        cblas_daxpy(n1 * n2, lambda_e, L.data(), 1, C.data(), 1);

        compute_RW_term ( row_norm_adj1, row_norm_adj2, S, n1, n2, RW_term);
        // compute_RW_term_dense( dense_adj1, dense_adj2, S, n1, n2, RW_term);

        // log(RW_term)
        vdLn(n1*n2, RW_term.data(), RW_term.data());
        // C -= lambda_n * log(RW_term)
        cblas_daxpy(n1*n2, -lambda_n, RW_term.data(), 1, C.data(), 1);
        // C -= lambda_a * log(H)  (H_log is precomputed)
        cblas_daxpy(n1*n2, -lambda_a, H_log.data(), 1, C.data(), 1);
        auto time_C_done = std::chrono::high_resolution_clock::now();
        C_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_C_done - time_L_done).count()  ;

        // ---- C_old / W check ----
        if (!has_C_old) {
            cblas_dcopy(n1*n2, C.data(), 1, C_old.data(), 1);
            has_C_old = true;
        } 
        else {
            double W_old = cblas_ddot(n1*n2, S.data(), 1, C_old.data(), 1);
            double W     = cblas_ddot(n1*n2, S.data(), 1, C.data(), 1);
            if (W <= W_old) {
                cblas_dcopy(n1*n2, C.data(), 1, C_old.data(), 1);
            } else {
                cblas_dcopy(n1*n2, C_old.data(), 1, C.data(), 1);
            }
        }

        // ---- Cost = C - lambda_p * log(S) ----
        cblas_dcopy(n1*n2, S.data(), 1, S_log.data(), 1);
        vdLn(n1*n2, S_log.data(), S_log.data());
        cblas_dcopy(n1*n2, C.data(), 1, Cost.data(), 1);
        cblas_daxpy(n1*n2, -lambda_p, S_log.data(), 1, Cost.data(), 1);
        mkl_domatcopy('C', 'T', n1, n2, 1.0, Cost.data(), n1, Cost_trans.data(), n2);

        std::vector<double> a_row(n2); 
        std::vector<double> b_col(n1, 1.0 / n2); 
        auto time_iter_1 = std::chrono::high_resolution_clock::now();
        // std::cout << "Updating C before optimizing a and b with inner loop time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1-time_iter_0).count() << " ms" << std::endl;
        Cost_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1 - time_C_done).count()  ;
        // std::vector<double> Cost_copy(n1*n2);
        // cblas_dcopy(n1*n2, Cost.data(), 1, Cost_copy.data(), 1);
        // ---- Sinkhorn inner iterations ----
        for (int j = 0; j < inner_iters; ++j) {

            //Combined minaa and minbb
            /*
            auto t_0 = std::chrono::high_resolution_clock::now() ;
            minaa_minbb_v2(Cost, n1, n2, lambda_total, a_row, b_col);
            auto t_1 = std::chrono::high_resolution_clock::now() ;
            std::cout << "inner iteration " << j << " times : " << 
            std::chrono::duration_cast<std::chrono::milliseconds>(t_1 - t_0).count() << std::endl;
            */
        
            // printf(" \n ---- outer iter %d --- inner iter %d --- \n", it, j);
            // Cost - b  (b is n1 x 1, broadcast over columns)
            auto t_0 = std::chrono::high_resolution_clock::now() ;
            // if( j > 0 ) {
            //     // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);
            //     cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);
            // }
            auto t_1 = std::chrono::high_resolution_clock::now() ;
            // buffer = Cost - B_mat
            // mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, Cost.data(), n1, -1.0, B_mat.data(), n1, buffer.data(), n1);
            auto t_2 = std::chrono::high_resolution_clock::now() ;
            before_minaa += std::chrono::duration_cast<std::chrono::milliseconds>(t_2 - t_0).count()  ;


            // a_row = minaa(buffer, n1, n2, lambda_total);
            // a_row = minaa_v2(buffer, n1, n2, lambda_total);
            a_row = minaa_v3_dense(Cost, b_col, n1, n2, lambda_total);
            auto t_3 = std::chrono::high_resolution_clock::now() ;
            compute_minaa += std::chrono::duration_cast<std::chrono::milliseconds>(t_3 - t_2).count()  ;
            if(j == 1)   std::cout << "one minaa v2 " << std::chrono::duration_cast<std::chrono::milliseconds>(t_3 - t_2).count()   << std::endl;

            /////  old version minaa /////
            // auto t_minaa_v2_st = std::chrono::high_resolution_clock::now() ;
            // std::vector<double> a_row_old = minaa(buffer, n1, n2, lambda_total);
            // auto t_minaa_v2_end = std::chrono::high_resolution_clock::now() ;
            // std::cout << "one minaa " << std::chrono::duration_cast<std::chrono::milliseconds>(t_minaa_v2_end - t_minaa_v2_st).count() << std::endl;
            

            /*
            // Cost - a  (a is 1 x n2, broadcast over rows)
            // A_mat = 1_n1 * a_row^T (n1 x n2)
            // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, a_row.data(), 1, 0.0, A_mat.data(), n1);
            cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, a_row.data(), 1, 0.0, A_mat.data(), n1); 
            auto t_4 = std::chrono::high_resolution_clock::now() ;
            // buffer = Cost - A_mat
            mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, Cost.data(), n1, -1.0, A_mat.data(), n1, buffer.data(), n1);
            auto t_5 = std::chrono::high_resolution_clock::now() ;
            between_minaa_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_5 - t_3).count();
            b_col = minbb_v2(buffer, n1, n2, lambda_total);
            auto t_6 = std::chrono::high_resolution_clock::now() ;
            compute_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_6 - t_5).count();
            if(j == 1)   std::cout << "one minbb " << std::chrono::duration_cast<std::chrono::milliseconds>(t_6 - t_5).count() << std::endl;
            */
            

            auto t_4 = std::chrono::high_resolution_clock::now() ;
            auto t_5 = std::chrono::high_resolution_clock::now() ;
            between_minaa_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_5 - t_3).count()  ;
            b_col = minaa_v3_dense(Cost_trans, a_row, n2, n1, lambda_total);
            auto t_6 = std::chrono::high_resolution_clock::now() ;
            compute_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_6 - t_5).count()  ;
            // // old version minbb //
            // b_col = minbb(buffer, n1, n2, lambda_total);


            // //trranspose 
            // auto t_trans_0 = std::chrono::high_resolution_clock::now() ;
            // std::vector<double> buffer_trans(n1 * n2);
            // auto t_trans_1 = std::chrono::high_resolution_clock::now() ;
            // mkl_domatcopy('C', 'T', n1, n2, 1.0, buffer.data(), n1, buffer_trans.data(), n2);
            // auto t_trans_2 = std::chrono::high_resolution_clock::now() ;
            // std::vector<double> b_col_trans = minaa_v2(buffer_trans, n2, n1, lambda_total);
            // auto t_trans_3 = std::chrono::high_resolution_clock::now() ;
            // std::cout << "minbb with transpose and minaa total " << std::chrono::duration_cast<std::chrono::milliseconds>(t_trans_3 - t_trans_0).count() <<
            // " init mat " << std::chrono::duration_cast<std::chrono::milliseconds>(t_trans_1 - t_trans_0).count() <<
            // " transpose " << std::chrono::duration_cast<std::chrono::milliseconds>(t_trans_2 - t_trans_1).count() <<
            // " minaa " << std::chrono::duration_cast<std::chrono::milliseconds>(t_trans_3 - t_trans_2).count() <<
            // std::endl;

            
            // printf("size b_col_trans %d\n", b_col_trans.size());
            // for(int i=0; i< b_col.size(); i++) {
            //     if( std::abs(b_col[i] - b_col_trans[i]) > 1e-3)
            //         printf("error on minbb with transpose and minaa computation index %d old %e new %e\n", i, b_col[i], b_col_trans[i]);
            // }

            // for(int i=0; i< b_col_v2.size(); i++) {
            //     if( std::abs(b_col[i] - b_col_v2[i]) > 1e-3)
            //         printf("error on minbb computation index %d old %e new %e\n", i, b_col[i], b_col_v2[i]);
            // }
            // for(int i=0; i< a_row.size(); i++) {
            //     if( std::abs(a_row_old[i] - a_row[i]) > 1e-5)
            //         printf("error on minaa computation index %d old %e new %e\n", i, a_row_old[i], a_row[i]);
            // }
            // std::cout << "inner iteration " << j << " times : " 
            // << std::chrono::duration_cast<std::chrono::milliseconds>(t_1-t_0).count() << " " 
            // << std::chrono::duration_cast<std::chrono::milliseconds>(t_2-t_1).count() << " "
            // << std::chrono::duration_cast<std::chrono::milliseconds>(t_3-t_2).count() << " "
            // << std::chrono::duration_cast<std::chrono::milliseconds>(t_4-t_3).count() << " "
            // << std::chrono::duration_cast<std::chrono::milliseconds>(t_5-t_4).count() << " " 
            // << std::chrono::duration_cast<std::chrono::milliseconds>(t_6-t_5).count() << " " << std::endl;

        }
        auto time_iter_2 = std::chrono::high_resolution_clock::now();        
        // std::cout << "optimizing a and b with inner loop time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_2-time_iter_1).count() << " ms" << std::endl;
        inner_loop += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_2 - time_iter_1).count()  ;
  
        // ---- S = 0.05 * S_old + 0.95 * r * exp((a + b - Cost)/lambda_total) * c ----
        /*
        // A_mat = 1_n1 * a_row^T
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, a_row.data(), 1, 0.0, A_mat.data(), n1);
        // B_mat = b_col * 1_n2^T
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);
        // buffer = A_mat + B_mat
        mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, A_mat.data(), n1, 1.0, B_mat.data(), n1, buffer.data(), n1);

        // buffer = buffer - Cost
        mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, buffer.data(), n1, -1.0, Cost.data(), n1, buffer.data(), n1);
   
        // buffer = buffer / lambda_total   
        cblas_dscal(n1*n2, 1.0 / lambda_total, buffer.data(), 1);

        // buffer = exp(buffer)
        vdExp(n1*n2, buffer.data(), buffer.data());
        */
        // e^(a+b-Cost / lambda)
        // buffer = update_S_mat(buffer, Cost, a_row, b_col, n1, n2, lambda_total);

        /*
        // RC_mat = r * 1_n2^T
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, r.data(), n1, one_vec_n2.data(), 1, 0.0, RC_mat.data(), n1);
        // buffer = buffer ⊙ RC_mat
        vdMul(n1*n2, buffer.data(), RC_mat.data(), buffer.data());
        // C_mat = 1_n1 * c^T
        // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, c.data(), 1, 0.0, C_mat.data(), n1);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, c.data(), 1, 0.0, RC_mat.data(), n1);
        // buffer = buffer ⊙ C_mat
        // vdMul(n1*n2, buffer.data(), C_mat.data(), buffer.data());
        vdMul(n1*n2, buffer.data(), RC_mat.data(), buffer.data());

        // S = 0.95 * buffer + 0.05 * S_old
        cblas_dcopy(n1*n2, buffer.data(), 1, S.data(), 1);
        cblas_dscal(n1*n2, 0.95, S.data(), 1);
        cblas_daxpy(n1*n2, 0.05, S_old.data(), 1, S.data(), 1);
        */
        update_S_mat(S, S_old, Cost, a_row, b_col, n1, n2, lambda_total);
        auto time_iter_3 = std::chrono::high_resolution_clock::now();        
        // std::cout << "updating S after optimizing a and b time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_3-time_iter_2).count() << " ms" << std::endl;
        update_S += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_3 - time_iter_2).count()  ;
        
        //diff_S = sum |S - S_old|
        mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, S.data(), n1, -1.0, S_old.data(), n1, buffer.data(), n1);
        vdAbs(n1*n2, buffer.data(), diff_abs.data());
        double diff_S = cblas_dasum(n1*n2, diff_abs.data(), 1);
        printf("inner iter %d and diff of S %.6f\n\n", it, diff_S);


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
    std::cout << "total time of 10 outer iterations : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count()   << " ms" << std::endl;
    std::cout << " L_compute_time : " << L_compute_time <<  " C_compute_time : " << C_compute_time <<  " Cost_compute_time : " << Cost_compute_time <<  
    " before_mina : " << before_minaa << " mina : " << compute_minaa << " between_mina_minb : " << between_minaa_minbb << " minb : " << compute_minbb << " update_S : " << update_S << " ms" << std::endl;
    std::cout << "done con_prox_pt_opt_init" << std::endl;
    return S;
}   
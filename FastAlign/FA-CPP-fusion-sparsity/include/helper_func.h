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

std::vector<double> minbb_v2(const std::vector<double>& H, int n1, int n2, double epsilon) {
    // std::vector<double> col_min(n2);
    std::vector<double> res(n1);
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (int i = 0; i < n1; ++i) {
        double min_i;
        // if (i == 0) printf("threads = %d\n", omp_get_num_threads());
        const double* row = &H[(size_t) i];
        int idx = cblas_idamin(n2, row, n1);
        min_i = row[idx];

        double s_i = 0.0;
        for (int j = 0; j < n2; j++) {
            double shifted = H[(size_t)j * n1 + i] - min_i;
            s_i += std::exp(-(shifted / epsilon));
        }
        res[i] = min_i - epsilon * std::log(s_i / n2);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return res;
}


// H_in: n1 x n2, row-major
std::vector<double> mina(const std::vector<double>& H_in, int n1, int n2, double epsilon) {
    // const int N = n1 * n2;
    std::vector<double> H(n1 * n2);
    cblas_dcopy(n1 * n2, H_in.data(), 1, H.data(), 1);        // H = H_in
    
    // H = -H_in / epsilon
    double alpha = -1.0 / epsilon;
    cblas_dscal(n1 * n2, alpha, H.data(), 1);
    // H = exp(H)
    vdExp(n1 * n2, H.data(), H.data());
    // in_a = ones(n1) / n1
    std::vector<double> in_a(n1, 1.0 / static_cast<double>(n1));

    
    // s = H^T * in_a  (n2 vector): s_j = sum_i in_a_i * H_ij
    std::vector<double> s(n2);
    cblas_dgemv(CblasColMajor, CblasTrans, n1, n2, 1.0, H.data(), n1, in_a.data(), 1, 0.0, s.data(), 1);

    // s = -epsilon * log(s)
    vdLn(n2, s.data(), s.data());
    cblas_dscal(n2, -epsilon, s.data(), 1);
    
    return s;  // shape: (1 x n2)
}



std::vector<double> minb(const std::vector<double>& H_in, int n1, int n2, double epsilon) {
    
    std::vector<double> H(n1 * n2);
    cblas_dcopy(n1 * n2, H_in.data(), 1, H.data(), 1);        // H = H_in
    double alpha = -1.0 / epsilon;
    cblas_dscal(n1 * n2, alpha, H.data(), 1);
    vdExp(n1 * n2, H.data(), H.data());
    // in_b = ones(n2) / n2
    std::vector<double> in_b(n2, 1.0 / static_cast<double>(n2));

    // s = H * in_b  (n1 vector): s_i = sum_j in_b_j * H_ij
    std::vector<double> s(n1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n1, n2, 1.0, H.data(), n1,  in_b.data(), 1, 0.0, s.data(), 1);
    vdLn(n1, s.data(), s.data());
    cblas_dscal(n1, -epsilon, s.data(), 1);
   
    return s;  // shape: (n1 x 1)
}


std::vector<double> minaa_v2(const std::vector<double>& H, int n1, int n2, double epsilon) {
    // std::vector<double> col_min(n2);
    std::vector<double> res(n2);
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (int j = 0; j < n2; ++j) {
        // if (j == 0) printf("threads = %d\n", omp_get_num_threads());
        const double* col = &H[(size_t)j * n1];
        // int idx = cblas_idamin(n1, col, 1);
        // double min_j = col[idx];
        double min_j = col[0];
        for (int i = 1; i < n1; ++i)
            min_j = std::min(min_j, col[i]);
        // printf("min_%d %e \n", j, min_j);

        double s_j = 0.0;
        for (int i = 0; i < n1; i++) {
            double shifted = H[(size_t)j * n1 + i] - min_j;
            s_j += std::exp(-(shifted / epsilon));
        }
        res[j] = min_j - epsilon * std::log(s_j / n1);
        // printf("s_%d %e \n", j, s_j);
        // if(res2[j] - res[j] > 1e-6) printf(" res is not same mine %f - mina res %e\n", res2[j], res[j]);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return res;
}


//sparse version do row wise computation unlike dense which does col version
std::vector<double> minaa_v3( sparse_matrix_t &H, const std::vector<double>& a_row, int n1, int n2, double epsilon) {
    std::vector<double> res(n1);
    // Export CSR data
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start = nullptr, *row_end = nullptr;
    MKL_INT *col_indx = nullptr;
    double *values = nullptr;
    sparse_status_t st = mkl_sparse_d_export_csr( H, &indexing, &rows, &cols, &row_start, &row_end, &col_indx, &values);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("mkl_sparse_d_export_csr failed");

    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows; ++i) {
        double min_i;
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            values[idx] -= a_row[idx]-row_start[i];   //C-a
            if ( idx == row_start[i])   min_i = values[idx];
            min_i = std::min(min_i, values[i]);
        }
        double s_i = 0.0;
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            double shifted = values[idx] - min_i;
            s_i += std::exp(-(shifted / epsilon));
            values[idx] += a_row[idx]-row_start[i];   //C+a
        }
        res[i] = min_i - epsilon * std::log(s_i / n2);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return res;
}



void minaa_minbb_v2(const std::vector<double>& H, int n1, int n2, double epsilon, std::vector<double>& a, std::vector<double>& b) {
    std::vector<double> H_local = H;
    std::vector<double> H_transpose(n1*n2);
    mkl_domatcopy('C', 'T', n1, n2, 1.0, H.data(), n1, H_transpose.data(), n2); //transpose
    unsigned int numThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
    omp_set_num_threads(numThreads);
    #pragma omp parallel for 
    for (int j = 0; j < n2; ++j) {
        double min_j;
        // C = C-b and compute min
        double* col = &H_local[(size_t)j * n1];
        col[0] = col[0] - b[0];
        min_j = std::abs(col[0]);
        for (int i = 1; i < n1; i++) {
            col[i] = col[i] - b[i];
            if(std::abs(col[i]) < min_j)  min_j = col[i];
        }
        // printf("min_%d %e \n", j, min_j);

        double s_j = 0.0;
        for (int i = 0; i < n1; i++) {
            double shifted = col[i] - min_j;
            s_j += std::exp(-(shifted / epsilon));
        }
        a[j] = min_j - epsilon * std::log(s_j / n1);
        // printf("s_%d %e \n", j, s_j);
        // if(res2[j] - res[j] > 1e-6) printf(" res is not same mine %f - mina res %e\n", res2[j], res[j]);
    }
    #pragma omp parallel for 
    for (int j = 0; j < n1; ++j) {
        double min_j;
        // const double* col = &H_transpose[(size_t)j * n2];
        // int idx = cblas_idamin(n2, col, 1);
        // min_j = col[idx];
        // printf("min_%d %e \n", j, min_j);

        double* col = &H_transpose[(size_t)j * n2];
        col[0] = col[0] - a[0]; 
        min_j = std::abs(col[0]);
        for (int i = 1; i < n2; i++) {
            col[i] = col[i] - a[i];
            if(std::abs(col[i]) < min_j)  min_j = col[i];
        }

        double s_j = 0.0;
        for (int i = 0; i < n2; i++) {
            double shifted = col[i] - min_j;
            s_j += std::exp(-(shifted / epsilon));
        }
        b[j] = min_j - epsilon * std::log(s_j / n2);
        // printf("s_%d %e \n", j, s_j);
        // if(res2[j] - res[j] > 1e-6) printf(" res is not same mine %f - mina res %e\n", res2[j], res[j]);
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return;
}


// minaa(H) = mina(H - min(H,dim=0), eps) + min(H,dim=0)
std::vector<double> minaa(const std::vector<double>& H, int n1, int n2, double epsilon) {
    auto time_0 = std::chrono::high_resolution_clock::now();    
    std::vector<double> H_shifted(n1 * n2);
    std::vector<double> col_min(n2);
    std::vector<double> ones_n1(n1, 1.0);
    std::vector<double> M(n1 * n2);
    auto time_1 = std::chrono::high_resolution_clock::now();    
    for (int j = 0; j < n2; ++j) {
        const double* col = &H[(size_t)j * n1];
        // index of min |value| in column j
        int idx = cblas_idamin(n1, col, 1);
        // col_min_idx[j] = idx;   // 0-based index
        col_min[j] = col[idx];
    }
    auto time_2 = std::chrono::high_resolution_clock::now();    
    // build matrix M = 1_n1 * col_min^T (n1 x n2)
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, ones_n1.data(), n1, col_min.data(), 1, 0.0, M.data(), n1);
    // H_shifted = H - M
    mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, H.data(), n1, -1.0, M.data(), n1, H_shifted.data(), n1);
    auto time_3= std::chrono::high_resolution_clock::now();    
    // mina on H_shifted
    std::vector<double> res = mina(H_shifted, n1, n2, epsilon); // size n2
    auto time_4= std::chrono::high_resolution_clock::now();    
    // add column minima back: res += col_min
    cblas_daxpy(n2, 1.0, col_min.data(), 1, res.data(), 1);
    auto time_5= std::chrono::high_resolution_clock::now();    
    // std::cout << "in minaa init : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
    // " col_min compute : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
    // " compute H_shifted(Cs mat) : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << 
    // " mina : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() << 
    // " add row_min after minb : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_5-time_4).count() << 
    // " ms" << std::endl;

    return res;  // shape: (1 x n2)
}



// minbb(H) = minb(H - min(H,dim=1), eps) + min(H,dim=1)
std::vector<double> minbb(const std::vector<double>& H, int n1, int n2, double epsilon) {
    // const int N = n1 * n2;
    auto time_0 = std::chrono::high_resolution_clock::now();        
    std::vector<double> H_shifted(n1 * n2);
    std::vector<double> row_min(n1);
    // std::vector<double> ones_n2(n2, 1.0);
    // std::vector<double> M(n1 * n2);
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    auto time_1 = std::chrono::high_resolution_clock::now();    
    #pragma omp parallel for    
    for (int i = 0; i < n1; ++i) {
        const double* row = &H[i];   // start of row i
        // index of min |value| across columns  - cblas_idamin is absolute min - is all values in H positive?
        int idx = cblas_idamin(n2, row, n1);  // stride = n1
        // correct element
        row_min[i] = row[idx * n1];
    }
    // #pragma omp parallel for
    // for (int i = 0; i < n1; ++i) {
    //     const double* row = H + i;
    //     int idx = cblas_idamin(n2, row, n1);   // stride = n1
    //     row_min[i] = row[idx * n1];
    // }
    auto time_2 = std::chrono::high_resolution_clock::now();        
    // build matrix M = row_min * 1_n2^T  (n1 x n2)
    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, row_min.data(), n1, ones_n2.data(), 1, 0.0, M.data(), n1);

    // // H_shifted = H - M
    // // mkl_domatadd('R', 'N', 'N', n1, n2, 1.0, H.data(), n2, -1.0, M.data(), n2, H_shifted.data(), n2);
    // mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, H.data(), n1, -1.0, M.data(), n1, H_shifted.data(), n1);
    #pragma omp parallel for
    for (int j = 0; j < n2; ++j) {
        if (j == 0) printf(" ---- minbb  --- threads = %d\n", omp_get_num_threads());
        const double* Hcol = H.data() + j * n1;
        double* Hscol = H_shifted.data() + j * n1;
        cblas_dcopy(n1, Hcol, 1, Hscol, 1);
        cblas_daxpy(n1, -1.0, row_min.data(), 1, Hscol, 1);
    }
    auto time_3 = std::chrono::high_resolution_clock::now(); 
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    // minb on shifted
    std::vector<double> res = minb(H_shifted, n1, n2, epsilon); // size n1
    auto time_4 = std::chrono::high_resolution_clock::now(); 
    // add row minima back: res += row_min
    cblas_daxpy(n1, 1.0, row_min.data(), 1, res.data(), 1);
    auto time_5 = std::chrono::high_resolution_clock::now(); 
    std::cout << "in minbb init : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   <<
    " row_min compute : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   <<
    " compute H_shifter(Cs mat) : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()   << 
    " minb : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count()   << 
    " add row_min after minb : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_5-time_4).count()   << 
    " ms" << std::endl;
    return res;  // shape: (n1 x 1)
}

std::vector<double> compute_L_fixed_dense( const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, const std::vector<double>& r, const std::vector<double>& c, 
        int n1, int n2) {
    // 1) Square intra_dist1 and intra_dist2 elementwise
    std::vector<double> A1(intra_dist1.size());
    std::vector<double> A2(intra_dist2.size());
    std::vector<double> one_vec_n1(n1, 1.0);
    std::vector<double> one_vec_n2(n2, 1.0);
    vdMul(n1 * n1, intra_dist1.data(), intra_dist1.data(), A1.data()); // A1 = intra_dist1^2
    vdMul(n2 * n2, intra_dist2.data(), intra_dist2.data(), A2.data()); // A2 = intra_dist2^2
    // First term: (intra_dist1^2) @ r @ one_vec_n2^T
    // v1 = A1 * r   (n1 x 1)
    std::vector<double> v1(n1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n1, n1, 1.0, A1.data(), n1, r.data(), 1, 0.0, v1.data(), 1);
    // term1 = v1 * one_vec_n2^T  (outer product, n1 x n2)
    std::vector<double> L_fixed(n1 * n2);
    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, 1, 1.0, v1.data(), n1, one_vec_n2.data(), n2, 0.0, term1.data(), n1);         // n1 x n2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, v1.data(), n1, one_vec_n2.data(), 1, 0.0, L_fixed.data(), n1);         // n1 x n2
    // Second term: one_vec_n1 @ c @ (intra_dist2^2)^T
    // B0 = one_vec_n1 * c  (n1 x n2)
    std::vector<double> B0(n1 * n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, c.data(), 1, 0.0, B0.data(), n1);            // n1 x n2
    // term2 = B0 @ (A2)^T   where A2 = intra_dist2^2, size n2 x n2
    std::vector<double> term2(n1 * n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, n2, 1.0, B0.data(), n1, A2.data(), n2, 1.0, L_fixed.data(), n1);         // n1 x n2
    // 4) L_fixed = term1 + term2
    // std::vector<double> L_fixed = term1;
    // cblas_daxpy(n1 * n2, 1.0, term2.data(), 1, L_fixed.data(), 1);

    // for(int i=0; i<100; i++) { printf(" %.6f ", L_fixed[i]);}
    return L_fixed; // n1 x  n2, col-major
}

void compute_RW_term_dense( std::vector<double> dense_adj1, std::vector<double> dense_adj2, const std::vector<double>& S, int n1, int n2, std::vector<double>& RW_term) {
    std::vector<double> buffer(n1*n2);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n1, n2, n1, 1.0, dense_adj1.data(), n1, S.data(), n1, 0.0, buffer.data(), n1);         //col major
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, n2, 1.0, buffer.data(), n1, dense_adj2.data(), n2, 0.0, RW_term.data(), n1);         //col major
}

void compute_A2T_term_dense( std::vector<double> intra_dist1, std::vector<double> intra_dist2, const std::vector<double>& S, int n1, int n2, std::vector<double>& A2T_term) {
    std::vector<double> buffer(n1*n2);
    MKL_INT scalar = -2.0 ;
    // A_S = intra_dist1 @ S  
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, n1, 1.0, intra_dist1.data(), n1, S.data(), n1, 0.0, buffer.data(), n1);         //col major
    // A2T_term = A_S @ intra_dist2^T 
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, n2, (double) -2 , buffer.data(), n1, intra_dist2.data(), n2, 0.0, A2T_term.data(), n1);           //col major
}

std::vector<double> update_S_mat2 (std::vector<double>& S, const std::vector<double>& Cost, const std::vector<double>& a_row, const std::vector<double>& b_col, int n1, int n2, double epsilon) {
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for 
    for (int j = 0; j < n2; ++j) {
        double a_val = a_row[j];
        for (int i = 0; i < n1; ++i) {
            S[j * n1 + i] = std::exp(( a_val + b_col[i] - Cost[j * n1 + i]) / epsilon);
        } 
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return S;
}

std::vector<double>  get_batch_rwr_scores_dense(sparse_matrix_t graph, std::vector<MKL_INT> anchors, MKL_INT n,  float restart_prob, int max_iters, bool connect_isolated) {
    //connect isolated nodes and row normalize A  
    std::vector<double> all_ones(n, (double) 1.0);
    std::vector<double> graph_row_sum(n, (double) 0.0);
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    auto time_0 = std::chrono::high_resolution_clock::now();
    
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
        edge_list.reserve(isolated.size() * n * 2);
        for (MKL_INT i : isolated) {
            for (MKL_INT j = 0; j < n; ++j) {
                edge_list.push_back(static_cast<double>(i));
                edge_list.push_back(static_cast<double>(j));
            }
            // update row sum directly
            graph_row_sum[i] = static_cast<double>(n);
        }
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
    auto time_1 = std::chrono::high_resolution_clock::now();

    MKLCSC Dinv = make_diag_csc(graph_row_sum, n, n, true);
    sparse_matrix_t graph_norm = nullptr;    
    mkl_sparse_optimize(Dinv.spMat);
    mkl_sparse_optimize(graph_connected);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph_connected, &graph_norm);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_spmm failed in get_batch_rwr_scores");
    mkl_sparse_optimize(graph_norm);
    sparse_matrix_t graph_norm_csr = nullptr;
    status = mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_convert_csr failed in get_batch_rwr_scores");
    status = mkl_sparse_optimize(graph_norm_csr);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_optimize failed in get_batch_rwr_scores");  
    int l = anchors.size();
    auto time_2 = std::chrono::high_resolution_clock::now();
    
    //one hot encoding E matrix
    std::vector<double> E(n * l, 0.0);
    for (int j = 0; j < l; ++j) {
        E[ anchors[j] + j * n ] = 1.0;   // one-hot at (row=anchor[j], col=j)
    }
    //these are column major
    std::vector<double> X_vals = E;
    std::vector<double> Xnext_vals(n*l, 0.0);
    std::vector<double> graph_norm_csr_dense = sparse_to_dense_colmajor(graph_norm_csr, n, n);
    auto time_3 = std::chrono::high_resolution_clock::now();
    for(int i=0; i < max_iters-1; i++) {
        Xnext_vals = E;
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, l, n,(double) (1-restart_prob), graph_norm_csr_dense.data(), n, X_vals.data(), n, restart_prob, Xnext_vals.data(), n);
        X_vals = Xnext_vals;
    }
    auto time_4= std::chrono::high_resolution_clock::now();
    std::cout << "connect isolated : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()  
    << " normalize : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()  
    << " rwr before iterations: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()  
    << " rwr iteration=100: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count()  
    << " ms" << std::endl;

    //column major dense matrix nxl
	return Xnext_vals;
}

std::vector<double> get_product_rwr_mat_dense( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, int n1, int n2, 
    int avg_nnz1_per_n1, int avg_nnz2_per_n2,
    double rwr_restart_prob, double gamma, int max_iters = 50, double tol = 1e-2) {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_matrix_t graph1_normalized = row_normalize_graph( graph1_py, n1, avg_nnz1_per_n1);
    sparse_matrix_t graph2_normalized = row_normalize_graph( graph2_py, n2, avg_nnz2_per_n2);
    std::vector<double> graph1_norm_dense = sparse_to_dense_colmajor(graph1_normalized, n1, n1);
    std::vector<double> graph2_norm_dense = sparse_to_dense_colmajor(graph2_normalized, n2, n2);

    std::vector<double> out_cross_dist(n1 * n2, 0.0);
    std::vector<double> out_cross_dist_old(n1 * n2, 0.0);
    std::vector<double> temp(n1 * n2);
    std::vector<double> temp3(n1 * n2);
    std::vector<double> diff(n1 * n2);
    std::vector<double> diff_abs(n1 * n2);
    for (int i = 0; i < max_iters; ++i) {
        out_cross_dist_old = out_cross_dist;
        //‌ *** Version two with dgemm **** //
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, n1, 1.0, graph1_norm_dense.data(), n1, out_cross_dist.data(), n1, 0.0, temp.data(), n1);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, n2, 1.0, temp.data(), n1, graph2_norm_dense.data(), n2, 0.0, temp3.data(), n1);
        mkl_domatadd('C', 'N', 'N', n1, n2, (1+rwr_restart_prob), in_cross_dist.data(), n1, ((1 - rwr_restart_prob) * gamma), temp3.data(), n1, out_cross_dist.data(), n1);
        //‌ *** end of Version two with dgemm **** //
        //compute max of absolute values of (out_cross_dist - out_cross_dist_old)
        vdSub(n1 * n2, out_cross_dist_old.data(), out_cross_dist.data(), diff.data());   
        vdAbs(n1*n2, diff.data(), diff_abs.data());
        double max_diff = *std::max_element(diff_abs.begin(), diff_abs.end());
        if (max_diff < tol) break;
    }
    //out_cross_dist = (1 - self.gamma) * out_cross_dist
    cblas_dscal(n1 * n2, 1.0 - gamma, out_cross_dist.data(), 1);
    return out_cross_dist;
}

sparse_matrix_t make_coo_from_vec(const std::vector<double>& vec, int m, int n) {
	MKL_INT nnz = 0;
	for(size_t i=0; i<vec.size(); i++) {
		if(vec[i] != 0)
			nnz++;
	}
	std::vector<MKL_INT> rows(nnz), cols(nnz);
	std::vector<double> vals(nnz);
	int k = 0;
	for(size_t i=0; i<vec.size(); i++) {
        if(vec[i] != 0){
			vals[k] = vec[i];
			rows[k] = i % m;
			cols[k] = (MKL_INT) (i / m);	
			k++;
		}	
    }
	sparse_matrix_t A = nullptr;
    auto st = mkl_sparse_d_create_coo(&A, SPARSE_INDEX_BASE_ZERO, m, n, nnz, rows.data(), cols.data(), vals.data());
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("create_coo failed");
    mkl_sparse_optimize(A);
	return A;
}

MKLCSC make_csc_from_vec(const std::vector<double>& vec, MKL_INT m, MKL_INT n) {
	MKL_INT nnz = 0;
	for(size_t i=0; i<vec.size(); i++) {
		if(vec[i] != 0)
			nnz++;
	}
    
    MKLCSC A;    
    A.col_start.resize(n);             
    A.col_end.resize(n);             
    A.row_ind.resize(nnz);             
    A.vals.resize(nnz);             

	MKL_INT k = 0;
	for (MKL_INT j = 0; j < n; ++j) {
        A.col_start[j] =  k;
        for (MKL_INT i = 0; i < m; ++i) {
            if (vec[(size_t)j * m + i] != 0) {
                A.row_ind[k] =  i;
                A.vals[k]    = (double) vec[(size_t)j * m + i];
                ++k;
            }
        }
        A.col_end[j] = k;
    }

    auto st = mkl_sparse_d_create_csc(&A.spMat, SPARSE_INDEX_BASE_ZERO, m, n, A.col_start.data(), A.col_end.data(), A.row_ind.data(), A.vals.data());
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_create_csc failed");
    mkl_sparse_optimize(A.spMat);
	return A;
}

sparse_matrix_t mask_dense_with_sparse( const std::vector<double>& dense_mat, sparse_matrix_t mask) {
    sparse_matrix_t masked_mat;
    matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto st = mkl_sparse_copy(mask, descr, &masked_mat);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("Copy failed");

    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start, *row_end, *col_ind;
    double *mask_vals;
    st = mkl_sparse_d_export_csr( masked_mat, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, &mask_vals);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("Export CSR failed");

    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT idx = row_start[i]; idx < row_end[i]; ++idx) {
            MKL_INT j = col_ind[idx];
            mask_vals[idx] = dense_mat[j * rows + i];
        }
    }
    mkl_set_num_threads(numThreads);
    omp_set_num_threads(1);
    return masked_mat;
}
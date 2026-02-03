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

void print_sparse_matrix_info(sparse_matrix_t A, const char* name = "matrix") {
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start, *row_end, *col_indx;
    double *values;

    sparse_status_t status = mkl_sparse_d_export_csr(
        A, &indexing, &rows, &cols,
        &row_start, &row_end, &col_indx, &values
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error: cannot export %s (status=%d)\n", name, status);
        return;
    }

    printf("=== %s ===\n", name);
    printf("Dimensions: %" PRId64 " x %" PRId64 "\n", (int64_t)rows, (int64_t)cols);
    printf("Index base: %s\n", (indexing == SPARSE_INDEX_BASE_ZERO) ? "0-based" : "1-based");

    MKL_INT nnz = row_end[rows - 1] - row_start[0];
    printf("Nonzeros: %" PRId64 "\n", (int64_t)nnz);

    printf("Sample entries (row, col, val):\n");
    MKL_INT printed = 0;
    for (MKL_INT i = 0; i < rows && printed < 10; ++i) {
        for (MKL_INT k = row_start[i]; k < row_end[i] && printed < 10; ++k) {
            printf("(%lld, %lld, %.3f)\n",
                   (long long)i,
                   (long long)col_indx[k],
                   values[k]);
            printed++;
        }
    }
    printf("====================\n");
}

void print_num_zeros(std::vector<double>  M, const char* name ) {
    int counter = 0;
    for(int i=0; i< M.size(); i++) {
        if(M[i] == (double) 0.0)    counter++;
    }
    printf(" %s has %d zeros out of %d values\n", name, counter, M.size());
    return ;
}

std::vector<double> readFileToVector(const std::string &filename) {
    std::ifstream fin(filename);
    std::vector<double> data;
    double val;
    while (fin >> val) {
        data.push_back(val);
    }
    return data;
}

sparse_matrix_t make_mat_from_edgelist(const std::vector<double>& edgelist, int m, int n) {
    MKL_INT nnz  = edgelist.size() / 2;
	std::vector<MKL_INT> rows(nnz), cols(nnz);
	std::vector<double> vals(nnz);
	for(int i=0; i < nnz; i++) {
        vals[i] = 1.0;
        rows[i] = (MKL_INT) edgelist[2*i];
        cols[i] = (MKL_INT) edgelist[2*i+1];	
    }

	sparse_matrix_t A = nullptr;
    auto st = mkl_sparse_d_create_coo(&A, SPARSE_INDEX_BASE_ZERO, m, n, nnz, rows.data(), cols.data(), vals.data());
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("create_coo failed");
    mkl_sparse_optimize(A);

    sparse_matrix_t A_csr;
    mkl_sparse_convert_csr(A, SPARSE_OPERATION_NON_TRANSPOSE, &A_csr);
    
    mkl_sparse_destroy(A);

    return A_csr;
}

void compare( const std::vector<double>& cpp_vec, const std::string& filename, const std::string& label) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return;
    }

    // read python values
    std::vector<double> py_vec(cpp_vec.size());
    for (size_t i = 0; i < py_vec.size() && fin; ++i) {
        fin >> py_vec[i];
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    double a_err = 0.0, b_err = 0.0;
    int counter = 0;
    int counter_1 = 0;
    for (size_t i = 0; i < cpp_vec.size(); ++i) {
        double a = py_vec[i];
        double b = cpp_vec[i];
        double da = std::abs(a - b);

        if (da > max_abs) {
            max_abs = da;
            a_err = a;
            b_err = b;
        }

        double denom = std::max({std::abs(a), std::abs(b), 1e-12});
        // if (da > (denom * 0.1)) std::cout << label << " more than 10 percent error-> cpp: " << b_err << "  python: " << a_err << "\n";
        if (da > (denom * 0.1)) counter++;
        if (da > (denom * 0.01)) counter_1++;
        double rel = da / denom;
        if (rel > max_rel) max_rel = rel;
    
    }
    std::cout << label << " out of " << cpp_vec.size() << " elements " << counter << " elements has more than 10 percent error" << std::endl;
    std::cout << label << " out of " << cpp_vec.size() << " elements " << counter_1 << " elements has more than 1 percent error" << std::endl;
    std::cout << label << " max_abs = " << max_abs << "  max_rel = " << max_rel
              << "  cpp: " << b_err << "  python: " << a_err << "\n\n";
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

std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<double>& H_vec) {
    std::vector<MKL_INT> anchor_g1;
    std::vector<MKL_INT> anchor_g2;
    for(size_t i=0; i< H_vec.size()/2; i++) {
        anchor_g1.push_back((MKL_INT) H_vec[i*2]);
        anchor_g2.push_back((MKL_INT) H_vec[i*2+1]);
    }
    return {anchor_g1, anchor_g2};
}

std::vector<double>  get_batch_rwr_scores(sparse_matrix_t graph, std::vector<MKL_INT> anchors, MKL_INT n,  float restart_prob, int max_iters, bool connect_isolated) {
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
    // std::vector<double> X_vals(n*l, 1.0);      // X(t), all ones
    std::vector<double> X_vals = E;
    std::vector<double> Xnext_vals(n*l, 0.0);
    std::vector<double> graph_norm_csr_dense = sparse_to_dense_colmajor(graph_norm_csr, n, n);
    auto time_3 = std::chrono::high_resolution_clock::now();
    // for(int i=0; i < max_iters; i++) {
    for(int i=0; i < max_iters-1; i++) {
        // status = mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, (double) (1-restart_prob), graph_norm_csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X_vals.data(), l, n, 0.0, Xnext_vals.data(), n);
        // if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("error spmm");
        // cblas_daxpy(n * l, restart_prob, E.data(), 1, Xnext_vals.data(), 1);
        // std::swap(X_vals, Xnext_vals);
        Xnext_vals = E;
        // status = mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, (double) (1-restart_prob), graph_norm_csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X_vals.data(), l, n, restart_prob, Xnext_vals.data(), n);
        // if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("error spmm");
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


void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchors, double val) {
    // printf("intpu vec size %d  anchors pair size %d\n ", A.size(), anchors.first.size());
    const MKL_INT k = (MKL_INT) anchors.first.size();
    std::vector<MKL_INT> idx(k);
    for (MKL_INT t = 0; t < k; ++t) {
        if (anchors.first[t] < 0 || anchors.first[t] >= nrows || anchors.second[t] < 0 || anchors.second[t] >= ncols) {
            printf( "error anchor_cost_zero: out-of-range anchor (%d,%d)\n", anchors.first[t], anchors.second[t]);
        }
        idx[t] = anchors.first[t] + anchors.second[t] * nrows;   //if A stored column wise 
    }
    // Build a vector of values to scatter in the matrix
    std::vector<double> z(k, val);
    // A[idx[t]] = val 
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


sparse_matrix_t row_normalize_graph( sparse_matrix_t graph, int n) {
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
        // mkl_sparse_optimize(graph_connected);
        mkl_sparse_destroy(add_mat);

        // //update graph_row_sum
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph_connected, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    }
    else {
        graph_connected = graph;
    }
    
    MKLCSC Dinv = make_diag_csc(graph_row_sum, n, n, true);
    sparse_matrix_t graph_norm = nullptr;
    mkl_sparse_optimize(Dinv.spMat);
    mkl_sparse_optimize(graph_connected);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph_connected, &graph_norm);
    if (status != SPARSE_STATUS_SUCCESS) { std::cerr << "error : mkl_sparse_spmm failed" << std::endl; }
    sparse_matrix_t graph_norm_csr;
    mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    
    return graph_norm_csr;
}

std::vector<double> get_product_rwr_mat( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, int n1, int n2, double rwr_restart_prob, 
    double gamma, int max_iters = 50, double tol = 1e-2) {
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_matrix_t graph1_normalized = row_normalize_graph( graph1_py, n1);
    sparse_matrix_t graph2_normalized = row_normalize_graph( graph2_py, n2);
    std::vector<double> graph1_norm_dense = sparse_to_dense_colmajor(graph1_normalized, n1, n1);
    std::vector<double> graph2_norm_dense = sparse_to_dense_colmajor(graph2_normalized, n2, n2);

    std::vector<double> out_cross_dist(n1 * n2, 0.0);
    std::vector<double> out_cross_dist_old(n1 * n2, 0.0);
    std::vector<double> temp(n1 * n2);
    std::vector<double> temp2(n1 * n2);
    std::vector<double> temp3(n1 * n2);
    std::vector<double> diff(n1 * n2);
    std::vector<double> diff_abs(n1 * n2);
    for (int i = 0; i < max_iters; ++i) {
        out_cross_dist_old = out_cross_dist;
        //second term computation
        //out_cross_dist is column major too

        //‌ *** Version one with spmm **** //
        //temp is n1xn2 temp = G1.cross_dist
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        // double coeff = (1 - rwr_restart_prob) * gamma;
        // //we need temp2 = temp^T
        // mkl_domatcopy('C', 'T', n1, n2, 1.0, temp.data(), n1, temp2.data(), n2);
        // //temp3 is n2xn1  temp3 = G2.(temp^T) = G2.(temp2)
        // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, temp2.data(), n1, n2, 0.0, temp3.data(), n2);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        // //adding with scaling first element and transposing second element
        // //out_cross_dist:  n1 x n2
        // mkl_domatadd('C', 'N', 'T', n1, n2, (1+rwr_restart_prob), in_cross_dist.data(), n1, 1.0, temp3.data(), n2, out_cross_dist.data(), n1);

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
auto has_nan = [](const std::vector<double>& v){
    for (double x : v) if (std::isnan(x)) return true;
    return false;
};

bool has_nan_or_inf(const std::vector<double>& v) {
    for (double x : v)
        if (std::isnan(x) || std::isinf(x)) return true;
    return false;
}

double getMax(const std::vector<double>& v) {
    double max = v[0];
    for(size_t i=1; i<v.size(); i++) {
        if (v[i] > max )     max = v[i];
    }
    return max;
}

double getMin(const std::vector<double>& v) {
    double min = v[0];
    for(size_t i=1; i<v.size(); i++) {
        if ( v[i] < min ) min = v[i];
    }
    return min;
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



std::vector<double> minb(const std::vector<double>& H_in, int n1, int n2, double epsilon, int it) {
    
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
std::vector<double> minbb(const std::vector<double>& H, int n1, int n2, double epsilon, int it) {
    // const int N = n1 * n2;
    auto time_0 = std::chrono::high_resolution_clock::now();        
    std::vector<double> H_shifted(n1 * n2);
    std::vector<double> row_min(n1);
    // std::vector<double> ones_n2(n2, 1.0);
    // std::vector<double> M(n1 * n2);

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
        const double* Hcol = H.data() + j * n1;
        double* Hscol = H_shifted.data() + j * n1;
        cblas_dcopy(n1, Hcol, 1, Hscol, 1);
        cblas_daxpy(n1, -1.0, row_min.data(), 1, Hscol, 1);
    }
    auto time_3 = std::chrono::high_resolution_clock::now(); 
    // minb on shifted
    std::vector<double> res = minb(H_shifted, n1, n2, epsilon, it); // size n1
    auto time_4 = std::chrono::high_resolution_clock::now(); 
    // add row minima back: res += row_min
    cblas_daxpy(n1, 1.0, row_min.data(), 1, res.data(), 1);
    auto time_5 = std::chrono::high_resolution_clock::now(); 
    std::cout << "in minbb init : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() <<
    " row_min compute : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() <<
    " compute H_shifter(Cs mat) : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << 
    " minb : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() << 
    " add row_min after minb : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_5-time_4).count() << 
    " ms" << std::endl;
    return res;  // shape: (n1 x 1)
}


std::vector<double> compute_L_fixed( const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, const std::vector<double>& r, const std::vector<double>& c, 
    const std::vector<double>& one_vec_n1, const std::vector<double>& one_vec_n2, int n1, int n2) {
    // 1) Square intra_dist1 and intra_dist2 elementwise
    std::vector<double> A1(intra_dist1.size());
    std::vector<double> A2(intra_dist2.size());
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

std::vector<double> con_prox_pt_opt( const std::string data_path, sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, int n1, int n2,
    std::vector<double> cross_dist, std::vector<double> intra_dist1, std::vector<double> intra_dist2, int inner_iters, int outer_iters,
    double lambda_n, double lambda_a, double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs, 
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, std::vector<double> anchors_test) {

    print_num_zeros(intra_dist1, "intra_dist1");
    print_num_zeros(intra_dist2, "intra_dist2");
    print_num_zeros(cross_dist, "cross_dist");
    
    auto time_0 = std::chrono::high_resolution_clock::now();
    double lambda_total = lambda_n + lambda_a + lambda_p;
    double lambda_e = lambda_e_in * n1 * n2;
    std::vector<double> one_vec_n1(n1, 1.0);
    std::vector<double> one_vec_n2(n2, 1.0);
    std::vector<double> a  = one_vec_n1;
    std::vector<double> r  = one_vec_n1;
    cblas_dscal(n1, 1.0 / static_cast<double>(n1), a.data(), 1);
    cblas_dscal(n1, 1.0 / static_cast<double>(n1), r.data(), 1);
    std::vector<double> b  = one_vec_n2;
    std::vector<double> c  = one_vec_n2;
    cblas_dscal(n2, 1.0 / static_cast<double>(n2), b.data(), 1);
    cblas_dscal(n2, 1.0 / static_cast<double>(n2), c.data(), 1);
    std::vector<double> S(n1 * n2, 1.0 / static_cast<double>(n1 * n2));      // S = (1/(n1*n2)) * 1_n1 * 1_n2^T
    std::vector<double> buffer(n1*n2);
    std::vector<double> S_old(n1*n2);
    std::vector<double> A2T_term(n1*n2);
    std::vector<double> L(n1*n2);
    std::vector<double> C(n1*n2);
    std::vector<double> C_old(n1*n2);
    std::vector<double> Cost(n1*n2);
    std::vector<double> S_log(n1*n2);
    std::vector<double> diff_abs(n1*n2);
    std::vector<double> A_mat(n1*n2), B_mat(n1*n2, 1.0 / n2);
    std::vector<double> RC_mat(n1*n2);
    std::vector<double> A1T_S(n1 * n2);      // n1 x n2
    std::vector<double> RW_term(n1*n2);
    std::vector<double> H_log(n1 * n2, 1e-6);
    init_H_with_anchors(H_log, n1, n2, anchor_pairs);
    vdLn(n1 * n2, H_log.data(), H_log.data());
    bool has_C_old = false;

    auto time_1 = std::chrono::high_resolution_clock::now();
    std::cout << "initialization in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() << " ms" << std::endl;

    sparse_matrix_t row_norm_adj1 = row_normalize_graph(graph1_py, n1);
    sparse_matrix_t row_norm_adj2 = row_normalize_graph(graph2_py, n2);
    //Sparse adj to dense
    std::vector<double> dense_adj1 = sparse_to_dense_colmajor(row_norm_adj1, n1, n1);
    std::vector<double> dense_adj2 = sparse_to_dense_colmajor(row_norm_adj2, n2, n2);

    auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "row normalizing graphs in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << " ms" << std::endl;
    // std::vector<double> adj1_dense = sparse_to_dense_colmajor(row_norm_adj1, n1, n1);
    // std::vector<double> adj2_dense = sparse_to_dense_colmajor(row_norm_adj2, n2, n2);
    // compare(adj1_dense, data_path + "/row_norm_adj1.txt", "row_norm_adj1");
    // compare(adj2_dense, data_path + "/row_norm_adj2.txt", "row_norm_adj2");
    std::vector<double> L_fixed = compute_L_fixed(intra_dist1, intra_dist2, r, c, one_vec_n1, one_vec_n2, n1, n2);
    auto time_3 = std::chrono::high_resolution_clock::now();
    std::cout << "L_fixed computation in optimization : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count() << " ms" << std::endl;
    // compare(L_fixed, data_path + "/L_fixed.txt.txt", "L_fixed");

    auto L_compute_time=0, C_compute_time=0, Cost_compute_time=0, before_minaa=0, compute_minaa=0, between_minaa_minbb=0, compute_minbb=0, update_S=0, inner_loop=0;

    // outer_iters = 2;
    for (int it = 0; it < outer_iters; ++it) {
        printf("----- outer iterration %d ----\n", it);
        auto time_iter_0 = std::chrono::high_resolution_clock::now();
        cblas_dcopy(n1 * n2, S.data(), 1, S_old.data(), 1);
        auto dgemm_0 = std::chrono::high_resolution_clock::now();
        // A_S = intra_dist1 @ S   (n1 x n1) @ (n1 x n2) = (n1 x n2)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, n1, 1.0, intra_dist1.data(), n1, S.data(), n1, 0.0, buffer.data(), n1);         //col major
        // A2T_term = A_S @ intra_dist2^T   (n1 x n2) @ (n2 x n2) = (n1 x n2)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, n2, 1.0, buffer.data(), n1, intra_dist2.data(), n2, 0.0, A2T_term.data(), n1);           //col major
        auto dgemm_1 = std::chrono::high_resolution_clock::now();
        std::cout << "2dgemm dist1*S*dist2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(dgemm_1-dgemm_0).count() << " ms" << std::endl;
        // L = L_fixed - 2 * A2T_term
        cblas_dcopy(n1 * n2, L_fixed.data(), 1, L.data(), 1);
        cblas_daxpy(n1 * n2, -2.0, A2T_term.data(), 1, L.data(), 1);
        auto time_L_done = std::chrono::high_resolution_clock::now();
        L_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_L_done - time_iter_0).count();
        // compare(L, data_path + "/L_outer_iter_" + std::to_string(it) + ".txt", "L_iter_"+ std::to_string(it));
        // ---- C = cross_dist + lambda_e * L - lambda_n * log(row_norm_adj1.T @ S @ row_norm_adj2) - lambda_a * log(H) ----
        // Start from cross_dist
        cblas_dcopy(n1 * n2, cross_dist.data(), 1, C.data(), 1);

        // C += lambda_e * L
        cblas_daxpy(n1 * n2, lambda_e, L.data(), 1, C.data(), 1);

        // Compute RW_term = row_norm_adj1.T @ S @ row_norm_adj2 (n1 x n2)
        // A1T_S = row_norm_adj1.T @ S
        // op(A) = TRANSPOSE, A: n1 x n1, S: n1 x n2 -> A1T_S: n1 x n2
        // matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        // auto spmm_0 = std::chrono::high_resolution_clock::now();
        // sparse_status_t st = mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, A1T_S.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm (A1^T S) failed");
        // auto spmm_0_1 = std::chrono::high_resolution_clock::now();
        // mkl_sparse_optimize(row_norm_adj2);
        // auto spmm_0_2 = std::chrono::high_resolution_clock::now();
        // mkl_domatcopy('C', 'T', n1, n2, 1.0, A1T_S.data(), n1, buffer.data(), n2); //transpose
        // auto spmm_0_3 = std::chrono::high_resolution_clock::now();
        // std::vector<double> temp(n1*n2);
        // auto spmm_0_4 = std::chrono::high_resolution_clock::now();
        // mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,  1.0, row_norm_adj2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, buffer.data(), n1, n2, 0.0, temp.data(), n2);      //spmm
        // auto spmm_0_5 = std::chrono::high_resolution_clock::now();
        // mkl_domatcopy( 'C', 'T', n2, n1, 1.0, temp.data(), n2, RW_term.data(), n1);
        // auto spmm_1 = std::chrono::high_resolution_clock::now();
        // std::cout << "2spmm G1*S*G2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(spmm_1-spmm_0).count() << " ms" << std::endl;
        // std::cout << "2spmm G1*S*G2 ---- 1st spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(spmm_0_1-spmm_0).count() <<
        // " sparse optimize " << std::chrono::duration_cast<std::chrono::milliseconds>(spmm_0_2-spmm_0_1).count() << 
        // " mat copytrans " << std::chrono::duration_cast<std::chrono::milliseconds>(spmm_0_3-spmm_0_2).count() << 
        // " temp init " << std::chrono::duration_cast<std::chrono::milliseconds>(spmm_0_4-spmm_0_3).count() << 
        // " 2nd spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(spmm_0_5-spmm_0_4).count() << 
        // " mat copytrans " << std::chrono::duration_cast<std::chrono::milliseconds>(spmm_1-spmm_0_5).count() << " ms" << std::endl;



        //// ***** experiment spmm to dgemm ***** /////
        auto timing_1 = std::chrono::high_resolution_clock::now();
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n1, n2, n1, 1.0, dense_adj1.data(), n1, S.data(), n1, 0.0, buffer.data(), n1);         //col major
        auto timing_2 = std::chrono::high_resolution_clock::now();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, n2, 1.0, buffer.data(), n1, dense_adj2.data(), n2, 0.0, RW_term.data(), n1);         //col major
        auto timing_3 = std::chrono::high_resolution_clock::now();
        std::cout << "2dgemm for second 2matmul G1*S*G2 total ---- " <<  std::chrono::duration_cast<std::chrono::milliseconds>(timing_3-timing_1).count() <<
        " 1st dgemm " << std::chrono::duration_cast<std::chrono::milliseconds>(timing_2-timing_1).count() <<
        " 2nd dgemm " << std::chrono::duration_cast<std::chrono::milliseconds>(timing_3-timing_2).count() << " ms" << std::endl;
        // int err_cnt = 0, err_cnt2=0;
        // for(int i=0; i<n1*n2; i++) {
        //     double diff = std::abs(temp[i] - RW_term[i]);
        //     if (diff > 1e-5) err_cnt++;
        //     if (diff > 1e-3)  err_cnt2++;
        // }
        // printf("num of error %d, err cnt2 %d\n", err_cnt, err_cnt2);


        // cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, n2, 1.0, A1T_S.data(), n1, adj2_dense.data(), n2, 0.0, RW_term.data(), n1);

        // log(RW_term)
        vdLn(n1*n2, RW_term.data(), RW_term.data());
        // C -= lambda_n * log(RW_term)
        cblas_daxpy(n1*n2, -lambda_n, RW_term.data(), 1, C.data(), 1);
        // C -= lambda_a * log(H)  (H_log is precomputed)
        cblas_daxpy(n1*n2, -lambda_a, H_log.data(), 1, C.data(), 1);
        auto time_C_done = std::chrono::high_resolution_clock::now();
        C_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_C_done - time_L_done).count();

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

        // compare(C, data_path + "/C_outer_iter_" + std::to_string(it) + ".txt", "C_iter_"+ std::to_string(it));

        // ---- Cost = C - lambda_p * log(S) ----
        cblas_dcopy(n1*n2, S.data(), 1, S_log.data(), 1);
        vdLn(n1*n2, S_log.data(), S_log.data());
        cblas_dcopy(n1*n2, C.data(), 1, Cost.data(), 1);
        cblas_daxpy(n1*n2, -lambda_p, S_log.data(), 1, Cost.data(), 1);

        // std::vector<double> a_row(n1, 1.0 / n1); 
        // std::vector<double> b_col(n2, 1.0 / n2);   
        std::vector<double> a_row(n2); 
        std::vector<double> b_col(n1); 
        auto time_iter_1 = std::chrono::high_resolution_clock::now();
        // std::cout << "Updating C before optimizing a and b with inner loop time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1-time_iter_0).count() << " ms" << std::endl;
        Cost_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_1 - time_C_done).count();
        // ---- Sinkhorn inner iterations ----
        for (int j = 0; j < inner_iters; ++j) {
            // printf(" \n ---- outer iter %d --- inner iter %d --- \n", it, j);
            // Cost - b  (b is n1 x 1, broadcast over columns)
            auto t_0 = std::chrono::high_resolution_clock::now() ;
            if( j > 0 ) {
                // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);
            }
            auto t_1 = std::chrono::high_resolution_clock::now() ;
            // buffer = Cost - B_mat
            mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, Cost.data(), n1, -1.0, B_mat.data(), n1, buffer.data(), n1);
            auto t_2 = std::chrono::high_resolution_clock::now() ;
            before_minaa += std::chrono::duration_cast<std::chrono::milliseconds>(t_2 - t_0).count();
            a_row = minaa(buffer, n1, n2, lambda_total);
            auto t_3 = std::chrono::high_resolution_clock::now() ;
            compute_minaa += std::chrono::duration_cast<std::chrono::milliseconds>(t_3 - t_2).count();
            // Cost - a  (a is 1 x n2, broadcast over rows)
            // A_mat = 1_n1 * a_row^T (n1 x n2)
            // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, a_row.data(), 1, 0.0, A_mat.data(), n1);
            cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, a_row.data(), 1, 0.0, A_mat.data(), n1); 
            auto t_4 = std::chrono::high_resolution_clock::now() ;
            // buffer = Cost - A_mat
            mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, Cost.data(), n1, -1.0, A_mat.data(), n1, buffer.data(), n1);
            auto t_5 = std::chrono::high_resolution_clock::now() ;
            between_minaa_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_5 - t_3).count();
            // b_col = minbb(buffer, lambda_total)  (length n1)
            b_col = minbb(buffer, n1, n2, lambda_total, j);
            auto t_6 = std::chrono::high_resolution_clock::now() ;
            compute_minbb += std::chrono::duration_cast<std::chrono::milliseconds>(t_6 - t_5).count();
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
        inner_loop += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_2 - time_iter_1).count();
        // printf("here  -- 3\n");
        // compare(a_row, data_path + "/a_outer_iter_" + std::to_string(it) + ".txt", "a_iter_"+ std::to_string(it));
        // compare(b_col, data_path + "/b_outer_iter_" + std::to_string(it) + ".txt", "b_iter_"+ std::to_string(it));

        // ---- S = 0.05 * S_old + 0.95 * r * exp((a + b - Cost)/lambda_total) * c ----
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

        auto time_iter_3 = std::chrono::high_resolution_clock::now();        
        // std::cout << "updating S after optimizing a and b time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_3-time_iter_2).count() << " ms" << std::endl;
        update_S += std::chrono::duration_cast<std::chrono::milliseconds>(time_iter_3 - time_iter_2).count();
        
        // //diff_S = sum |S - S_old|
        // mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, S.data(), n1, -1.0, S_old.data(), n1, buffer.data(), n1);
        // vdAbs(n1*n2, buffer.data(), diff_abs.data());
        // double diff_S = cblas_dasum(n1*n2, diff_abs.data(), 1);
        // printf("inner iter %d and diff of S %.6f\n\n", it, diff_S);

        
        // double maxS = S[0], minS = S[0];
        // for (int i = 1; i < n1 * n2; ++i) {
        //     if (S[i] > maxS) maxS = S[i];
        //     if (S[i] < minS) minS = S[i];
        // }
        // std::cerr << "Before exp: minS = " << minS << ", maxS = " << maxS << "\n";
        // if (it == outer_iter-1) 
        //     std::cout << "total inner iteration time " << std::chrono::duration_cast<std::chrono::milliseconds>(total_inner).count() << " ms" << std::endl;
    }
    auto time_4 = std::chrono::high_resolution_clock::now();
    std::cout << "total time of 10 outer iterations : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() << " ms" << std::endl;
    
    std::cout << " L_compute_time : " << L_compute_time <<  " C_compute_time : " << C_compute_time <<  " Cost_compute_time : " << Cost_compute_time <<  
    " before_mina : " << before_minaa << " mina : " << compute_minaa << " between_mina_minb : " << between_minaa_minbb << " minb : " << compute_minbb << " update_S : " << update_S << " ms" << std::endl;
    
    std::cout << "done con_prox_pt_opt_init" << std::endl;

    return S;

}


// S: column-major n1 x n2 matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
std::unordered_map<int, double> hits_ks_ltr_scores( const std::vector<double>& S, int n1, int n2, 
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {
    const int num_tests = anchor_pairs_test.first.size();
    std::unordered_map<int, int> hit_counts;
    std::unordered_map<int, double> hits_ks;
    for (int k : ks) hit_counts[k] = 0;
    std::vector<int> idx(n2);

    for (int t = 0; t < num_tests; ++t) {
        int i = anchor_pairs_test.first[t];
        int j_true = anchor_pairs_test.second[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[i + a * n1] > S[i + b * n1];
            });

        for (int r = 0; r < n2; ++r) {
            if (idx[r] == j_true) {
                for (int k : ks)
                    if (r < k) hit_counts[k]++;
                break;
            }
        }
    }
    // normalize
    for (int k : ks) {
        hits_ks[k] = static_cast<double>(hit_counts[k]) / num_tests;
    }

    return hits_ks;
}

// S: column-major n1 x n2 similarity matrix
// anchor_pairs_test: vector of (left_node, right_node)
// ks: e.g. {1, 10, 30, 50}
std::unordered_map<int, double> hits_ks_rtl_scores( const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {

    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();

    std::unordered_map<int, int> hit_counts;
    std::unordered_map<int, double> hits_ks;
    for (int k : ks) hit_counts[k] = 0;

    std::vector<int> idx(n1);

    for (int t = 0; t < num_tests; ++t) {
        int j = right_ids[t];
        int i_true = left_ids[t];
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[a + j * n1] > S[b + j * n1];
            });
        for (int r = 0; r < n1; ++r) {
            if (idx[r] == i_true) {
                for (int k : ks)
                    if (r < k) hit_counts[k]++;
                break;
            }
        }
    }
    for (int k : ks)
        hits_ks[k] = double(hit_counts[k]) / num_tests;

    return hits_ks;
}

std::unordered_map<int, double> hits_ks_mean_scores( const std::vector<double>& S, int n1, int n2,
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, const std::vector<int>& ks) {
    std::unordered_map<int, double> hits_ks;
    auto hits_ltr = hits_ks_ltr_scores( S, n1, n2, anchor_pairs_test, ks);
    auto hits_rtl = hits_ks_rtl_scores( S, n1, n2, anchor_pairs_test, ks);
    for (int k : ks) {
        hits_ks[k] = 0.5 * (hits_ltr[k] + hits_rtl[k]);
    }
    return hits_ks;
}


double mrr_ltr_score_cpp( const std::vector<double>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();
    std::vector<int> idx(n2);
    double mrr_sum = 0.0;

    for (int t = 0; t < num_tests; ++t) {
        int i = left_ids[t];        // left node
        int j_true = right_ids[t];  // correct right node
        std::iota(idx.begin(), idx.end(), 0);
        // sort similarity row i descending
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[i + a * n1] > S[i + b * n1];
            });
        // find rank
        for (int r = 0; r < n2; ++r) {
            if (idx[r] == j_true) {
                mrr_sum += 1.0 / double(r + 1);  // 1-based rank
                break;
            }
        }
    }

    return mrr_sum / num_tests;
}

double mrr_rtl_score_cpp( const std::vector<double>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    const auto& left_ids  = anchor_pairs_test.first;
    const auto& right_ids = anchor_pairs_test.second;
    const int num_tests = left_ids.size();
    std::vector<int> idx(n1);
    double mrr_sum = 0.0;

    for (int t = 0; t < num_tests; ++t) {
        int j = right_ids[t];       // right node
        int i_true = left_ids[t];   // correct left node
        std::iota(idx.begin(), idx.end(), 0);
        // sort similarity column j descending
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                return S[a + j * n1] > S[b + j * n1];
            });
        // find rank
        for (int r = 0; r < n1; ++r) {
            if (idx[r] == i_true) {
                mrr_sum += 1.0 / double(r + 1);  // 1-based rank
                break;
            }
        }
    }

    return mrr_sum / num_tests;
}

double mrr_mean_score_cpp( const std::vector<double>& S, int n1, int n2, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test) {
    double mrr_ltr = mrr_ltr_score_cpp(S, n1, n2, anchor_pairs_test);
    double mrr_rtl = mrr_rtl_score_cpp(S, n1, n2, anchor_pairs_test);
    return 0.5 * (mrr_ltr + mrr_rtl);
}
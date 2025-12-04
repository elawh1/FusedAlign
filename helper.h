#include <mkl.h>
#include <cstdio>
#include <cinttypes>   
#include <iostream>
#include <vector>
#include <mat.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <mkl_vml.h>
// #include <mkl_vml_functions.h>


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




struct MKLCSC {
    sparse_matrix_t spMat = nullptr;
    std::vector<MKL_INT> col_start, col_end, row_ind;
    std::vector<double>  vals;
    ~MKLCSC() { if (spMat) mkl_sparse_destroy(spMat); }
};


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

sparse_matrix_t make_coo_from_vec(const std::vector<double>& vec, int m, int n) {
	MKL_INT nnz = 0;
	for(int i=0; i<vec.size(); i++) {
		if(vec[i] != 0)
			nnz++;
	}
	std::vector<MKL_INT> rows(nnz), cols(nnz);
	std::vector<double> vals(nnz);
	int k = 0;
	for(int i=0; i<vec.size(); i++) {
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
	for(int i=0; i<vec.size(); i++) {
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


sparse_matrix_t make_one_hot_csc(std::vector<MKL_INT> vec, int m, int n) {
    MKL_INT nnz = vec.size();
    std::vector<MKL_INT> col_start(n), col_end(n);     
    std::vector<MKL_INT> row_ind(nnz);                 
    std::vector<double>  vals(nnz);

    std::sort(vec.begin(), vec.end());

    int k = 0;
	for (MKL_INT j = 0; j < nnz; ++j) {
        if (j   >  0)    col_end[j-1] = j;
        col_start[j] = j;
        row_ind[j] = vec[j];
        vals[k] = (double) 1.0;
    }

	sparse_matrix_t A = nullptr;
    auto st = mkl_sparse_d_create_csc(&A, SPARSE_INDEX_BASE_ZERO, m, n, col_start.data(), col_end.data(), row_ind.data(), vals.data());
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_create_csc failed");
    mkl_sparse_optimize(A);
	return A;

}


//this one works if g2_rwr nxl dense matrix stores row major
// std::vector<double> l2_norm(std::vector<double>  g2_rwr, int n, int d) {
//     std::vector<double> norms(n);

//     // Compute row-wise L2 norm: norms[i] = sqrt(sum_j g2_rwr[i,j]^2)
//     for (int i = 0; i < n; i++)
//         norms[i] = cblas_dnrm2(d, &g2_rwr[i * d], 1);

//     // Invert norms (avoid divide-by-zero)
//     for (int i = 0; i < n; i++)
//         if (norms[i] > 0) norms[i] = 1.0 / norms[i];

//     // Scale each row by 1/norm[i]
//     for (int i = 0; i < n; i++)
//         cblas_dscal(d, norms[i], &g2_rwr[i * d], 1);

//     return g2_rwr;
// }

std::vector<double> l2_norm(std::vector<double> g2_rwr, int n, int d) {
    std::vector<double> norms(n);
    // Row-wise L2 norm: row i = { g2_rwr[i + j*n] for j=0..d-1 }
    for (int i = 0; i < n; i++)
        norms[i] = cblas_dnrm2(d, &g2_rwr[i], n);   // N=d, incx = n
    // Invert norms
    for (int i = 0; i < n; i++)
        if (norms[i] > 0.0) norms[i] = 1.0 / norms[i];
    // Scale each row by 1/norm[i]
    for (int i = 0; i < n; i++)
        cblas_dscal(d, norms[i], &g2_rwr[i], n);   // N=d, incx = n
    return g2_rwr;
}



std::vector<double> get_normalized_neg_exp_dist (std::vector<double>  rwr1, std::vector<double>  rwr2, int n1, int n2, int l) {
    std::vector<double> rwr1_norm = l2_norm( rwr1, n1, l);
    std::vector<double> rwr2_norm = l2_norm( rwr2, n2, l);

    std::vector<double> cross_rwr_dist(n1 * n2);

    // cross_rwr_dist = exp( - rwr1_norm * rwr2_norm^T )
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, n2, l, -1.0, rwr1_norm.data(), l, rwr2_norm.data(), l, 0.0, cross_rwr_dist.data(), n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, l, -1.0, rwr1_norm.data(), n1, rwr2_norm.data(), n2, 0.0, cross_rwr_dist.data(), n1);

    vdExp(n1 * n2, cross_rwr_dist.data(), cross_rwr_dist.data());
    return cross_rwr_dist;
}

std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<double>& H_vec, MKL_INT m, MKL_INT n) {
    std::vector<MKL_INT> anchor_g1;
    std::vector<MKL_INT> anchor_g2;

    for(MKL_INT i = 0; i < m; i++) {
        for(MKL_INT j = 0; j < n; j++) {
            if(H_vec[(size_t)j * m + i] != (double) 0.0) {
                // printf("anchor %d %d\n", i, j);
                anchor_g1.push_back(i);
                anchor_g2.push_back(j);
            }
        }
    }
    return {anchor_g1, anchor_g2};
}

std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<double>& H_vec) {
    std::vector<MKL_INT> anchor_g1;
    std::vector<MKL_INT> anchor_g2;
    for(int i=0; i< H_vec.size()/2; i++) {
        anchor_g1.push_back((MKL_INT) H_vec[i*2]);
        anchor_g2.push_back((MKL_INT) H_vec[i*2+1]);
    }
    return {anchor_g1, anchor_g2};
}

std::vector<double>  get_batch_rwr_scores(sparse_matrix_t graph, std::vector<MKL_INT> anchors, MKL_INT n,  float restart_prob, int max_iters, bool connect_isolated) {
	std::vector<double>  rwr;
    //connect isolated nodes and row normalize A  
    std::vector<double> all_ones(n, (double) 1.0);
    std::vector<double> graph_row_sum(n, (double) 0.0);
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
    // if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed");
    if (status != SPARSE_STATUS_SUCCESS) {
        switch (status) {
            case SPARSE_STATUS_NOT_INITIALIZED:
                std::cerr << "MKL Error: The routine was not initialized." << std::endl;
                break;
            case SPARSE_STATUS_ALLOC_FAILED:
                std::cerr << "MKL Error: Memory allocation failed." << std::endl;
                break;
            default:
                std::cerr << "MKL Error: An unknown error occurred with status code " << status << std::endl;
                break;
        }
    } 
    // -----    TODO   -----  Connect isolated nodes
    
    //
    MKLCSC Dinv = make_diag_csc(graph_row_sum, n, n, true);
    mkl_verbose(1);
    sparse_matrix_t graph_norm = nullptr;
    
    mkl_sparse_optimize(Dinv.spMat);
    mkl_sparse_optimize(graph);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph, &graph_norm);
    if (status != SPARSE_STATUS_SUCCESS) {
        switch (status) {
            case SPARSE_STATUS_NOT_INITIALIZED:
                std::cerr << "MKL Error: The routine was not initialized." << std::endl;
                break;
            case SPARSE_STATUS_ALLOC_FAILED:
                std::cerr << "MKL Error: Memory allocation failed." << std::endl;
                break;
            default:
                std::cerr << "MKL Error: An unknown error occurred with status code " << status << std::endl;
                break;
        }
    } 
    // throw std::runtime_error("mkl_sparse_spmm failed");
    mkl_sparse_optimize(graph_norm);
    mkl_verbose(0);
    
    int l = anchors.size();
    //these are column major
    std::vector<double> X_vals(n*l, 1.0);      // X(t), all ones
    std::vector<double> Xnext_vals(n*l, 0.0);

    sparse_matrix_t graph_norm_csr = nullptr;
    status = mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    assert(status == SPARSE_STATUS_SUCCESS && graph_norm_csr);
    status = mkl_sparse_optimize(graph_norm_csr);
    
    //one hot encoding E matrix
    std::vector<double> E(n * l, 0.0);
    for (int j = 0; j < l; ++j) {
        E[ anchors[j] + j * n ] = 1.0;   // one-hot at (row=anchor[j], col=j)
    }

    for(int i=0; i < max_iters; i++) {
        status = mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, (double) (1-restart_prob), graph_norm_csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X_vals.data(), l, n, 0.0, Xnext_vals.data(), n);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("error spmm");
        cblas_daxpy(n * l, restart_prob, E.data(), 1, Xnext_vals.data(), 1);

        // for (int j=0;j<l;++j){
        //     double s=0.0; for (int i2=0;i2<n;++i2) s += Xnext_vals[i2 + j*n];
        //     double mx=0; for(int i2=0;i2<n;++i2) mx = std::max(mx, X_vals[i2 + j*n]);
        //     if (std::abs(s-1.0) > 1e-10) printf("col %d    sum after +alpha*E = %.17g    max=%.3f \n", j, s, mx);
        // }

        std::swap(X_vals, Xnext_vals);

        // printf("rwr \n");
        // int cnt = 0;
        // for(int i=0; i < X_vals.size(); i++) {
        //     if (X_vals[i] > 0.2)    cnt++;
        // }
        // printf("total %d up threashod %d\n", X_vals.size(), cnt);

    }
    //column major dense matrix nxl
	return Xnext_vals;
}

int get_transport_cost() {

	return 1;	
}



void anchor_cost_zero(std::vector<double>& A, MKL_INT nrows, MKL_INT ncols, std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchors, double val)
{
    printf("intpu vec size %d  anchors pair size %d\n ", A.size(), anchors.first.size());
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
void init_H_with_anchors(std::vector<double>& H, MKL_INT nrows, MKL_INT ncols, 
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchors)
{
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


sparse_matrix_t row_normalize_graph( sparse_matrix_t graph, int n) 
{
    std::vector<double> all_ones(n, (double) 1.0);
    std::vector<double> graph_row_sum(n, (double) 0.0);
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
    // if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed");
    if (status != SPARSE_STATUS_SUCCESS) { std::cerr << "error : mkl_sparse_d_mv failed" << std::endl; }

    MKLCSC Dinv = make_diag_csc(graph_row_sum, n, n, true);
    sparse_matrix_t graph_norm = nullptr;
    mkl_sparse_optimize(Dinv.spMat);
    mkl_sparse_optimize(graph);
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Dinv.spMat, graph, &graph_norm);
    if (status != SPARSE_STATUS_SUCCESS) { std::cerr << "error : mkl_sparse_spmm failed" << std::endl; }
    sparse_matrix_t graph_norm_csr;
    mkl_sparse_convert_csr(graph_norm, SPARSE_OPERATION_NON_TRANSPOSE, &graph_norm_csr);
    
    return graph_norm_csr;
}

std::vector<double> get_product_rwr_mat( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, int n1, int n2, double rwr_restart_prob, 
    double gamma, int max_iters = 50, double tol = 1e-2) 
{
    std::cout << "inside get_product_rwr_mat" << std::endl;

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_matrix_t graph1_normalized = row_normalize_graph( graph1_py, n1);
    sparse_matrix_t graph2_normalized = row_normalize_graph( graph2_py, n2);

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
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_ROW_MAJOR, out_cross_dist.data(), n2, n2, 0.0, temp.data(), n2);
        //temp is n1xn2 temp = G1.cross_dist
        auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        double coeff = (1 - rwr_restart_prob) * gamma;
        //we need temp2 = temp^T
        mkl_domatcopy('C', 'T', n1, n2, 1.0, temp.data(), n1, temp2.data(), n2);
        //temp3 is n2xn1  temp3 = G2.(temp^T) = G2.(temp2)
        // st = mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_ROW_MAJOR, temp2.data(), n2, n2, 0.0, temp3.data(), n1);
        st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, temp2.data(), n1, n2, 0.0, temp3.data(), n2);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        // mkl_domatcopy('C', 'T', n1, n2, 1.0, temp2.data(), n2, out_cross_dist.data(), n2);
        //adding with scaling first element and transposing second element
        // mkl_domatadd('R', 'N', 'T', n1, n2, (1+rwr_restart_prob), in_cross_dist.data(), n2, 1.0, temp2.data(), n2, out_cross_dist.data(), n2);
        //out_cross_dist:  n1 x n2
        mkl_domatadd('C', 'N', 'T', n1, n2, (1+rwr_restart_prob), in_cross_dist.data(), n1, 1.0, temp3.data(), n2, out_cross_dist.data(), n1);
        //compute max of absolute values of (out_cross_dist - out_cross_dist_old)
        vdSub(n1 * n2, out_cross_dist_old.data(), out_cross_dist.data(), diff.data());   
        vdAbs(n1*n2, diff.data(), diff_abs.data());
        // out_cross_dist_abs_max_idx = cblas_idamax(n1*n2, diff_abs, 1);
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


// H_in: n1 x n2, row-major
std::vector<double> mina(const std::vector<double>& H_in, int n1, int n2, double epsilon)
{
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



std::vector<double> minb(const std::vector<double>& H_in, int n1, int n2, double epsilon, int it)
{
    
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
std::vector<double> minaa(const std::vector<double>& H, int n1, int n2, double epsilon)
{
    // const int N = n1 * n2;
    std::vector<double> H_shifted(n1 * n2);
    std::vector<double> col_min(n2);

    // column-wise minima using MKL idamin over each column
    // for (int j = 0; j < n2; ++j) {
    //     int idx = cblas_idamin(n1, &H[j * n1], 1);    // contiguous col j
    //     // if((std::isnan(H[j * n1 + idx] ) || std::isinf(H[j * n1 + idx] ) )) printf("errorrr - H has inf  j %d \n", j);
    //     if((j * n1 + idx) >= n1*n2 || (j * n1 + idx) < 0) printf("errorrr - row min idx out of range\n");
    //     else  col_min[j] = H[j * n1 + idx];
    // }
    for (int j = 0; j < n2; ++j) {
        const double* col = &H[j * n1];
        double m = col[0];
        for (int i = 1; i < n1; ++i) {
            if (col[i] < m) m = col[i];
        }
        col_min[j] = m;
    }

    // build matrix M = 1_n1 * col_min^T (n1 x n2)
    std::vector<double> ones_n1(n1, 1.0);
    std::vector<double> M(n1 * n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, ones_n1.data(), n1, col_min.data(), 1, 0.0, M.data(), n1);

    // H_shifted = H - M
    mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, H.data(), n1, -1.0, M.data(), n1, H_shifted.data(), n1);

    // mina on shifted
    std::vector<double> res = mina(H_shifted, n1, n2, epsilon); // size n2

    // add column minima back: res += col_min
    cblas_daxpy(n2, 1.0, col_min.data(), 1, res.data(), 1);

    return res;  // shape: (1 x n2)
}



// minbb(H) = minb(H - min(H,dim=1), eps) + min(H,dim=1)
std::vector<double> minbb(const std::vector<double>& H, int n1, int n2, double epsilon, int it)
{
    // const int N = n1 * n2;
    std::vector<double> H_shifted(n1 * n2);
    std::vector<double> row_min(n1);

    // row-wise minima via idamin over each row (contiguous)
    // for (int i = 0; i < n1; ++i) {
    //     // int idx = cblas_idamin(n2, &H[i * n2], 1);
    //     // row_min[i] = H[i * n2 + idx];
    //     int idx = cblas_idamin(n2, &H[i], n1);  // stride = n1 (column-major)
    //     // if (it == 0 ) printf(" %.6f ", H[i + idx * n1]);

    //     if(i + idx * n1 >= n1*n2 || i + idx * n1 < 0) printf("errorrr - row min idx out of range\n");
    //     else  row_min[i] = H[i + idx * n1];
    // }
    for (int i = 0; i < n1; ++i) {
        double m = H[i + 0 * n1];          // first column, row i
        for (int j = 1; j < n2; ++j) {
            double v = H[i + j * n1];      // row i, col j (col-major)
            if (v < m) m = v;
        }
        row_min[i] = m;
    }
  
    // build matrix M = row_min * 1_n2^T  (n1 x n2)
    std::vector<double> ones_n2(n2, 1.0);
    std::vector<double> M(n1 * n2);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, row_min.data(), n1, ones_n2.data(), 1, 0.0, M.data(), n1);

    // H_shifted = H - M
    // mkl_domatadd('R', 'N', 'N', n1, n2, 1.0, H.data(), n2, -1.0, M.data(), n2, H_shifted.data(), n2);
    mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, H.data(), n1, -1.0, M.data(), n1, H_shifted.data(), n1);

    // minb on shifted
    std::vector<double> res = minb(H_shifted, n1, n2, epsilon, it); // size n1

    // add row minima back: res += row_min
    cblas_daxpy(n1, 1.0, row_min.data(), 1, res.data(), 1);

    return res;  // shape: (n1 x 1)
}


std::vector<double> compute_L_fixed( const std::vector<double>& intra_dist1, const std::vector<double>& intra_dist2, const std::vector<double>& r, const std::vector<double>& c, 
    const std::vector<double>& one_vec_n1, const std::vector<double>& one_vec_n2, int n1, int n2)
{
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
    std::vector<double> term1(n1 * n2);
    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, 1, 1.0, v1.data(), n1, one_vec_n2.data(), n2, 0.0, term1.data(), n1);         // n1 x n2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, v1.data(), n1, one_vec_n2.data(), 1, 0.0, term1.data(), n1);         // n1 x n2

    // Second term: one_vec_n1 @ c @ (intra_dist2^2)^T
    // B0 = one_vec_n1 * c  (n1 x n2)
    std::vector<double> B0(n1 * n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, c.data(), 1, 0.0, B0.data(), n1);            // n1 x n2

    // term2 = B0 @ (A2)^T   where A2 = intra_dist2^2, size n2 x n2
    std::vector<double> term2(n1 * n2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, n2, 1.0, B0.data(), n1, A2.data(), n2, 0.0, term2.data(), n1);         // n1 x n2

    // 4) L_fixed = term1 + term2
    std::vector<double> L_fixed = term1;
    cblas_daxpy(n1 * n2, 1.0, term2.data(), 1, L_fixed.data(), 1);

    // for(int i=0; i<100; i++) { printf(" %.6f ", L_fixed[i]);}
    return L_fixed; // n1 x  n2, col-major
}




std::vector<double> con_prox_pt_opt( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, int n1, int n2,
    std::vector<double> cross_dist, std::vector<double> intra_dist1, std::vector<double> intra_dist2, int inner_iters, int outer_iters,
    double lambda_n, double lambda_a, double lambda_p, double lambda_e_in, const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs, 
    const std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>>& anchor_pairs_test, std::vector<double> anchors_test)
{
    sparse_matrix_t row_norm_adj1 = row_normalize_graph(graph1_py, n1);
    sparse_matrix_t row_norm_adj2 = row_normalize_graph(graph2_py, n2);

    double lambda_total = lambda_n + lambda_a + lambda_p;
    double lambda_e = lambda_e_in * n1 * n2;

    std::vector<double> one_vec_n1(n1, 1.0);
    std::vector<double> one_vec_n2(n2, 1.0);
    
    // a = one_vec_n1 / n1, r = one_vec_n1 / n1
    std::vector<double> a  = one_vec_n1;
    std::vector<double> r  = one_vec_n1;
    cblas_dscal(n1, 1.0 / static_cast<double>(n1), a.data(), 1);
    cblas_dscal(n1, 1.0 / static_cast<double>(n1), r.data(), 1);
    // b = one_vec_n2^T / n2, c = one_vec_n2^T / Before exp: n2
    std::vector<double> b  = one_vec_n2;
    std::vector<double> c  = one_vec_n2;
    cblas_dscal(n2, 1.0 / static_cast<double>(n2), b.data(), 1);
    cblas_dscal(n2, 1.0 / static_cast<double>(n2), c.data(), 1);
    std::vector<double> S(n1 * n2, 1.0 / static_cast<double>(n1 * n2));      // S = (1/(n1*n2)) * 1_n1 * 1_n2^T
    // std::vector<double> S (n1 * n2);
    // cblas_dger(CblasColMajor, n1, n2, 1.0 / static_cast<double>(n1 * n2), one_vec_n1.data(), 1, one_vec_n2.data(), 1, S.data(), n1);

    std::vector<double> H(n1 * n2, 1e-6);
    // cblas_dcopy(n1 * n2, S.data(), 1, H.data(), 1);           // H = S (all ones / (n1*n2))
    // cblas_dscal(n1 * n2, 1e-6 * n1 * n2, H.data(), 1);        // H = 1e-6
    // anchor_cost_zero(H, n1, n2, anchor_pairs, 1.0);
    init_H_with_anchors(H, n1, n2, anchor_pairs);

    std::vector<double> L_fixed = compute_L_fixed(intra_dist1, intra_dist2, r, c, one_vec_n1, one_vec_n2, n1, n2);
    std::vector<double> S_old(n1*n2);
    std::vector<double> A_S(n1*n2);
    std::vector<double> A2T_term(n1*n2);
    std::vector<double> L(n1*n2);
    std::vector<double> C(n1*n2);
    std::vector<double> C_old(n1*n2);
    std::vector<double> Cost(n1*n2);
    std::vector<double> S_log(n1*n2);
    std::vector<double> diff(n1*n2), diff_abs(n1*n2);
    std::vector<double> A_mat(n1*n2), B_mat(n1*n2);
    std::vector<double> RC_mat(n1*n2), C_mat(n1*n2);
    std::vector<double> A1T_S(n1 * n2);      // n1 x n2
    std::vector<double> A1T_S_T(n2 * n1);
    std::vector<double> A2_A1T_S_T(n2 * n1);
    std::vector<double> RW_term(n1*n2);
    // std::vector<double> a_row(n2);   
    // std::vector<double> b_col(n1);
    std::vector<double> b_row = one_vec_n2;          // length n2, (1 x n2)
    cblas_dscal(n2, 1.0 / double(n2), b_row.data(), 1);
    std::vector<double> b_col(n1);                   // length n1, (n1 x 1)
    std::vector<double> tmp_n1n2(n1*n2);
    std::vector<double> tmp_n1n2_2(n1*n2);
    std::vector<double> tmp_n2n1(n1 * n2);

    matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    std::vector<double> H_log = H;
    vdLn(n1 * n2, H_log.data(), H_log.data());

    bool has_C_old = false;

    // outer_iters = 2;
    for (int it = 0; it < outer_iters; ++it) {

        cblas_dcopy(n1 * n2, S.data(), 1, S_old.data(), 1);
        // A_S = intra_dist1 @ S   (n1 x n1) @ (n1 x n2) = (n1 x n2)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, n1, 1.0, intra_dist1.data(), n1, S.data(), n1, 0.0, A_S.data(), n1);         //col major

        // A2T_term = A_S @ intra_dist2^T   (n1 x n2) @ (n2 x n2) = (n1 x n2)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1, n2, n2, 1.0, A_S.data(), n1, intra_dist2.data(), n2, 0.0, A2T_term.data(), n1);           //col major
        // L = L_fixed - 2 * A2T_term
        
        cblas_dcopy(n1 * n2, L_fixed.data(), 1, L.data(), 1);
        cblas_daxpy(n1 * n2, -2.0, A2T_term.data(), 1, L.data(), 1);
        // ---- C = cross_dist + lambda_e * L - lambda_n * log(row_norm_adj1.T @ S @ row_norm_adj2) - lambda_a * log(H) ----
        // Start from cross_dist
        cblas_dcopy(n1 * n2, cross_dist.data(), 1, C.data(), 1);

        // C += lambda_e * L
        cblas_daxpy(n1 * n2, lambda_e, L.data(), 1, C.data(), 1);

        // Compute RW_term = row_norm_adj1.T @ S @ row_norm_adj2 (n1 x n2)
        // A1T_S = row_norm_adj1.T @ S
        // op(A) = TRANSPOSE, A: n1 x n1, S: n1 x n2 -> A1T_S: n1 x n2
        // sparse_status_t st = mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_ROW_MAJOR, S.data(), n2, n2, 0.0, A1T_S.data(),n2);
        sparse_status_t st = mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE, 1.0, row_norm_adj1, descr, SPARSE_LAYOUT_COLUMN_MAJOR, S.data(), n2, n1, 0.0, A1T_S.data(), n1);
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm (A1^T S) failed");

        // A1T_S_T = (A1T_S)^T (n2 x n1)
        // mkl_domatcopy('R', 'T', n1, n2, 1.0, A1T_S.data(), n2, A1T_S_T.data(), n1);  //row major
        mkl_domatcopy('C', 'T', n1, n2, 1.0, A1T_S.data(), n1, A1T_S_T.data(), n2);    //col major

        // st = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj2, descr, SPARSE_LAYOUT_ROW_MAJOR, A1T_S_T.data(), n1, n1, 0.0, A2_A1T_S_T.data(), n1);  //row major
        st = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, row_norm_adj2, descr, SPARSE_LAYOUT_COLUMN_MAJOR, A1T_S_T.data(), n1, n2, 0.0, A2_A1T_S_T.data(), n2);     // col major
        if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm (A2 A1^T S^T) failed");

        // RW_term = (A2_A1T_S_T)^T  (n1 x n2)
        // mkl_domatcopy('R', 'T', n2, n1, 1.0, A2_A1T_S_T.data(), n1, RW_term.data(), n2);
        mkl_domatcopy('C', 'T', n2, n1, 1.0, A2_A1T_S_T.data(), n2, RW_term.data(), n1);

        // log(RW_term)
        vdLn(n1*n2, RW_term.data(), RW_term.data());
        // C -= lambda_n * log(RW_term)
        cblas_daxpy(n1*n2, -lambda_n, RW_term.data(), 1, C.data(), 1);
        // C -= lambda_a * log(H)  (H_log is precomputed)
        cblas_daxpy(n1*n2, -lambda_a, H_log.data(), 1, C.data(), 1);

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
      
        if (has_nan(C) || has_nan(Cost) || has_nan(RW_term) || has_nan(a) || has_nan(b) || has_nan(S)) {
            std::cout << "NaN appeared at outer iter " << it << "\n";
            if(has_nan(C))
                std::cout << "NaN appeared in C at outer iter " << it << "\n";
            if(has_nan(Cost))
                std::cout << "NaN appeared in Cost at outer iter " << it << "\n";
            if(has_nan(RW_term))
                std::cout << "NaN appeared in RW_term at outer iter " << it << "\n";
            if(has_nan(a))
                std::cout << "NaN appeared in a at outer iter " << it << "\n";
            if(has_nan(b))
                std::cout << "NaN appeared in b at outer iter " << it << "\n";
            if(has_nan(S))
                std::cout << "NaN appeared in S at outer iter " << it << "\n";

        }
        // ---- Sinkhorn inner iterations ----
        for (int j = 0; j < inner_iters; ++j) {
            // Cost - b  (b is n1 x 1, broadcast over columns)
            // B_mat = b_col * 1_n2^T (n1 x n2)
            // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), 1, one_vec_n2.data(), n2, 0.0, B_mat.data(), n2);
            // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);
            if(j == 0)  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, b_row.data(), 1, 0.0, B_mat.data(), n1);
            else    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);

            // tmp_n1n2 = Cost - B_mat
            // mkl_domatadd('R', 'N', 'N', n1, n2, 1.0, Cost.data(), n2, -1.0, B_mat.data(), n2, tmp_n1n2.data(), n2);
            mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, Cost.data(), n1, -1.0, B_mat.data(), n1, tmp_n1n2.data(), n1);

            // a_row = minaa(tmp_n1n2, lambda_total)  (length n2)
            a = minaa(tmp_n1n2, n1, n2, lambda_total);
           
            // Cost - a  (a is 1 x n2, broadcast over rows)
            // A_mat = 1_n1 * a_row^T (n1 x n2)
            // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), 1, a_row.data(), n2, 0.0, A_mat.data(), n2);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, a_row.data(), 1, 0.0, A_mat.data(), n1);

            // tmp_n1n2 = Cost - A_mat
            // mkl_domatadd('R', 'N', 'N', n1, n2, 1.0, Cost.data(), n2, -1.0, A_mat.data(), n2, tmp_n1n2.data(), n2);
            mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, Cost.data(), n1, -1.0, A_mat.data(), n1, tmp_n1n2.data(), n1);

            // b_col = minbb(tmp_n1n2, lambda_total)  (length n1)
            b_col = minbb(tmp_n1n2, n1, n2, lambda_total, j);
        }


        // ---- S = 0.05 * S_old + 0.95 * r * exp((a + b - Cost)/lambda_total) * c ----

        // A_mat = 1_n1 * a_row^T
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), 1, a_row.data(), n2, 0.0, A_mat.data(), n2);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, a.data(), 1, 0.0, A_mat.data(), n1);
        // B_mat = b_col * 1_n2^T
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), 1, one_vec_n2.data(), n2, 0.0, B_mat.data(), n2);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, b_col.data(), n1, one_vec_n2.data(), 1, 0.0, B_mat.data(), n1);
        // tmp_n1n2 = A_mat + B_mat
        // mkl_domatadd('R', 'N', 'N', n1, n2, 1.0, A_mat.data(), n2, 1.0, B_mat.data(), n2, tmp_n1n2.data(), n2);
     
        mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, A_mat.data(), n1, 1.0, B_mat.data(), n1, tmp_n1n2.data(), n1);

        // tmp_n1n2 = tmp_n1n2 - Cost
        // mkl_domatadd('R', 'N', 'N', n1, n2, 1.0, tmp_n1n2.data(), n2, -1.0, Cost.data(), n2, tmp_n1n2.data(), n2);
        mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, tmp_n1n2.data(), n1, -1.0, Cost.data(), n1, tmp_n1n2.data(), n1);
   
        // tmp_n1n2 = tmp_n1n2 / lambda_total
        cblas_dscal(n1*n2, 1.0 / lambda_total, tmp_n1n2.data(), 1);

        // tmp_n1n2 = exp(tmp_n1n2)
        vdExp(n1*n2, tmp_n1n2.data(), tmp_n1n2.data());

        // RC_mat = r * 1_n2^T
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, r.data(), 1, one_vec_n2.data(), n2, 0.0, RC_mat.data(), n2);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, r.data(), n1, one_vec_n2.data(), 1, 0.0, RC_mat.data(), n1);
        // C_mat = 1_n1 * c^T
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), 1, c.data(), n2, 0.0, C_mat.data(), n2);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2, 1, 1.0, one_vec_n1.data(), n1, c.data(), 1, 0.0, C_mat.data(), n1);
    
        // tmp_n1n2 = tmp_n1n2 ⊙ RC_mat
        vdMul(n1*n2, tmp_n1n2.data(), RC_mat.data(), tmp_n1n2.data());
       
        // tmp_n1n2 = tmp_n1n2 ⊙ C_mat
        vdMul(n1*n2, tmp_n1n2.data(), C_mat.data(), tmp_n1n2.data());
      
        // S = 0.95 * tmp_n1n2 + 0.05 * S_old
        cblas_dcopy(n1*n2, tmp_n1n2.data(), 1, S.data(), 1);
        cblas_dscal(n1*n2, 0.95, S.data(), 1);
        cblas_daxpy(n1*n2, 0.05, S_old.data(), 1, S.data(), 1);
        // diff_S = sum |S - S_old|
        // mkl_domatadd('R', 'N', 'N', n1, n2, 1.0, S.data(), n2, -1.0, S_old.data(), n2, diff.data(), n2);
        mkl_domatadd('C', 'N', 'N', n1, n2, 1.0, S.data(), n1, -1.0, S_old.data(), n1, diff.data(), n1);
        vdAbs(n1*n2, diff.data(), diff_abs.data());
        double diff_S = cblas_dasum(n1*n2, diff_abs.data(), 1);
        printf("inner iter %d and diff of S %.6f\n", it, diff_S);
        double maxS = S[0], minS = S[0];
        for (int i = 1; i < n1 * n2; ++i) {
            if (S[i] > maxS) maxS = S[i];
            if (S[i] < minS) minS = S[i];
        }
        std::cerr << "Before exp: minS = " << minS << ", maxS = " << maxS << "\n";
    }
    std::cout << "done con_prox_pt_opt_init" << std::endl;

    return S;

}
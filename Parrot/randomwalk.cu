#ifdef MKL_AVAILABLE
#include <mkl.h>
#include <omp.h>
#include <mkl_vml.h>
#endif
#include <cinttypes>   
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <numeric>
#include <set>


#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>


#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t status_ = (call);                                            \
        if (status_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(status_));            \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUSPARSE(call)                                                     \
    do {                                                                         \
        cusparseStatus_t status_ = (call);                                       \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, static_cast<int>(status_));              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
do {                                                                           \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__         \
                  << " -> status " << static_cast<int>(st__) << std::endl;     \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

// __global__
// void get_batch_rwr_scores_cuda(double* res, float* graph, int* anchors, int n, int avg_nnz_per_n, bool directed, float restart_prob, int max_iters, bool connect_isolated)
// {
//   printf("cuda\n");
// }

std::vector<double>  get_batch_rwr_scores_cuda(sparse_matrix_t graph, std::vector<MKL_INT> anchors, MKL_INT n, MKL_INT avg_nnz_per_n, bool directed, float restart_prob, int max_iters, bool connect_isolated, const std::vector<double>& graph_edgelist) {
    
    
    
    
    // matrix_descr descr;
    // descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    auto time_0 = std::chrono::high_resolution_clock::now();
    
    //connect isolated nodes and row normalize A  
    std::vector<double> all_ones(n, (double) 1.0);
    std::vector<double> graph_row_sum(n, (double) 0.0);
    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, (double) 1.0, graph, descr, all_ones.data(), (double) 1.0, graph_row_sum.data());
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mv failed in get_batch_rwr_scores");
    
    // int n_int = (int) n;
    // int nnz = graph_edgelist.size() / 2;
    // printf("%d %d\n", n_int, nnz);
    // std::vector<int> h_row(nnz), h_col(nnz);
    // for(int i=0; i < nnz; i++) {
    //     h_row[i] = graph_edgelist[2*i];
    //     h_col[i] = graph_edgelist[2*i+1];	
    // }
    // std::vector<double> h_val(nnz, (double) 1.0);

    // int *d_row = nullptr, *d_col = nullptr;
    // double *d_val = nullptr, *d_x = nullptr, *d_y = nullptr;

    // CHECK_CUDA(cudaMalloc((void**)&d_row, nnz * sizeof(int)));
    // CHECK_CUDA(cudaMalloc((void**)&d_col, nnz * sizeof(int)));
    // CHECK_CUDA(cudaMalloc((void**)&d_val, nnz * sizeof(double)));
    // CHECK_CUDA(cudaMalloc((void**)&d_x, n_int * sizeof(double)));
    // CHECK_CUDA(cudaMalloc((void**)&d_y, n_int * sizeof(double)));

    // CHECK_CUDA(cudaMemcpy(d_row, h_row.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(d_col, h_col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(d_val, h_val.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(d_x,   all_ones.data(),   n_int * sizeof(double), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(d_y,   graph_row_sum.data(),   n_int * sizeof(double), cudaMemcpyHostToDevice));

    // cusparseHandle_t handle;
    // CHECK_CUSPARSE(cusparseCreate(&handle));

    // cusparseSpMatDescr_t matA;
    // cusparseDnVecDescr_t vecX, vecY;

    // CHECK_CUSPARSE(cusparseCreateCoo(
    //     &matA,
    //     n_int,
    //     n_int,
    //     nnz,
    //     d_row,
    //     d_col,
    //     d_val,
    //     CUSPARSE_INDEX_32I,
    //     CUSPARSE_INDEX_BASE_ZERO,
    //     CUDA_R_64F));

    // CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n_int, d_x, CUDA_R_64F));
    // CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n_int, d_y, CUDA_R_64F));

    // double alpha = 1.0f;
    // double beta  = 1.0f;

    // size_t bufferSize = 0;
    // CHECK_CUSPARSE(cusparseSpMV_bufferSize(
    //     handle,
    //     CUSPARSE_OPERATION_NON_TRANSPOSE,
    //     &alpha,
    //     matA,
    //     vecX,
    //     &beta,
    //     vecY,
    //     CUDA_R_64F,
    //     CUSPARSE_SPMV_ALG_DEFAULT,
    //     &bufferSize));

    // void* dBuffer = nullptr;
    // CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // CHECK_CUSPARSE(cusparseSpMV(
    //     handle,
    //     CUSPARSE_OPERATION_NON_TRANSPOSE,
    //     &alpha,
    //     matA,
    //     vecX,
    //     &beta,
    //     vecY,
    //     CUDA_R_64F,
    //     CUSPARSE_SPMV_ALG_DEFAULT,
    //     dBuffer));

    // CHECK_CUDA(cudaMemcpy(graph_row_sum.data(), d_y, n_int * sizeof(double), cudaMemcpyDeviceToHost));

    // cusparseDestroySpMat(matA);
    // cusparseDestroyDnVec(vecX);
    // cusparseDestroyDnVec(vecY);
    // cusparseDestroy(handle);

    // cudaFree(dBuffer);
    // cudaFree(d_row);
    // cudaFree(d_col);
    // cudaFree(d_val);
    // cudaFree(d_x);
    // cudaFree(d_y);
    // return graph_row_sum;


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


    MKL_INT rows, cols;
    MKL_INT *row_start, *row_end, *col_indx;
    double *values;
    sparse_index_base_t indexing;

    mkl_sparse_d_export_csr(
        graph_norm_csr,
        &indexing,
        &rows,
        &cols,
        &row_start,
        &row_end,
        &col_indx,
        &values);

    std::vector<int> h_rowPtr(rows + 1);

    for (int i = 0; i < rows; i++) {
        h_rowPtr[i] = row_start[i];
    }
    h_rowPtr[rows] = row_end[rows - 1];

    // If MKL indexing is one-based, convert to zero-based for cuSPARSE
    if (indexing == SPARSE_INDEX_BASE_ONE) {
        for (int i = 0; i <= rows; i++) h_rowPtr[i]--;
        for (int i = 0; i < row_end[rows - 1]; i++) col_indx[i]--;
    }
    int nnz = h_rowPtr[rows];

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

    // ------------------------------------------------------------
    // Allocate GPU memory
    // ------------------------------------------------------------
    int *d_rowPtr = nullptr, *d_colInd = nullptr;
    double *d_vals = nullptr, *d_X = nullptr, *d_Xnext = nullptr, *d_E = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals, nnz * sizeof(double)));

    CHECK_CUDA(cudaMalloc((void**)&d_X, n * l * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Xnext, n * l * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_E, n * l * sizeof(double)));

    // ------------------------------------------------------------
    // Copy data to GPU
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, col_indx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_X, E.data(), n * l * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Xnext, Xnext_vals.data(), n * l * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, E.data(), n * l * sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matX, matXnext; 

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        rows, cols, nnz,
        d_rowPtr,
        d_colInd,
        d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    // Column-major dense matrices, ld = n
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matX,
        n, l, n,
        d_X,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matXnext,
        n, l, n,
        d_Xnext,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));

    const double alpha = 1.0 - restart_prob;

    size_t bufferSize = 0;
    void* dBuffer = nullptr;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matX,
        &restart_prob,
        matXnext,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    auto time_3 = std::chrono::high_resolution_clock::now();
    for(int i=0; i < max_iters-1; i++) {
        // Xnext_vals = E;
        // // const sparse_operation_t operation, const double alpha, const sparse_matrix_t A, const struct matrix_descr descr, const sparse_layout_t layout, const double *B, const MKL_INT columns, const MKL_INT ldb, const double beta, double *C, const MKL_INT ldc);
        // auto status = mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, (double) (1-restart_prob), graph_norm_csr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X_vals.data(), l, n, restart_prob, Xnext_vals.data(), n);
        // if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("error spmm");
        // X_vals = Xnext_vals;

        // Xnext = E
        CHECK_CUDA(cudaMemcpy(
            d_Xnext,
            d_E,
            n * l * sizeof(double),
            cudaMemcpyDeviceToDevice));

        // Xnext = (1-r) * A^T * X + r * Xnext
        CHECK_CUSPARSE(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matX,
            &restart_prob,
            matXnext,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));

        // Swap X and Xnext for next iteration
        std::swap(d_X, d_Xnext);
        std::swap(matX, matXnext);

        // Update descriptor value pointers after swap
        CHECK_CUSPARSE(cusparseDnMatSetValues(matX, d_X));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matXnext, d_Xnext));

    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto time_4= std::chrono::high_resolution_clock::now();
    std::cout << "connect isolated and normalize : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_0).count()  
    << " rwr before iterations: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()  
    << " rwr iteration=100: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count()  
    << " ms" << std::endl;


    Xnext_vals.assign(n * l, 0.0);
    CHECK_CUDA(cudaMemcpy(Xnext_vals.data(), d_X, n * l * sizeof(double), cudaMemcpyDeviceToHost));
        // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matX));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matXnext));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Xnext));
    CHECK_CUDA(cudaFree(d_E));

    //column major dense matrix nxl
	return Xnext_vals;
}



std::vector<double> get_product_rwr_mat_cuda( sparse_matrix_t graph1_py, sparse_matrix_t graph2_py, const std::vector<double>& in_cross_dist, 
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
    std::vector<double> temp(n1 * n2);
    std::vector<double> CC(n1 * n2);


    MKL_INT rows_g1, cols_g1;
    MKL_INT *row_start_g1, *row_end_g1, *col_indx_g1;
    double *values_g1;
    sparse_index_base_t indexing_g1;

    mkl_sparse_d_export_csr(
        graph1_normalized,
        &indexing_g1,
        &rows_g1,
        &cols_g1,
        &row_start_g1,
        &row_end_g1,
        &col_indx_g1,
        &values_g1);

    std::vector<int> h_rowPtr_g1(rows_g1 + 1);

    for (int i = 0; i < rows_g1; i++) {
        h_rowPtr_g1[i] = row_start_g1[i];
    }
    h_rowPtr_g1[rows_g1] = row_end_g1[rows_g1 - 1];

    // If MKL indexing is one-based, convert to zero-based for cuSPARSE
    if (indexing_g1 == SPARSE_INDEX_BASE_ONE) {
        for (int i = 0; i <= rows_g1; i++) h_rowPtr_g1[i]--;
        for (int i = 0; i < row_end_g1[rows_g1 - 1]; i++) col_indx_g1[i]--;
    }
    int nnz_g1 = h_rowPtr_g1[rows_g1];

    MKL_INT rows_g2, cols_g2;
    MKL_INT *row_start_g2, *row_end_g2, *col_indx_g2;
    double *values_g2;
    sparse_index_base_t indexing_g2;

    mkl_sparse_d_export_csr(
        graph2_normalized,
        &indexing_g2,
        &rows_g2,
        &cols_g2,
        &row_start_g2,
        &row_end_g2,
        &col_indx_g2,
        &values_g2);

    std::vector<int> h_rowPtr_g2(rows_g2 + 1);

    for (int i = 0; i < rows_g2; i++) {
        h_rowPtr_g2[i] = row_start_g2[i];
    }
    h_rowPtr_g2[rows_g2] = row_end_g2[rows_g2 - 1];

    // If MKL indexing is one-based, convert to zero-based for cuSPARSE
    if (indexing_g2 == SPARSE_INDEX_BASE_ONE) {
        for (int i = 0; i <= rows_g2; i++) h_rowPtr_g2[i]--;
        for (int i = 0; i < row_end_g2[rows_g2 - 1]; i++) col_indx_g2[i]--;
    }
    int nnz_g2 = h_rowPtr_g2[rows_g2];
    // ------------------------------------------------------------
    // Allocate GPU memory
    // ------------------------------------------------------------
    int *d_rowPtr_g1 = nullptr, *d_colInd_g1 = nullptr;
    double *d_vals_g1 = nullptr;
    int *d_rowPtr_g2 = nullptr, *d_colInd_g2 = nullptr;
    double *d_vals_g2 = nullptr;
    double *d_out_cross_dist = nullptr;
    double *d_out_cross_dist_old = nullptr;
    double *d_in_cross_dist = nullptr;
    double *d_temp = nullptr;
    double *d_CC = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr_g1, (rows_g1 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd_g1, nnz_g1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_g1, nnz_g1 * sizeof(double)));

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr_g2, (rows_g2 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd_g2, nnz_g2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_g2, nnz_g2 * sizeof(double)));

    CHECK_CUDA(cudaMalloc((void**)&d_out_cross_dist,     n1 * n2 * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_out_cross_dist_old, n1 * n2 * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_in_cross_dist,      n1 * n2 * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_temp,               n1 * n2 * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_CC,                 n1 * n2 * sizeof(double)));

    // ------------------------------------------------------------
    // Copy data to GPU
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_rowPtr_g1, h_rowPtr_g1.data(), (rows_g1 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd_g1, col_indx_g1, nnz_g1 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_g1, values_g1, nnz_g1 * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_rowPtr_g2, h_rowPtr_g2.data(), (rows_g2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd_g2, col_indx_g2, nnz_g2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_g2, values_g2, nnz_g2 * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_out_cross_dist, out_cross_dist.data(), n1 * n2 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_cross_dist,  in_cross_dist.data(),  n1 * n2 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_temp, 0, n1 * n2 * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_CC,   0, n1 * n2 * sizeof(double)));

    cusparseHandle_t sp_handle = nullptr;
    cublasHandle_t blas_handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&sp_handle));
    CHECK_CUBLAS(cublasCreate(&blas_handle));

    cusparseSpMatDescr_t G1 = nullptr, G2 = nullptr;
    cusparseDnMatDescr_t OUT_n1xn2 = nullptr;
    cusparseDnMatDescr_t TEMP_n1xn2 = nullptr;
    cusparseDnMatDescr_t OUT_n2xn1 = nullptr;
    cusparseDnMatDescr_t CC_n2xn1 = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(
        &G1,
        rows_g1, cols_g1, nnz_g1,
        d_rowPtr_g1,
        d_colInd_g1,
        d_vals_g1,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &G2,
        rows_g2, cols_g2, nnz_g2,
        d_rowPtr_g2,
        d_colInd_g2,
        d_vals_g2,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

     // Same underlying out_cross_dist buffer, but two shape views:
    // view 1: n1 x n2, ld=n1
    // view 2: n2 x n1, ld=n2
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &OUT_n1xn2, n1, n2, n1, d_out_cross_dist, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &OUT_n2xn1, n2, n1, n2, d_out_cross_dist, CUDA_R_64F, CUSPARSE_ORDER_COL));
        
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &TEMP_n1xn2, n1, n2, n1, d_temp, CUDA_R_64F, CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &CC_n2xn1, n2, n1, n2, d_CC, CUDA_R_64F, CUSPARSE_ORDER_COL));


    void* dBuffer1 = nullptr;
    void* dBuffer2 = nullptr;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    {
        double alpha = 1.0;
        double beta  = 0.0;

        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G1,
            OUT_n1xn2,
            &beta,
            TEMP_n1xn2,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            &bufferSize1));

        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));

        CHECK_CUSPARSE(cusparseSpMM_preprocess(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G1,
            OUT_n1xn2,
            &beta,
            TEMP_n1xn2,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer1));
    }

    {
        double alpha = (1.0 - rwr_restart_prob) * gamma;
        double beta  = (1.0 + rwr_restart_prob);

        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G2,
            OUT_n2xn1,
            &beta,
            CC_n2xn1,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            &bufferSize2));

        CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));

        CHECK_CUSPARSE(cusparseSpMM_preprocess(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            G2,
            OUT_n2xn1,
            &beta,
            CC_n2xn1,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer2));
    }


    auto time_2 = std::chrono::high_resolution_clock::now();
    auto time_only_spmm = time_2 - time_2;
    for (int i = 0; i < max_iters; ++i) {
        auto time_2_0 = std::chrono::high_resolution_clock::now();

        // out_cross_dist_old = out_cross_dist;
        // mkl_domatcopy('C', 'T', n1, n2, 1.0, in_cross_dist.data(), n1, CC.data(), n2);
        
        // // 2 spmm correct working version
        // auto st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n2, n1, 0.0, temp.data(), n1);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("first mkl_sparse_d_mm failed");
        // mkl_domatcopy('C', 'T', n1, n2, 1.0, temp.data(), n1, out_cross_dist.data(), n2);
        // double coeff = (1 - rwr_restart_prob) * gamma;
        // auto time_2_0_1 = std::chrono::high_resolution_clock::now();
        // st = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, coeff, graph2_normalized, descr, SPARSE_LAYOUT_COLUMN_MAJOR, out_cross_dist.data(), n1, n2, (1+rwr_restart_prob), CC.data(), n2);
        // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("second mkl_sparse_d_mm failed");
        // auto time_2_0_2 = std::chrono::high_resolution_clock::now();
        // mkl_domatcopy('C', 'T', n2, n1, 1.0, CC.data(), n2, out_cross_dist.data(), n1);
        

        // out_cross_dist_old = out_cross_dist
        CHECK_CUDA(cudaMemcpy(
            d_out_cross_dist_old,
            d_out_cross_dist,
            n1 * n2 * sizeof(double),
            cudaMemcpyDeviceToDevice));

        // CC = in_cross_dist^T
        {
            const double alpha = 1.0;
            const double beta  = 0.0;

            // input  d_in_cross_dist: n1 x n2, ld = n1
            // output d_CC          : n2 x n1, ld = n2
            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n2, n1,
                &alpha,
                d_in_cross_dist, n1,
                &beta,
                d_in_cross_dist, n2, // unused because beta=0
                d_CC, n2));
        }

        // temp = G1 * out_cross_dist
        {
            double alpha = 1.0;
            double beta  = 0.0;

            CHECK_CUSPARSE(cusparseSpMM(
                sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                G1,
                OUT_n1xn2,
                &beta,
                TEMP_n1xn2,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer1));
        }

        // out_cross_dist = temp^T
        {
            const double alpha = 1.0;
            const double beta  = 0.0;

            // temp: n1 x n2 -> out_cross_dist viewed as n2 x n1
            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n2, n1,
                &alpha,
                d_temp, n1,
                &beta,
                d_temp, n2, // unused because beta=0
                d_out_cross_dist, n2));
        }

        // CC = coeff * G2 * out_cross_dist + (1+rwr_restart_prob) * CC
        {
            double alpha = (1.0 - rwr_restart_prob) * gamma;
            double beta  = (1.0 + rwr_restart_prob);

            CHECK_CUSPARSE(cusparseSpMM(
                sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                G2,
                OUT_n2xn1,
                &beta,
                CC_n2xn1,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer2));
        }

        // out_cross_dist = CC^T
        {
            const double alpha = 1.0;
            const double beta  = 0.0;

            CHECK_CUBLAS(cublasDgeam(
                blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n1, n2,
                &alpha,
                d_CC, n2,
                &beta,
                d_CC, n1, // unused because beta=0
                d_out_cross_dist, n1));
        }

        // host-side convergence check
        CHECK_CUDA(cudaMemcpy(
            out_cross_dist.data(),
            d_out_cross_dist,
            n1 * n2 * sizeof(double),
            cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaMemcpy(
            out_cross_dist_old.data(),
            d_out_cross_dist_old,
            n1 * n2 * sizeof(double),
            cudaMemcpyDeviceToHost));


        auto time_2_1 = std::chrono::high_resolution_clock::now();
        time_only_spmm += time_2_1 - time_2_0;
        //compute max of absolute values of (out_cross_dist - out_cross_dist_old)
        if (out_cross_dist_converge(out_cross_dist, out_cross_dist_old, tol)) break;
        auto time_2_2 = std::chrono::high_resolution_clock::now();

        
     }
    auto time_3 = std::chrono::high_resolution_clock::now();
    
    //out_cross_dist = (1 - self.gamma) * out_cross_dist
    cblas_dscal(n1 * n2, 1.0 - gamma, out_cross_dist.data(), 1);
    auto time_4 = std::chrono::high_resolution_clock::now();

    std::cout << "in get_product_rwr_mat row normalize " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   << 
    " init " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   << 
    " for loop " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()   << 
    " cblas_dscal " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() <<
    " time only spmm " << std::chrono::duration_cast<std::chrono::milliseconds>(time_only_spmm).count()
    << std::endl;
    

     CHECK_CUDA(cudaFree(dBuffer1));
    CHECK_CUDA(cudaFree(dBuffer2));

    CHECK_CUSPARSE(cusparseDestroyDnMat(CC_n2xn1));
    CHECK_CUSPARSE(cusparseDestroyDnMat(OUT_n2xn1));
    CHECK_CUSPARSE(cusparseDestroyDnMat(TEMP_n1xn2));
    CHECK_CUSPARSE(cusparseDestroyDnMat(OUT_n1xn2));

    CHECK_CUSPARSE(cusparseDestroySpMat(G2));
    CHECK_CUSPARSE(cusparseDestroySpMat(G1));

    CHECK_CUSPARSE(cusparseDestroy(sp_handle));
    CHECK_CUBLAS(cublasDestroy(blas_handle));

    CHECK_CUDA(cudaFree(d_CC));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_in_cross_dist));
    CHECK_CUDA(cudaFree(d_out_cross_dist_old));
    CHECK_CUDA(cudaFree(d_out_cross_dist));

    CHECK_CUDA(cudaFree(d_rowPtr_g1));
    CHECK_CUDA(cudaFree(d_colInd_g1));
    CHECK_CUDA(cudaFree(d_vals_g1));

    CHECK_CUDA(cudaFree(d_rowPtr_g2));
    CHECK_CUDA(cudaFree(d_colInd_g2));
    CHECK_CUDA(cudaFree(d_vals_g2));

    return out_cross_dist;
}
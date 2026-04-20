#include <iostream>
#include <vector>
#ifdef MKL_AVAILABLE
#include <mkl.h>
#include <mkl_vml.h>
#include <omp.h>
#endif
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <ctime> 
#include <chrono>
#include <cinttypes> 
#include "helper2.h"

// Reference SpMM: CSR (n1 x n1) * dense row-major B (n1 x n2) -> row-major C (n1 x n2)
void spmm_csr_rowmajor_ref(
    int n1, int n2,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& vals,
    const std::vector<double>& B_row,   // size n1*n2
    std::vector<double>& C_row          // size n1*n2
) {
    std::fill(C_row.begin(), C_row.end(), 0.0);

    for (int i = 0; i < n1; ++i) {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            int k = col_ind[p];
            double a = vals[p];
            for (int j = 0; j < n2; ++j) {
                C_row[i * n2 + j] += a * B_row[k * n2 + j];
            }
        }
    }
}

int main(int argc, char* argv[]){ 

    std::string data_path;
    MKL_INT g1_num_nodes, g2_num_nodes;
    bool use_attr = false, directed = false;
    int reps = 1;
    int d = 0;
    if (argc < 2) {
        printf("provide input directory and repetition and d dimension \n");
        return 0;
    }
    else if (argc == 3) {
        data_path = argv[1];
        reps = std::stoi(argv[2]);
    }
    else {
        data_path = argv[1];
        reps = std::stoi(argv[2]);
        d = std::stoi(argv[3]);
    }
    
    load_config(data_path, g1_num_nodes, g2_num_nodes, use_attr, directed);
    printf("MKL threads: %d\n", mkl_get_max_threads());
    printf("MKL dynamic: %d\n", mkl_get_dynamic());
    // printf("MKL threading layer: %d\n", mkl_get_threading_layer());
    printf("OMP max threads: %d\n", omp_get_max_threads());
    printf("OMP in parallel region? %d\n", omp_in_parallel());
    printf("graph1 #nodes %d - directed %d -- use node attributes %d\n\n", g1_num_nodes, directed, use_attr);

    if (d == 0) d = g1_num_nodes;
    sparse_matrix_t graph_csr = nullptr, graph_csc = nullptr;
    CSCStorage g1_csc_storage;
    MKL_INT avg_nnz;
    {
        std::vector<double> g1_edgelist = readFileToVector(data_path + "/graph1.txt", directed);
        printf("edgelist created\n");
        // make_csr_csc_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes, graph_csr, graph_csc);        //row major csr
        // printf("\n");
        // for(int i=0; i< 20; i++) printf("%e ", g1_edgelist[i]);
        // printf("\n");
        // SparseCSRCSCStorage g1_storage;
        // make_csr_csc_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes, graph_csr, graph_csc, g1_storage);
        graph_csr = make_mat_from_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes);
        printf("matrix created\n");
        // graph_csc = make_mat_from_edgelist_csc(g1_edgelist, g1_num_nodes, g1_num_nodes);
        make_mat_from_edgelist_csc(g1_edgelist, g1_num_nodes, g1_num_nodes, graph_csc, g1_csc_storage);
        printf("matrix csc created\n");
        avg_nnz = (int) g1_edgelist.size()/g1_num_nodes;
    }
    printf("all matrices created\n");
    // print_sparse_matrix_info(graph_csr, SparseFormat::CSR, "graph_csr");
    // print_sparse_matrix_info(graph_csc, SparseFormat::CSC, "graph_csc");

    //
    std::vector<double> D_col, D_row;
    int n1 = g1_num_nodes;
    // int n2 = g1_num_nodes;
    int n2 = d;
    generate_random_dense(n1, n2, D_col);
    printf("random dense created\n");
    make_col_and_row_major_from_col_major(D_col, n1, n2, D_row);
    printf("random dense row major created\n");


    // std::vector<double> D_col;
    // int n1 = g1_num_nodes;
    // // int n2 = g1_num_nodes;
    // int n2 = d;
    // generate_random_dense(n1, n2, D_col);
    // printf("random dense created\n");


    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    if (n1 != 0 && n2 > std::numeric_limits<size_t>::max() / n1)
    throw std::overflow_error("n1*n2 overflow");

    size_t N = n1 * n2 ;

    if (N > std::vector<double>().max_size())
    throw std::length_error("N > vector<double>::max_size()");

    std::cout << "n1=" << n1 << " n2=" << n2 << " N=" << N << "\n";
    std::cout << "vector max_size = " << std::vector<double>().max_size() << "\n";
    std::cout << "requested GB = "
          << (long double)N * sizeof(double) / (1024.0L*1024.0L*1024.0L) << "\n";

    // std::vector<double> C5(N);
    // printf("out matrix init\n");

    //warmups
    // for (int r = 0; r < 5; ++r) {
    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, 
    //     //     SPARSE_LAYOUT_COLUMN_MAJOR, D_col.data(), n2, n1, 0.0, C5.data(), n1), "spmm 1 failed");
    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //     //     D_row.data(), n2, n2, 0.0, C5.data(), n2) , "spmm 2 failed");
    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, graph_csr, descr, 
    //     //     SPARSE_LAYOUT_COLUMN_MAJOR, D_col.data(), n2, n1, 0.0, C5.data(), n1), "spmm 1 failed");
    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //     //         D_row.data(), n2, n2, 0.0, C5.data(), n2) , "spmm 2 failed");

    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, 
    //     //     SPARSE_LAYOUT_COLUMN_MAJOR, D_col.data(), n2, n1, 0.0, C5.data(), n1), "spmm 1 failed");
    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //     //     D_col.data(), n2, n2, 0.0, C5.data(), n2) , "spmm 2 failed");
    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, graph_csr, descr, 
    //     //     SPARSE_LAYOUT_COLUMN_MAJOR, D_col.data(), n2, n1, 0.0, C5.data(), n1), "spmm 1 failed");
    //     check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //             D_row.data(), n2, n2, 0.0, C5.data(), n2) , "spmm 2 failed");
    // }
    // C5.clear(); C5.shrink_to_fit();
    // printf("warm up done\n");


    std::vector<double> C1(N);
    // CSR * COL MAJOR
    double total_ms = 0.0;
    for (int r = 0; r < reps; ++r) {
        auto t1 = std::chrono::high_resolution_clock::now();
        check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, 
            SPARSE_LAYOUT_COLUMN_MAJOR, D_col.data(), n2, n1, 0.0, C1.data(), n1), "spmm 1 failed");
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_ms += ms;
    }
    std::cout << "CSR + COL MAJOR" << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";

    C1.clear();  C1.shrink_to_fit();

    // std::vector<double> C2(N);
    // // CSR * ROW MAJOR
    // total_ms = 0.0;
    // for (int r = 0; r < reps; ++r) {
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //     //         D_row.data(), n2, n2, 0.0, C2.data(), n2) , "spmm 2 failed");
    //     check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //             D_col.data(), n2, n2, 0.0, C2.data(), n2) , "spmm 2 failed");
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //     total_ms += ms;
    // }
    // std::cout << "CSR + ROW MAJOR" << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";

    // sparse_index_base_t indexing;
    // MKL_INT rows, cols;
    // MKL_INT *rows_start = nullptr, *rows_end = nullptr;
    // MKL_INT *col_ind = nullptr;
    // double *vals = nullptr;

    // check_mkl(
    //     mkl_sparse_d_export_csr(
    //         graph_csr,
    //         &indexing,
    //         &rows,
    //         &cols,
    //         &rows_start,
    //         &rows_end,
    //         &col_ind,
    //         &vals
    //     ),
    //     "export csr failed"
    // );
    // std::vector<double> C_ref(n1 * n2, 0.0);
    // for (int i = 0; i < n1; ++i) {
    //     for (int p = rows_start[i]; p < rows_end[i]; ++p) {
    //         int k = col_ind[p];
    //         double a = vals[p];
    //         for (int j = 0; j < n2; ++j) {
    //             C_ref[i * n2 + j] += a * D_row[k * n2 + j];
    //         }
    //     }
    // }
    // double max_err = 0.0;
    // for (int i = 0; i < n1 * n2; ++i)
    //     max_err = std::max(max_err, std::abs(C_ref[i] - C2[i]));

    // std::cout << "max abs error = " << max_err << "\n";
    // C2.clear();  C2.shrink_to_fit();
    // std::vector<double> C3(N);

    // // CSR ^ T * COL MAJOR 
    // total_ms = 0.0;
    // for (int r = 0; r < reps; ++r) {
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, graph_csr, descr, 
    //         SPARSE_LAYOUT_COLUMN_MAJOR, D_col.data(), n2, n1, 0.0, C1.data(), n1), "spmm 1 failed");
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //     total_ms += ms;
    // }
    // std::cout << "CSR ^ T * COL MAJOR " << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";

    // D_col.clear();  D_col.shrink_to_fit();
    // C3.clear();  C3.shrink_to_fit();

    std::vector<double> C4(N);
    // CSR ^ T * ROW MAJOR 
    total_ms = 0.0;
    for (int r = 0; r < reps; ++r) {
        auto t1 = std::chrono::high_resolution_clock::now();
        // check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
        //         D_row.data(), n2, n2, 0.0, C2.data(), n2) , "spmm 2 failed");
        check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
                D_row.data(), n2, n2, 0.0, C2.data(), n2) , "spmm 2 failed");
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_ms += ms;
    }
    std::cout << "CSR ^ T * ROW MAJOR " << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";

    
    // // CSC + COL MAJOR
    // total_ms = 0.0;
    // for (int r = 0; r < reps; ++r) {
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csc, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
    //             D_col.data(), n2, n1, 0.0, C3.data(), n1) , "spmm 3 failed");
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //     total_ms += ms;
    // }
    // std::cout << "CSC + COL MAJOR" << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";

    // // CSC + COL MAJOR
    // total_ms = 0.0;
    // for (int r = 0; r < reps; ++r) {
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csc, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //             D_row.data(), n2, n2, 0.0, C4.data(), n2) , "spmm 4 failed");
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //     total_ms += ms;
    // }
    // std::cout << "CSC + COL MAJOR" << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";

}




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
#include "spmm.h"
#include "mtx.h"

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

void print_dense_small (std::vector<double>& dense) {
    printf("\n");
    for( int i=0; i<  100; i++) printf("%e ", dense[i]);
    printf("\n");
}

int main(int argc, char* argv[]){ 

    // std::string data_path;
    // MKL_INT g1_num_nodes, g2_num_nodes;
    // bool use_attr = false, directed = false;
    // int reps = 1;
    // int d = 0;
    // if (argc < 2) {
    //     printf("provide input directory and repetition and d dimension \n");
    //     return 0;
    // }
    // else if (argc == 3) {
    //     data_path = argv[1];
    //     reps = std::stoi(argv[2]);
    // }
    // else {
    //     data_path = argv[1];
    //     reps = std::stoi(argv[2]);
    //     d = std::stoi(argv[3]);
    // }

    std::string mtx_path;
    int reps = 1;
    int d = 0;

    if (argc < 2) {
        printf("provide input directory and repetition and d dimension \n");
        return 0;
    }
    else if (argc == 3) {
        mtx_path = argv[1];
        reps = std::stoi(argv[2]);
    }
    else {
        mtx_path = argv[1];
        reps = std::stoi(argv[2]);
        d = std::stoi(argv[3]);
    }    

    printf("MKL threads: %d\n", mkl_get_max_threads());
    printf("MKL dynamic: %d\n", mkl_get_dynamic());
    printf("OMP max threads: %d\n", omp_get_max_threads());

    std::vector<double> g1_edgelist;
    MKL_INT g1_num_nodes;
    read_mtx(mtx_path, g1_edgelist, g1_num_nodes);
    sparse_matrix_t graph_csr = make_mat_from_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes);
    printf("csr matrix created with %d rows/cols and %d nnz\n", g1_num_nodes, g1_edgelist.size() / 2);
    g1_edgelist.clear(); g1_edgelist.shrink_to_fit();
 
    std::vector<double> D_row;
    int n1 = g1_num_nodes;
    int n2 = d;

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    if (n1 != 0 && n2 > std::numeric_limits<size_t>::max() / n1)
    throw std::overflow_error("n1*n2 overflow");
    size_t N = (size_t) n1 * (size_t) n2 ;

    if (N > std::vector<double>().max_size())   throw std::length_error("N > vector<double>::max_size()");
    std::cout << "n1=" << n1 << " n2=" << n2 << " N=" << N << "\n";
    std::cout << "vector max_size = " << std::vector<double>().max_size() << "\n";
    std::cout << "requested GB = "
          << (long double)N * sizeof(double) / (1024.0L*1024.0L*1024.0L) << "\n";


    generate_random_dense(n1, n2, D_row);
    // std::vector<double> C5(N);
    // printf("out matrix init\n");

    // //warmups
    // for (int r = 0; r < 5; ++r) {\
    //     check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
    //         D_col.data(), n2, n2, 0.0, C5.data(), n2) , "spmm 2 failed");
    // }
    // C5.clear(); C5.shrink_to_fit();
    // printf("warm up done\n");

    printf("-------------------------------------------------------\n");
    std::vector<double> C1(N);
    // // CSR * ROW MAJOR 
    double total_ms = 0.0;
    for (int r = 0; r < reps; ++r) {
        auto t1 = std::chrono::high_resolution_clock::now();
        check_mkl( mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph_csr, descr, SPARSE_LAYOUT_ROW_MAJOR,
                D_row.data(), n2, n2, 0.0, C1.data(), n2) , "spmm 2 failed");
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_ms += ms;
    }
    std::cout << "mkl spmm (CSR+Row-maj Dense) average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
    C1.clear();  C1.shrink_to_fit();

    printf("-------------------------------------------------------\n");
    std::vector<double> C2(N);
    // CSR * ROW MAJOR
    total_ms = 0.0;
    for (int r = 0; r < reps; ++r) {
        auto t1 = std::chrono::high_resolution_clock::now();
        omp_spmm(1.0, graph_csr, n1, n2, 0.0, D_row.data(), C2.data());
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_ms += ms;
    }
    std::cout << "omp spmm (CSR+Row-maj Dense) average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
    C2.clear();  C2.shrink_to_fit();

    printf("-------------------------------------------------------\n");
    std::vector<double> C3(N);
    // CSR * ROW MAJOR
    for (int bc : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
        total_ms = 0.0;
        for (int r = 0; r < reps; ++r) {
            auto t1 = std::chrono::high_resolution_clock::now();
            omp_spmm_jik_v1(1.0, graph_csr, n1, n2, 0.0, D_row.data(), C3.data(), bc);
            auto t2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            total_ms += ms;
        }
        std::cout << "omp B/C col blocked with bc " << bc << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
    }
    C3.clear();  C3.shrink_to_fit();
    // compare_dense(C2, C3, "compare results");

    printf("-------------------------------------------------------\n");
    std::vector<double> C7(N);
    // CSR * ROW MAJOR
    for (int bc : {16, 32, 64, 128, 256, 512}) {
        for (int bk : {2, 4, 8}) {
            total_ms = 0.0;
            for (int r = 0; r < reps; ++r) {
                auto t1 = std::chrono::high_resolution_clock::now();
                omp_spmm_jik_blockedcsr(1.0, graph_csr, n1, n2, 0.0, D_row.data(), C7.data(), bc, (int) n1 / bk);
                auto t2 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_ms += ms;
            }
            std::cout << "omp csr col blocked with bk " << bk << " bc " << bc << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
        }
    }
    C7.clear();  C7.shrink_to_fit();

    printf("-------------------------------------------------------\n");
    std::vector<double> C8(N);
    // CSR * ROW MAJOR
    for (int bc : {128, 256, 512, 1024, 2048}) {
        for (int bk : {2, 4, 8}) {
            total_ms = 0.0;
            for (int r = 0; r < reps; ++r) {
                auto t1 = std::chrono::high_resolution_clock::now();
                omp_spmm_rik(1.0, graph_csr, n1, n2, 0.0, D_row.data(), C8.data(), (int) n1 / bk, bc);
                auto t2 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_ms += ms;
            }
            std::cout << "omp csr row blocked with bk " << bk << " bc " << bc << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
        }
    }
    C8.clear();  C8.shrink_to_fit();
    // compare_dense(C2, C8, "compare results");

    // printf("-------------------------------------------------------\n");
    // std::vector<double> C9(N);
    // // CSR * ROW MAJOR
    // for (int bc : {128, 256, 512, 1024, 2048}) {
    //     for (int bk : {2, 4, 8, 16, 32, 64}) {
    //         total_ms = 0.0;
    //         for (int r = 0; r < reps; ++r) {
    //             auto t1 = std::chrono::high_resolution_clock::now();
    //             omp_spmm_rik_v2(1.0, graph_csr, n1, n2, 0.0, D_row.data(), C9.data(), (int) n1 / bk, bc);
    //             auto t2 = std::chrono::high_resolution_clock::now();
    //             double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //             total_ms += ms;
    //         }
    //         std::cout << "omp csr row blocked with bk " << bk << " bc " << bc << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
    //     }
    // }
    // C9.clear();  C9.shrink_to_fit();

 
    // Quick sweep to find best b_c:
    // for (int bc : {32, 64, 128, 256, 512, 1024}) {
    //     total_ms = 0.0;
    //     for (int r = 0; r < reps; ++r) {
    //         auto t1 = std::chrono::high_resolution_clock::now();
    //         omp_spmm_jik(1.0, graph_csr, n1, n2, 0.0, D_col.data(), C3.data(), bc, 48);
    //         auto t2 = std::chrono::high_resolution_clock::now();
    //         double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //         total_ms += ms;
    //     }
    //     std::cout << "omp blocked b_c= " << bc << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
    //     // compare_dense(C2, C3, "compare results");
    // }

    // compare_dense(C2, C1, "compare results");
    // compare_dense(C2, C3, "compare results");
    

    // std::vector<double> C4(N);
    // CSR * ROW MAJOR
    // total_ms = 0.0;
    // for (int r = 0; r < reps; ++r) {
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     // omp_spmm_kblock(1.0, graph_csr, n1, n2, 0.0, D_col.data(), C4.data(), 128, 48);
    //     omp_spmm_2d_tiled(1.0, graph_csr, n1, n2, 0.0, D_col.data(), C4.data(), 64, 64, 48);
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //     total_ms += ms;
    // }
    // std::cout << "omp row blocked" << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
    // C4.clear();  C4.shrink_to_fit();
    // compare_dense(C2, C4, "compare results");

    // for (int br : {32, 64, 128}) {
    //     for (int bk : {64, 128, 256, 512}) {
    //         total_ms = 0.0;
    //         for (int r = 0; r < reps; ++r) {
    //             auto t1 = std::chrono::high_resolution_clock::now();
    //             // omp_spmm_kblock(1.0, graph_csr, n1, n2, 0.0, D_col.data(), C4.data(), 128, 48);
    //             omp_spmm_2d_tiled(1.0, graph_csr, n1, n2, 0.0, D_col.data(), C4.data(), br, bk, 48);
    //             auto t2 = std::chrono::high_resolution_clock::now();
    //             double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    //             total_ms += ms;
    //         }
    //         std::cout << "omp row and col blocked br " << br << " bc " << bk << " average over " << reps << " runs: " << (total_ms / reps) << " ms\n";
    //     }
    // }

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
  
}




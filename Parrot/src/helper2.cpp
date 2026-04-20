// #include <mkl.h>
// #include <cmath>
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <random>
// #include <unordered_set>
// #include <stdexcept>
// #include <omp.h>
// #include <sstream>
// #include <cstdio>
// #include <mkl_spblas.h>
// // #include <metis.h>
// #include <cstdint>
// #include <limits>
// #include <numeric>
// #include <utility>
#include "helper2.h"

bool has_nan(const std::vector<double>& v) {
    for (double x : v) {
        if (std::isnan(x)) return true;
    }
    return false;
}

bool has_nan_or_inf(const std::vector<double>& v) {
    for (double x : v)
        if (std::isnan(x) || std::isinf(x)) return true;
    return false;
}

bool sparse_has_nan_or_inf(sparse_matrix_t A) {
    sparse_index_base_t indexing; 
    MKL_INT rows, cols; 
    MKL_INT *rows_start = nullptr, *rows_end = nullptr, *col_indx = nullptr;
    double *values = nullptr;

    sparse_status_t status = mkl_sparse_d_export_csr(A, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);
    if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_export_csr failed");
    if (values == nullptr) return false;
    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT idx = rows_start[i]; idx < rows_end[i]; ++idx) {
            double x = values[idx];
            if (std::isnan(x) || std::isinf(x)) return true;
        }
    }
    return false;
}

double sparse_max_abs(sparse_matrix_t A) {
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *rows_start = nullptr, *rows_end = nullptr, *col_indx = nullptr;
    double *values = nullptr;

    sparse_status_t status = mkl_sparse_d_export_csr(
        A, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_export_csr failed");
    }

    double mx = 0.0;
    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT k = rows_start[i]; k < rows_end[i]; ++k) {
            double x = values[k];
            if (std::isnan(x) || std::isinf(x)) continue;
            mx = std::max(mx, std::abs(x));
        }
    }
    return mx;
}

void sparse_print_large_values(sparse_matrix_t A, double threshold) {
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *rows_start = nullptr, *rows_end = nullptr, *col_indx = nullptr;
    double *values = nullptr;

    sparse_status_t status = mkl_sparse_d_export_csr(
        A, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_d_export_csr failed");
    }

    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT k = rows_start[i]; k < rows_end[i]; ++k) {
            double x = values[k];
            if (!std::isnan(x) && !std::isinf(x) && std::abs(x) > threshold) {
                std::cout << "sparse large value at row " << i
                          << " col " << col_indx[k]
                          << " : " << x << "\n";
            }
        }
    }
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

void load_config( const std::string& data_path, MKL_INT& g1_num_nodes, MKL_INT& g2_num_nodes, bool& use_attr, bool& directed, int& inner_iters, int& outer_iters) {
    std::ifstream fin(data_path + "/config.txt");
    if(!fin) {
        throw std::runtime_error("cannot open config.txt");
    }
    std::string line;
    while ( std::getline(fin, line) ) {
        std::istringstream iss(line);
        std::string key, eq;
        if (std::getline(iss, key, '=') && std::getline(iss, eq)) {
            if (key == "g1_num_nodes") g1_num_nodes = std::stoi(eq);
            else if (key == "g2_num_nodes") g2_num_nodes = std::stoi(eq);
            else if (key == "use_attr") use_attr = (eq == "1");
            else if (key == "directed") directed = (eq == "1");
            else if (key == "outer_iter") outer_iters = std::stoi(eq);
            else if (key == "inner_iter") inner_iters = std::stoi(eq);
        }
    }
}

std::vector<double> readFileToVector(const std::string &filename, bool directed) {
    std::ifstream fin(filename);
    std::vector<double> data;

    if(directed) {
        double val;
        while (fin >> val) {
            data.push_back(val);
        }
    }
    if(!directed) {
        double u, v;
        while (fin >> u >> v) {
            // original edge
            data.push_back(u);
            data.push_back(v);

            // reverse edge
            if (u != v) {          // avoid duplicating self-loops
                data.push_back(v);
                data.push_back(u);
            }
        }
    }
    return data;
}


bool compare_sparse_exact( sparse_matrix_t A, sparse_matrix_t B, double eps) {
    // Export A
    sparse_index_base_t idxA, idxB;
    MKL_INT rA, cA, rB, cB;
    MKL_INT *rsA, *reA, *ciA;
    MKL_INT *rsB, *reB, *ciB;
    double *valA, *valB;

    auto st = mkl_sparse_d_export_csr(
        A, &idxA, &rA, &cA, &rsA, &reA, &ciA, &valA);
    if (st != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Export A failed\n";
        return false;
    }

    st = mkl_sparse_d_export_csr(
        B, &idxB, &rB, &cB, &rsB, &reB, &ciB, &valB);
    if (st != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Export B failed\n";
        return false;
    }

    // Dimension check
    if (rA != rB || cA != cB) {
        std::cerr << "Dimension mismatch\n";
        return false;
    }

    // Index base check
    if (idxA != idxB) {
        std::cerr << "Index base mismatch\n";
        return false;
    }

    // Compare row pointers
    for (MKL_INT i = 0; i < rA; ++i) {
        if (rsA[i] != rsB[i] || reA[i] != reB[i]) {
            std::cerr << "Row pointer mismatch at row " << i << "\n";
            return false;
        }
    }

    // Compare column indices and values
    for (MKL_INT i = 0; i < rA; ++i) {
        for (MKL_INT p = rsA[i]; p < reA[i]; ++p) {

            if (ciA[p] != ciB[p]) {
                std::cerr << "Column index mismatch at row " << i << "\n";
                return false;
            }

            if (std::fabs(valA[p] - valB[p]) > eps) {
                std::cerr << "Value mismatch at row " << i
                          << " col " << ciA[p]
                          << " A=" << valA[p]
                          << " B=" << valB[p] << "\n";
                return false;
            }
        }
    }

    return true;
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

// sparse_matrix_t make_mat_from_edgelist_symmetric(const std::vector<double>& edgelist, int m, int n) {
//     MKL_INT nnz = 0;
//     std::vector<MKL_INT> rows, cols;
//     std::vector<double> vals;
//     rows.reserve(edgelist.size()*2);
//     cols.reserve(edgelist.size()*2);
//     vals.reserve(edgelist.size()*2);
// 	// std::vector<MKL_INT> rows(edgelist.size()), cols(edgelist.size());
// 	// std::vector<double> vals(edgelist.size());
// 	for(int i=0; i < edgelist / 2; i++) {
//         vals.push_back(1.0);
//         rows.push_back((MKL_INT) edgelist[2*i]);
//         cols.push_back((MKL_INT) edgelist[2*i+1]);
//         nnz++;
//         if (edgelist[2*i] != edgelist[2*i+1]) {
//             vals.push_back(1.0);
//             rows.push_back((MKL_INT) edgelist[2*i+1]);
//             cols.push_back((MKL_INT) edgelist[2*i]);
//             nnz++;
//         }
//     }
//     rows.resize(nnz);
//     cols.resize(nnz);
//     vals.resize(nnz);

// 	sparse_matrix_t A = nullptr;
//     auto st = mkl_sparse_d_create_coo(&A, SPARSE_INDEX_BASE_ZERO, m, n, nnz, rows.data(), cols.data(), vals.data());
//     if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("create_coo failed");
//     mkl_sparse_optimize(A);

//     sparse_matrix_t A_csr;
//     mkl_sparse_convert_csr(A, SPARSE_OPERATION_NON_TRANSPOSE, &A_csr);
    
//     mkl_sparse_destroy(A);

//     return A_csr;
// }

void print_sparse_matrix_info(sparse_matrix_t A, const char* name) {
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

void print_Cost_vals (const std::vector<double>& Cost) {
    int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0, cnt5 = 0, cnt6 = 0;
    double max_cost = Cost[0], min_cost = Cost[0];
    for(int i=0; i<Cost.size(); i++) {
        double val = Cost[i];
        min_cost = std::min(min_cost, val);
        max_cost = std::max(max_cost, val);
        // printf(" %e ", Cost[j*n1+i]);
        if(val >= 4.0)
            cnt1++;
        else if(val < 4.0 && val >= 3.0)
            cnt2++;
        else if(val < 3.0 && val >= 2.0)
            cnt3++;
        else if(val < 2.0 && val >= 1.0)
            cnt4++;
        else if(val < 1.0 && val >= 0.0)
            cnt5++;
        else   
            cnt6++;
    }
    printf("\nCost min %e max %e\n", min_cost, max_cost);
    printf("Cost matrix total %d -  # %d > 4.0 ---- \n # 4.0 > %d > 3.0 ----  \n # 3 > %d > 2 ---- \n # 2 > %d > 1 ---- \n # 1 > %d > 0 ---- \n # %d < 0 \n", 
    Cost.size(), cnt1, cnt2, cnt3, cnt4, cnt5, cnt6);
    return;
}


double percentile(std::vector<double> S, double p) {
    if (S.empty()) {
        throw std::runtime_error("Vector is empty");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::runtime_error("Percentile p must be between 0 and 1");
    }

    std::sort(S.begin(), S.end());

    size_t idx = static_cast<size_t>(p * (S.size() - 1));
    return S[idx];
}


void print_stats(const std::vector<double>& S) {
    if (S.empty()) {
        std::cout << "Vector is empty\n";
        return;
    }

    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    int64_t nz = 0;

    // First pass: min, max, sum
    for (double v : S) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        if (v == 0.0)   nz++;
        sum += v;
    }

    double mean = sum / S.size();

    // Second pass: variance
    double sq_sum = 0.0;
    for (double v : S) {
        double diff = v - mean;
        sq_sum += diff * diff;
    }

    double std = std::sqrt(sq_sum / S.size());

    std::cout << "size of vec : " << S.size() << "\n";
    std::cout << "num of zeros : " << nz << "\n";
    std::cout << "min: " << min_val << "\n";
    std::cout << "max: " << max_val << "\n";
    std::cout << "mean: " << mean << "\n";
    std::cout << "std: " << std << "\n";
    std::cout << "max/mean: " << (mean != 0.0 ? max_val / mean : 0.0) << "\n";

    std::cout << "90th percentile threshold: " << percentile(S, 0.90) << "\n";
    std::cout << "95th percentile threshold: " << percentile(S, 0.95) << "\n";
    std::cout << "99th percentile threshold: " << percentile(S, 0.99) << "\n";
}

void sparsify_vector(std::vector<double>& S, double threshold) {
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for
    for (double& v : S) {
        if (v < threshold) {
            v = 0.0;
        }
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
}

void sparsify_vector_reverse(std::vector<double>& S, double threshold) {
    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);
    mkl_set_num_threads(1);
    #pragma omp parallel for
    for (double& v : S) {
        if (v > threshold) {
            v = 0.0;
        }
    }
    omp_set_num_threads(1);
    mkl_set_num_threads(numThreads);
}

void graph_connect_isolated(sparse_matrix_t graph, std::vector<double>& graph_row_sum, sparse_matrix_t& graph_connected) {
    
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start = nullptr, *row_end = nullptr;
    MKL_INT *col_indx = nullptr;
    double *values = nullptr;
    sparse_status_t st = mkl_sparse_d_export_csr( graph, &indexing, &rows, &cols, &row_start, &row_end, &col_indx, &values);
    if (st != SPARSE_STATUS_SUCCESS)    throw std::runtime_error("mkl_sparse_d_export_csr failed");

    MKL_INT csr_offset = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    MKL_INT total_nnz = row_end[rows - 1] - csr_offset;
    MKL_INT avg_nnz_per_n = total_nnz / rows;
    std::vector<MKL_INT> isolated;
    bool isolated_node = false ;
    for (MKL_INT i = 0; i < rows; ++i) {
        graph_row_sum[i] = row_end[i] - row_start[i];
        if (row_start[i] == row_end[i]) {
            isolated.push_back(i);
            isolated_node = true;
        }   
    }
    
    printf("in graph_connect_isolated function num of isolated %ld and avg_nnz_per_n is %d\n", isolated.size(), avg_nnz_per_n);
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
                MKL_INT random_num = std::rand() % cols;
                if ( random_nodes.find(random_num) !=  random_nodes.end() || (random_num == i)) continue;
                if ( random_num >= rows )          std::cerr << "random num generated out of bound\n" << std::endl; 
                random_nodes.insert(random_num);
                edge_list.push_back(static_cast<double>(i));
                edge_list.push_back(static_cast<double>(random_num));
            }
            // update row sum directly
            graph_row_sum[i] = static_cast<MKL_INT>(avg_nnz_per_n);
            // printf("isolated node %d connected to %d nodes\n", i, edge_list.size());
        }
        printf("connect isolated nodes : num fo edges to add %d\n", edge_list.size()/2);
        // build CSR matrix (values are implicitly 1.0 inside)
        sparse_matrix_t add_mat = make_mat_from_edgelist(edge_list, rows, rows);
        //add add_mat to original graph
        auto status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, graph, 1.0, add_mat, &graph_connected);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_add failed in graph_connect_isolated");

        mkl_sparse_destroy(add_mat);
        // graph = graph_connected;
    }
    else {
        struct matrix_descr descr{};
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_status_t status = mkl_sparse_copy(graph, descr, &graph_connected);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_copy failed in graph_connect_isolated");
    }
    for(int i=0; i<graph_row_sum.size(); i++) {
        if(graph_row_sum[i] == 0)   printf("error graph_row_sum[%d] is 0\n", i);
    }
        
}


// namespace {

// inline void check_sparse(sparse_status_t st, const char* msg) {
//     if (st != SPARSE_STATUS_SUCCESS) {
//         throw std::runtime_error(msg);
//     }
// }

// // Deep-copy an MKL sparse matrix handle.
// sparse_matrix_t deep_copy_sparse(sparse_matrix_t A) {
//     sparse_matrix_t B = nullptr;
//     matrix_descr descr{};
//     descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//     check_sparse(mkl_sparse_copy(A, descr, &B), "mkl_sparse_copy failed");
//     return B;
// }

// // Build a unit-valued CSR matrix from an edge list and return an independent MKL sparse handle.
// sparse_matrix_t build_unit_csr_from_edges(
//     const std::vector<std::pair<MKL_INT, MKL_INT>>& edges_in,
//     MKL_INT nrows,
//     MKL_INT ncols)
// {
//     if (edges_in.empty()) {
//         return nullptr;
//     }

//     std::vector<std::pair<MKL_INT, MKL_INT>> edges = edges_in;
//     std::sort(edges.begin(), edges.end());
//     edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

//     std::vector<MKL_INT> row_counts(nrows, 0);
//     for (const auto& e : edges) {
//         if (e.first < 0 || e.first >= nrows || e.second < 0 || e.second >= ncols) {
//             throw std::runtime_error("edge index out of bounds in build_unit_csr_from_edges");
//         }
//         row_counts[e.first]++;
//     }

//     std::vector<MKL_INT> row_start(nrows), row_end(nrows);
//     std::vector<MKL_INT> col_indx(edges.size());
//     std::vector<double> values(edges.size(), 1.0);

//     MKL_INT nnz_ptr = 0;
//     for (MKL_INT i = 0; i < nrows; ++i) {
//         row_start[i] = nnz_ptr;
//         nnz_ptr += row_counts[i];
//         row_end[i] = nnz_ptr;
//     }

//     std::vector<MKL_INT> next(row_start.begin(), row_start.end());
//     for (const auto& e : edges) {
//         col_indx[next[e.first]++] = e.second;
//     }

//     sparse_matrix_t tmp = nullptr;
//     check_sparse(
//         mkl_sparse_d_create_csr(&tmp,
//                                 SPARSE_INDEX_BASE_ZERO,
//                                 nrows,
//                                 ncols,
//                                 row_start.data(),
//                                 row_end.data(),
//                                 col_indx.data(),
//                                 values.data()),
//         "mkl_sparse_d_create_csr failed in build_unit_csr_from_edges");

//     sparse_matrix_t out = deep_copy_sparse(tmp);
//     mkl_sparse_destroy(tmp);
//     return out;
// }

// } // namespace


// Assumes graph is an adjacency matrix.
// Row-normalization here is degree-based for an unweighted graph:
// every nonzero in row i becomes 1.0 / nnz(row i).
// sparse_matrix_t graph_row_normalize_connect_isolated(sparse_matrix_t graph, unsigned int seed ) {
//     if (graph == nullptr) {
//         throw std::runtime_error("graph is null");
//     }

//     sparse_index_base_t indexing;
//     MKL_INT rows, cols;
//     MKL_INT* row_start = nullptr;
//     MKL_INT* row_end   = nullptr;
//     MKL_INT* col_indx  = nullptr;
//     double*  values    = nullptr;

//     check_sparse(
//         mkl_sparse_d_export_csr(graph, &indexing, &rows, &cols,
//                                 &row_start, &row_end, &col_indx, &values),
//         "mkl_sparse_d_export_csr failed");

//     if (rows <= 0 || cols <= 0) {
//         throw std::runtime_error("graph has invalid dimensions");
//     }
//     if (rows != cols) {
//         throw std::runtime_error("graph_row_normalize_connect_isolated expects a square adjacency matrix");
//     }

//     const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
//     const MKL_INT total_nnz = row_end[rows - 1] - base;
//     const MKL_INT avg_nnz_per_row = (rows > 0) ? (total_nnz / rows) : 0;

//     // Compute row nnz and isolated flags in parallel.
//     std::vector<MKL_INT> row_nnz(rows, 0);
//     std::vector<char> is_isolated(rows, 0);

//     #pragma omp parallel for schedule(static)
//     for (MKL_INT i = 0; i < rows; ++i) {
//         MKL_INT deg = row_end[i] - row_start[i];
//         row_nnz[i] = deg;
//         is_isolated[i] = (deg == 0) ? 1 : 0;
//     }

//     std::vector<MKL_INT> isolated;
//     isolated.reserve(rows);
//     for (MKL_INT i = 0; i < rows; ++i) {
//         if (is_isolated[i]) isolated.push_back(i);
//     }

//     sparse_matrix_t connected_graph = nullptr;

//     if (isolated.empty()) {
//         connected_graph = deep_copy_sparse(graph);
//     } else {
//         // Same spirit as your earlier code:
//         // each isolated node gets connected to about avg_nnz_per_row random distinct nodes.
//         MKL_INT target_degree = std::max<MKL_INT>(1, avg_nnz_per_row);

//         std::vector<std::pair<MKL_INT, MKL_INT>> new_edges;
//         if (rows == 1) {
//             // Only possible connection is a self-loop.
//             new_edges.emplace_back(0, 0);
//         } else {
//             target_degree = std::min<MKL_INT>(target_degree, rows - 1);
//             new_edges.reserve(static_cast<size_t>(isolated.size()) * static_cast<size_t>(target_degree));

//             std::mt19937 rng(seed);

//             for (MKL_INT i : isolated) {
//                 std::unordered_set<MKL_INT> nbrs;
//                 nbrs.reserve(static_cast<size_t>(target_degree) * 2);

//                 // Sample distinct random neighbors, excluding self.
//                 std::uniform_int_distribution<MKL_INT> dist(0, rows - 2);

//                 while (static_cast<MKL_INT>(nbrs.size()) < target_degree) {
//                     MKL_INT j = dist(rng);
//                     if (j >= i) ++j;   // maps [0, rows-2] -> [0, rows-1] \ {i}
//                     nbrs.insert(j);
//                 }

//                 for (MKL_INT j : nbrs) {
//                     new_edges.emplace_back(i, j);
//                 }
//             }
//         }

//         sparse_matrix_t add_mat = build_unit_csr_from_edges(new_edges, rows, cols);
//         check_sparse(
//             mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, graph, 1.0, add_mat, &connected_graph),
//             "mkl_sparse_d_add failed");
//         if (add_mat) mkl_sparse_destroy(add_mat);
//     }

//     check_sparse(mkl_sparse_order(connected_graph), "mkl_sparse_order failed on connected_graph");

//     // Export connected graph and build row-normalized CSR.
//     sparse_index_base_t indexing2;
//     MKL_INT rows2, cols2;
//     MKL_INT* row_start2 = nullptr;
//     MKL_INT* row_end2   = nullptr;
//     MKL_INT* col_indx2  = nullptr;
//     double*  values2    = nullptr;

//     check_sparse(
//         mkl_sparse_d_export_csr(connected_graph, &indexing2, &rows2, &cols2,
//                                 &row_start2, &row_end2, &col_indx2, &values2),
//         "mkl_sparse_d_export_csr failed on connected_graph");

//     const MKL_INT base2 = (indexing2 == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
//     const MKL_INT connected_nnz = row_end2[rows2 - 1] - base2;

//     std::vector<MKL_INT> out_row_start(rows2);
//     std::vector<MKL_INT> out_row_end(rows2);
//     std::vector<MKL_INT> out_col_indx(connected_nnz);
//     std::vector<double>  out_values(connected_nnz);

//     #pragma omp parallel for schedule(static)
//     for (MKL_INT i = 0; i < rows2; ++i) {
//         MKL_INT begin = row_start2[i] - base2;
//         MKL_INT end   = row_end2[i]   - base2;
//         MKL_INT deg   = end - begin;

//         out_row_start[i] = begin;
//         out_row_end[i]   = end;

//         double inv_deg = (deg > 0) ? (1.0 / static_cast<double>(deg)) : 0.0;

//         for (MKL_INT p = begin; p < end; ++p) {
//             out_col_indx[p] = col_indx2[p] - base2;
//             out_values[p]   = inv_deg;
//         }
//     }

//     sparse_matrix_t tmp_norm = nullptr;
//     check_sparse(
//         mkl_sparse_d_create_csr(&tmp_norm,
//                                 SPARSE_INDEX_BASE_ZERO,
//                                 rows2,
//                                 cols2,
//                                 out_row_start.data(),
//                                 out_row_end.data(),
//                                 out_col_indx.data(),
//                                 out_values.data()),
//         "mkl_sparse_d_create_csr failed for normalized graph");

//     sparse_matrix_t graph_row_normalized = deep_copy_sparse(tmp_norm);

//     mkl_sparse_destroy(tmp_norm);
//     mkl_sparse_destroy(connected_graph);

//     check_sparse(mkl_sparse_order(graph_row_normalized), "mkl_sparse_order failed on graph_row_normalized");
//     return graph_row_normalized;
// }

// Drop-in blocked replacement for mkl_sparse_d_mm
// Assumes dense matrices are column-major.
// Internally splits columns into blocks and calls mkl_sparse_d_mm on each block.
//
// Parameters match mkl_sparse_d_mm:
//
// operation        : sparse op flag
// alpha            : scalar
// A                : sparse matrix (CSR handle typically)
// descr            : matrix descriptor
// layout           : must be SPARSE_LAYOUT_COLUMN_MAJOR here
// B                : input dense matrix
// columns          : number of columns of B and C
// ldb              : leading dimension of B
// beta             : scalar
// C                : output dense matrix
// ldc              : leading dimension of C
//
// Optional block_size has a default. You can tune 32/64/128/256/512.
//
inline sparse_status_t blocked_mkl_sparse_d_mm( sparse_operation_t operation, double alpha, sparse_matrix_t A, matrix_descr descr, sparse_layout_t layout, const double* B, 
    MKL_INT columns, MKL_INT ldb, double beta, double* C, MKL_INT ldc, MKL_INT block_size  ) {
    if (layout != SPARSE_LAYOUT_COLUMN_MAJOR)   return SPARSE_STATUS_NOT_SUPPORTED;
    if (columns <= 0 || ldb <= 0 || ldc <= 0 || block_size <= 0)  return SPARSE_STATUS_INVALID_VALUE;
    sparse_status_t st = SPARSE_STATUS_SUCCESS;

    // For C(:, j0:j1), beta should be the user beta.
    // If a given output block were accumulated multiple times, later calls would use beta=1.
    // But here each block is written exactly once, so user beta is correct for every block.
    for (MKL_INT col_start = 0; col_start < columns; col_start += block_size) {
        const MKL_INT jb = std::min(block_size, columns - col_start);
        // Column-major offset: pointer to first element of column col_start
        const double* B_block = B + static_cast<size_t>(col_start) * ldb;
        double*       C_block = C + static_cast<size_t>(col_start) * ldc;
        st = mkl_sparse_d_mm( operation, alpha, A, descr, layout, B_block, jb, ldb, beta, C_block, ldc);
        if (st != SPARSE_STATUS_SUCCESS) return st;
    }
    return SPARSE_STATUS_SUCCESS;
}

inline void prepare_sparse_mm_once( sparse_matrix_t A, matrix_descr descr, MKL_INT typical_block_cols, MKL_INT expected_calls  ) {
    auto st = mkl_sparse_set_mm_hint( A, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_COLUMN_MAJOR, typical_block_cols, expected_calls);
    if (st != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_set_mm_hint failed");
    }
    st = mkl_sparse_optimize(A);
    if (st != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("mkl_sparse_optimize failed");
    }
}

void daxpy_chunked(size_t N, double alpha, const double* x, double* y) {
    const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
    size_t off = 0;
    while (off < N) {
        size_t len = std::min(CHUNK, N - off);
        cblas_daxpy((MKL_INT)len, alpha, x + off, 1, y + off, 1);
        off += len;
    }
}

void dscal_chunked(size_t N, double alpha, double* x) {
    const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
    size_t off = 0;
    while (off < N) {
        size_t len = std::min(CHUNK, N - off);
        cblas_dscal((MKL_INT)len, alpha, x + off, 1);
        off += len;
    }
}

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
                DEBUG_ONLY(std::cout << "sparse large value at row " << i
                                     << " col " << col_indx[k]
                                     << " : " << x << "\n");
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

template <typename Real>
std::vector<Real> readFileToVector(const std::string &filename, bool directed) {
    std::ifstream fin(filename);
    std::vector<Real> data;

    if(directed) {
        Real val;
        while (fin >> val) {
            data.push_back(val);
        }
    }
    if(!directed) {
        Real u, v;
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

template std::vector<double> readFileToVector<double>(const std::string &filename, bool directed);
template std::vector<float> readFileToVector<float>(const std::string &filename, bool directed);


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


template <typename Real>
sparse_matrix_t make_mat_from_edgelist(const std::vector<Real>& edgelist, int m, int n) {
    MKL_INT nnz  = edgelist.size() / 2;
	std::vector<MKL_INT> rows(nnz), cols(nnz);
	std::vector<Real> vals(nnz);
	for(int i=0; i < nnz; i++) {
        vals[i] = Real(1);
        rows[i] = (MKL_INT) edgelist[2*i];
        cols[i] = (MKL_INT) edgelist[2*i+1];	
    }

	sparse_matrix_t A = nullptr;
    sparse_status_t st;
    if constexpr (std::is_same_v<Real, float>) {
        st = mkl_sparse_s_create_coo(&A, SPARSE_INDEX_BASE_ZERO, m, n, nnz, rows.data(), cols.data(), vals.data());
    } else {
        st = mkl_sparse_d_create_coo(&A, SPARSE_INDEX_BASE_ZERO, m, n, nnz, rows.data(), cols.data(), vals.data());
    }
    if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("create_coo failed");
    mkl_sparse_optimize(A);

    sparse_matrix_t A_csr;
    mkl_sparse_convert_csr(A, SPARSE_OPERATION_NON_TRANSPOSE, &A_csr);
    
    mkl_sparse_destroy(A);

    return A_csr;
}

template sparse_matrix_t make_mat_from_edgelist<double>(const std::vector<double>& edgelist, int m, int n);
template sparse_matrix_t make_mat_from_edgelist<float>(const std::vector<float>& edgelist, int m, int n);

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
#ifndef DEBUG
    (void)A;
    (void)name;
    return;
#else
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
#endif
}

void print_num_zeros(std::vector<double>  M, const char* name ) {
#ifndef DEBUG
    (void)M;
    (void)name;
    return;
#else
    int counter = 0;
    for(int i=0; i< M.size(); i++) {
        if(M[i] == (double) 0.0)    counter++;
    }
    printf(" %s has %d zeros out of %d values\n", name, counter, M.size());
    return ;
#endif
}

void compare( const std::vector<double>& cpp_vec, const std::string& filename, const std::string& label) {
#ifndef DEBUG
    (void)cpp_vec;
    (void)filename;
    (void)label;
    return;
#else
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
#endif
}

void print_Cost_vals (const std::vector<double>& Cost) {
#ifndef DEBUG
    (void)Cost;
    return;
#else
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
#endif
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
#ifndef DEBUG
    (void)S;
    return;
#else
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
#endif
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
    
    DEBUG_ONLY(printf("in graph_connect_isolated function num of isolated %ld and avg_nnz_per_n is %d\n", isolated.size(), avg_nnz_per_n));
    if(isolated_node) {
        DEBUG_ONLY(printf("connecting isolated nodes\n"));
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
        DEBUG_ONLY(printf("connect isolated nodes : num fo edges to add %d\n", edge_list.size()/2));
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
        if(graph_row_sum[i] == 0) DEBUG_ONLY(printf("error graph_row_sum[%d] is 0\n", i));
    }
        
}

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

template <typename Real>
void daxpy_chunked(size_t N, Real alpha, const Real* x, Real* y) {
    const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
    size_t off = 0;
    while (off < N) {
        size_t len = std::min(CHUNK, N - off);
        if constexpr (std::is_same_v<Real, float>) {
            cblas_saxpy((MKL_INT)len, alpha, x + off, 1, y + off, 1);
        } else {
            cblas_daxpy((MKL_INT)len, alpha, x + off, 1, y + off, 1);
        }
        off += len;
    }
}

template void daxpy_chunked<double>(size_t N, double alpha, const double* x, double* y);
template void daxpy_chunked<float>(size_t N, float alpha, const float* x, float* y);

template <typename Real>
void dscal_chunked(size_t N, Real alpha, Real* x) {
    const size_t CHUNK = (size_t)std::numeric_limits<MKL_INT>::max();
    size_t off = 0;
    while (off < N) {
        size_t len = std::min(CHUNK, N - off);
        if constexpr (std::is_same_v<Real, float>) {
            cblas_sscal((MKL_INT)len, alpha, x + off, 1);
        } else {
            cblas_dscal((MKL_INT)len, alpha, x + off, 1);
        }
        off += len;
    }
}

template void dscal_chunked<double>(size_t N, double alpha, double* x);
template void dscal_chunked<float>(size_t N, float alpha, float* x);

#include <mkl.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <stdexcept>
#include <omp.h>
#include <cstdio>
#include <mkl_spblas.h>
// #include <metis.h>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <cinttypes> 
#include <set>

void make_col_and_row_major_from_col_major(
    const std::vector<double>& D_col,   // input: n x k, column-major
    MKL_INT n,
    MKL_INT k,
    std::vector<double>& D_row
) {
    if ((MKL_INT)D_col.size() != (size_t)(n * k)) {
        throw std::runtime_error("D size does not match n*k");
    }
    D_row.resize(n * k);    // will store same logical matrix in row-major

    for (MKL_INT i = 0; i < n; ++i) {
        for (MKL_INT j = 0; j < k; ++j) {
            D_row[i * k + j] = D_col[i + j * n];
        }
    }
}



template <typename T>
void generate_random_dense(std::size_t n1, std::size_t n2, std::vector<T>& out) {
    static_assert(std::is_floating_point<T>::value,
                  "T must be float, double, or long double");

    if (n1 != 0 && n2 > static_cast<std::size_t>(-1) / n2) {
        throw std::overflow_error("n1 * n2 is too large");
    }
    const std::size_t size = n1 * n2;
    out.resize(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(1.0));
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = dist(gen);
    }
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

void load_config(const std::string& data_path, MKL_INT& g1_num_nodes, MKL_INT& g2_num_nodes, bool& use_attr, bool& directed) {
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


bool compare_sparse_exact(
    sparse_matrix_t A,
    sparse_matrix_t B,
    double eps = 1e-9)
{
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

// sparse_matrix_t make_mat_from_edgelist_csc(const std::vector<double>& edgelist,
//                                            MKL_INT m,
//                                            MKL_INT n) {
//     if (edgelist.size() % 2 != 0) {
//         throw std::runtime_error("edgelist size must be even");
//     }

//     MKL_INT nnz = static_cast<MKL_INT>(edgelist.size() / 2);

//     std::vector<MKL_INT> rows(nnz), cols(nnz);
//     std::vector<double> vals(nnz, 1.0);

//     for (MKL_INT i = 0; i < nnz; ++i) {
//         rows[i] = static_cast<MKL_INT>(edgelist[2 * i]);
//         cols[i] = static_cast<MKL_INT>(edgelist[2 * i + 1]);

//         if (rows[i] < 0 || rows[i] >= m || cols[i] < 0 || cols[i] >= n) {
//             throw std::runtime_error("edge index out of bounds");
//         }
//     }

//     sparse_matrix_t A_coo = nullptr;
//     auto st = mkl_sparse_d_create_coo(&A_coo,
//                                       SPARSE_INDEX_BASE_ZERO,
//                                       m, n, nnz,
//                                       rows.data(), cols.data(), vals.data());
//     if (st != SPARSE_STATUS_SUCCESS) {
//         throw std::runtime_error("create_coo failed");
//     }

//     sparse_matrix_t A_csc = nullptr;

//     // CSC(A) is equivalent to CSR(A^T)
//     st = mkl_sparse_convert_csr(A_coo,
//                                 SPARSE_OPERATION_TRANSPOSE,
//                                 &A_csc);
//     if (st != SPARSE_STATUS_SUCCESS) {
//         mkl_sparse_destroy(A_coo);
//         throw std::runtime_error("convert to CSC-like CSR(A^T) failed");
//     }

//     mkl_sparse_destroy(A_coo);
//     return A_csc;
// }

struct CSCStorage {
    std::vector<MKL_INT> col_start;
    std::vector<MKL_INT> col_end;
    std::vector<MKL_INT> row_ind;
    std::vector<double>  vals;
};

static inline void check_mkl(sparse_status_t st, const char* msg) {
    if (st != SPARSE_STATUS_SUCCESS) {
        printf("%s\n", st);
        throw std::runtime_error(msg);
    }
}

void make_mat_from_edgelist_csc(const std::vector<double>& edgelist,
                                MKL_INT m,
                                MKL_INT n,
                                sparse_matrix_t& graph_csc,
                                CSCStorage& storage) {
    if (edgelist.size() % 2 != 0) {
        throw std::runtime_error("edgelist size must be even");
    }

    if (graph_csc != nullptr) {
        mkl_sparse_destroy(graph_csc);
        graph_csc = nullptr;
    }

    const MKL_INT nnz = static_cast<MKL_INT>(edgelist.size() / 2);

    // Count entries per column
    std::vector<MKL_INT> col_count(n, 0);
    for (MKL_INT k = 0; k < nnz; ++k) {
        MKL_INT r = static_cast<MKL_INT>(edgelist[2 * k]);
        MKL_INT c = static_cast<MKL_INT>(edgelist[2 * k + 1]);

        if (r < 0 || r >= m || c < 0 || c >= n) {
            throw std::runtime_error("edge index out of bounds");
        }
        col_count[c]++;
    }

    // Persistent CSC storage
    storage.col_start.resize(n);
    storage.col_end.resize(n);
    storage.row_ind.resize(nnz);
    storage.vals.assign(nnz, 1.0);

    // Build CSC column pointers
    MKL_INT pos = 0;
    for (MKL_INT j = 0; j < n; ++j) {
        storage.col_start[j] = pos;
        pos += col_count[j];
        storage.col_end[j] = pos;
    }

    // Fill row indices
    std::vector<MKL_INT> next = storage.col_start;
    for (MKL_INT k = 0; k < nnz; ++k) {
        MKL_INT r = static_cast<MKL_INT>(edgelist[2 * k]);
        MKL_INT c = static_cast<MKL_INT>(edgelist[2 * k + 1]);

        MKL_INT p = next[c]++;
        storage.row_ind[p] = r;
        storage.vals[p]    = 1.0;
    }

    check_mkl(
        mkl_sparse_d_create_csc(&graph_csc,
                                SPARSE_INDEX_BASE_ZERO,
                                m, n,
                                storage.col_start.data(),
                                storage.col_end.data(),
                                storage.row_ind.data(),
                                storage.vals.data()),
        "mkl_sparse_d_create_csc failed"
    );

    // Optional:
    check_mkl(mkl_sparse_optimize(graph_csc), "mkl_sparse_optimize failed");
}

sparse_matrix_t make_mat_from_edgelist(const std::vector<double>& edgelist, int m, int n) {
    MKL_INT nnz  = edgelist.size() / 2;
	std::vector<MKL_INT> rows(nnz), cols(nnz);
	std::vector<double> vals(nnz);
    printf("csr nnz %d\n", nnz);
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
    mkl_sparse_optimize(A_csr);
    mkl_sparse_destroy(A);

    return A_csr;
}

struct SparseCSRCSCStorage {
    // CSR storage
    std::vector<MKL_INT> csr_row_start;
    std::vector<MKL_INT> csr_row_end;
    std::vector<MKL_INT> csr_col_ind;
    std::vector<double>  csr_vals;

    // CSC storage
    std::vector<MKL_INT> csc_col_start;
    std::vector<MKL_INT> csc_col_end;
    std::vector<MKL_INT> csc_row_ind;
    std::vector<double>  csc_vals;

    ~SparseCSRCSCStorage() = default;
};

void make_csr_csc_edgelist(const std::vector<double>& edgelist,
                           MKL_INT m,
                           MKL_INT n,
                           sparse_matrix_t& graph_csr,
                           sparse_matrix_t& graph_csc,
                           SparseCSRCSCStorage& storage) {
    if (edgelist.size() % 2 != 0) {
        throw std::runtime_error("edgelist size must be even");
    }

    if (graph_csr != nullptr) {
        mkl_sparse_destroy(graph_csr);
        graph_csr = nullptr;
    }
    if (graph_csc != nullptr) {
        mkl_sparse_destroy(graph_csc);
        graph_csc = nullptr;
    }

    const MKL_INT nnz = static_cast<MKL_INT>(edgelist.size() / 2);

    std::vector<MKL_INT> row_count(m, 0);
    std::vector<MKL_INT> col_count(n, 0);

    for (MKL_INT k = 0; k < nnz; ++k) {
        MKL_INT r = static_cast<MKL_INT>(edgelist[2 * k]);
        MKL_INT c = static_cast<MKL_INT>(edgelist[2 * k + 1]);

        if (r < 0 || r >= m || c < 0 || c >= n) {
            throw std::runtime_error("edge index out of bounds");
        }

        row_count[r]++;
        col_count[c]++;
    }

    // Allocate persistent CSR storage
    storage.csr_row_start.resize(m);
    storage.csr_row_end.resize(m);
    storage.csr_col_ind.resize(nnz);
    storage.csr_vals.assign(nnz, 1.0);

    // Allocate persistent CSC storage
    storage.csc_col_start.resize(n);
    storage.csc_col_end.resize(n);
    storage.csc_row_ind.resize(nnz);
    storage.csc_vals.assign(nnz, 1.0);

    // Build CSR row pointers
    {
        MKL_INT pos = 0;
        for (MKL_INT i = 0; i < m; ++i) {
            storage.csr_row_start[i] = pos;
            pos += row_count[i];
            storage.csr_row_end[i] = pos;
        }
    }

    // Build CSC col pointers
    {
        MKL_INT pos = 0;
        for (MKL_INT j = 0; j < n; ++j) {
            storage.csc_col_start[j] = pos;
            pos += col_count[j];
            storage.csc_col_end[j] = pos;
        }
    }

    // Working write pointers
    std::vector<MKL_INT> next_row = storage.csr_row_start;
    std::vector<MKL_INT> next_col = storage.csc_col_start;

    // Fill CSR and CSC
    for (MKL_INT k = 0; k < nnz; ++k) {
        MKL_INT r = static_cast<MKL_INT>(edgelist[2 * k]);
        MKL_INT c = static_cast<MKL_INT>(edgelist[2 * k + 1]);

        // CSR: row r contains col c
        MKL_INT p = next_row[r]++;
        storage.csr_col_ind[p] = c;
        storage.csr_vals[p]    = 1.0;

        // CSC: col c contains row r
        MKL_INT q = next_col[c]++;
        storage.csc_row_ind[q] = r;
        storage.csc_vals[q]    = 1.0;
    }

    for (int t = 0; t < 10; ++t) {
        printf("csr_col_ind[%d] = %lld, csr_vals[%d] = %.1f\n",
            t, (long long)storage.csr_col_ind[t], t, storage.csr_vals[t]);
    }
    printf("sizeof(MKL_INT) = %zu\n", sizeof(MKL_INT));

    check_mkl(
        mkl_sparse_d_create_csr(&graph_csr,
                                SPARSE_INDEX_BASE_ZERO,
                                m, n,
                                storage.csr_row_start.data(),
                                storage.csr_row_end.data(),
                                storage.csr_col_ind.data(),
                                storage.csr_vals.data()),
        "mkl_sparse_d_create_csr failed"
    );

    check_mkl(
        mkl_sparse_d_create_csc(&graph_csc,
                                SPARSE_INDEX_BASE_ZERO,
                                m, n,
                                storage.csc_col_start.data(),
                                storage.csc_col_end.data(),
                                storage.csc_row_ind.data(),
                                storage.csc_vals.data()),
        "mkl_sparse_d_create_csc failed"
    );

    check_mkl(mkl_sparse_order(graph_csr), "mkl_sparse_order(graph_csr) failed");
    // check_mkl(mkl_sparse_order(graph_csc), "mkl_sparse_order(graph_csc) failed");
}

enum class SparseFormat { CSR, CSC };
void print_sparse_matrix_info(sparse_matrix_t A,
                              SparseFormat fmt,
                              const char* name = "matrix") {
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    double* values = nullptr;

    printf("=== %s ===\n", name);

    if (fmt == SparseFormat::CSR) {
        MKL_INT *row_start = nullptr, *row_end = nullptr, *col_indx = nullptr;

        sparse_status_t status = mkl_sparse_d_export_csr(
            A, &indexing, &rows, &cols,
            &row_start, &row_end, &col_indx, &values
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            printf("Error: cannot export %s as CSR (status=%d)\n", name, status);
            return;
        }

        MKL_INT base = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;

        printf("Format: CSR\n");
        printf("Dimensions: %lld x %lld\n", (long long)rows, (long long)cols);
        printf("Index base: %s\n", (base == 0) ? "0-based" : "1-based");

        MKL_INT nnz = row_end[rows - 1] - row_start[0];
        printf("Nonzeros: %lld\n", (long long)nnz);

        printf("Sample entries (row, col, val):\n");
        MKL_INT printed = 0;
        for (MKL_INT i = 0; i < rows && printed < 10; ++i) {
            for (MKL_INT k = row_start[i]; k < row_end[i] && printed < 10; ++k) {
                MKL_INT kk = k - base;   // convert to C-array offset
                printf("(%lld, %lld, %.3f)\n",
                       (long long)(i + base),
                       (long long)col_indx[kk],
                       values[kk]);
                printed++;
            }
        }
    } else {
        MKL_INT *col_start = nullptr, *col_end = nullptr, *row_indx = nullptr;
        double *values = nullptr;

        sparse_status_t status = mkl_sparse_d_export_csc( A, &indexing, &rows, &cols, &col_start, &col_end, &row_indx, &values);

        if (status != SPARSE_STATUS_SUCCESS) {
            printf("Error: cannot export %s as CSC (status=%d)\n", name, status);
            return;
        }

        printf("Format: CSC\n");
        printf("Dimensions: %lld x %lld\n", (long long)rows, (long long)cols);
        printf("Index base: %s\n", (indexing == SPARSE_INDEX_BASE_ZERO) ? "0-based" : "1-based");

        MKL_INT nnz = 0;
        if (cols > 0) {
            if (indexing == SPARSE_INDEX_BASE_ZERO)
                nnz = col_end[cols - 1];
            else
                nnz = col_end[cols - 1] - 1;
        }
        printf("Nonzeros: %lld\n", (long long)nnz);

        printf("Sample entries (col, row, val):\n");
        MKL_INT printed = 0;

        for (MKL_INT j = 0; j < cols && printed < 10; ++j) {
            MKL_INT start = col_start[j];
            MKL_INT end   = col_end[j];

            if (indexing == SPARSE_INDEX_BASE_ONE) {
                start--;
                end--;
            }

            for (MKL_INT k = start; k < end && printed < 10; ++k) {
                MKL_INT r = row_indx[k];
                if (indexing == SPARSE_INDEX_BASE_ONE) r--;

                printf("(%lld, %lld, %.3f)\n",
                    (long long)j,
                    (long long)r,
                    values[k]);
                printed++;
            }
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

void compare_dense( const std::vector<double>& vec1, const std::vector<double>& vec2, const std::string& label) {

    double max_abs = 0.0;
    double max_rel = 0.0;
    double a_err = 0.0, b_err = 0.0;
    int counter = 0;
    int counter_1 = 0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        double a = vec1[i];
        double b = vec2[i];
        // printf("a %e b %e \n", a, b);
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
    std::cout << label << " out of " << vec1.size() << " elements " << counter << " elements has more than 10 percent error" << std::endl;
    std::cout << label << " out of " << vec1.size() << " elements " << counter_1 << " elements has more than 1 percent error" << std::endl;
    std::cout << label << " max_abs = " << max_abs << "  max_rel = " << max_rel
              << "  vec1: " << b_err << "  vec2: " << a_err << "\n\n";
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


namespace {

inline void check_sparse(sparse_status_t st, const char* msg) {
    if (st != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

// Deep-copy an MKL sparse matrix handle.
sparse_matrix_t deep_copy_sparse(sparse_matrix_t A) {
    sparse_matrix_t B = nullptr;
    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    check_sparse(mkl_sparse_copy(A, descr, &B), "mkl_sparse_copy failed");
    return B;
}

// Build a unit-valued CSR matrix from an edge list and return an independent MKL sparse handle.
sparse_matrix_t build_unit_csr_from_edges(
    const std::vector<std::pair<MKL_INT, MKL_INT>>& edges_in,
    MKL_INT nrows,
    MKL_INT ncols)
{
    if (edges_in.empty()) {
        return nullptr;
    }

    std::vector<std::pair<MKL_INT, MKL_INT>> edges = edges_in;
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector<MKL_INT> row_counts(nrows, 0);
    for (const auto& e : edges) {
        if (e.first < 0 || e.first >= nrows || e.second < 0 || e.second >= ncols) {
            throw std::runtime_error("edge index out of bounds in build_unit_csr_from_edges");
        }
        row_counts[e.first]++;
    }

    std::vector<MKL_INT> row_start(nrows), row_end(nrows);
    std::vector<MKL_INT> col_indx(edges.size());
    std::vector<double> values(edges.size(), 1.0);

    MKL_INT nnz_ptr = 0;
    for (MKL_INT i = 0; i < nrows; ++i) {
        row_start[i] = nnz_ptr;
        nnz_ptr += row_counts[i];
        row_end[i] = nnz_ptr;
    }

    std::vector<MKL_INT> next(row_start.begin(), row_start.end());
    for (const auto& e : edges) {
        col_indx[next[e.first]++] = e.second;
    }

    sparse_matrix_t tmp = nullptr;
    check_sparse(
        mkl_sparse_d_create_csr(&tmp,
                                SPARSE_INDEX_BASE_ZERO,
                                nrows,
                                ncols,
                                row_start.data(),
                                row_end.data(),
                                col_indx.data(),
                                values.data()),
        "mkl_sparse_d_create_csr failed in build_unit_csr_from_edges");

    sparse_matrix_t out = deep_copy_sparse(tmp);
    mkl_sparse_destroy(tmp);
    return out;
}

} // namespace


// Assumes graph is an adjacency matrix.
// Row-normalization here is degree-based for an unweighted graph:
// every nonzero in row i becomes 1.0 / nnz(row i).
sparse_matrix_t graph_row_normalize_connect_isolated(sparse_matrix_t graph, unsigned int seed = 42) {
    if (graph == nullptr) {
        throw std::runtime_error("graph is null");
    }

    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT* row_start = nullptr;
    MKL_INT* row_end   = nullptr;
    MKL_INT* col_indx  = nullptr;
    double*  values    = nullptr;

    check_sparse(
        mkl_sparse_d_export_csr(graph, &indexing, &rows, &cols,
                                &row_start, &row_end, &col_indx, &values),
        "mkl_sparse_d_export_csr failed");

    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("graph has invalid dimensions");
    }
    if (rows != cols) {
        throw std::runtime_error("graph_row_normalize_connect_isolated expects a square adjacency matrix");
    }

    const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    const MKL_INT total_nnz = row_end[rows - 1] - base;
    const MKL_INT avg_nnz_per_row = (rows > 0) ? (total_nnz / rows) : 0;

    // Compute row nnz and isolated flags in parallel.
    std::vector<MKL_INT> row_nnz(rows, 0);
    std::vector<char> is_isolated(rows, 0);

    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows; ++i) {
        MKL_INT deg = row_end[i] - row_start[i];
        row_nnz[i] = deg;
        is_isolated[i] = (deg == 0) ? 1 : 0;
    }

    std::vector<MKL_INT> isolated;
    isolated.reserve(rows);
    for (MKL_INT i = 0; i < rows; ++i) {
        if (is_isolated[i]) isolated.push_back(i);
    }

    sparse_matrix_t connected_graph = nullptr;

    if (isolated.empty()) {
        connected_graph = deep_copy_sparse(graph);
    } else {
        // Same spirit as your earlier code:
        // each isolated node gets connected to about avg_nnz_per_row random distinct nodes.
        MKL_INT target_degree = std::max<MKL_INT>(1, avg_nnz_per_row);

        std::vector<std::pair<MKL_INT, MKL_INT>> new_edges;
        if (rows == 1) {
            // Only possible connection is a self-loop.
            new_edges.emplace_back(0, 0);
        } else {
            target_degree = std::min<MKL_INT>(target_degree, rows - 1);
            new_edges.reserve(static_cast<size_t>(isolated.size()) * static_cast<size_t>(target_degree));

            std::mt19937 rng(seed);

            for (MKL_INT i : isolated) {
                std::unordered_set<MKL_INT> nbrs;
                nbrs.reserve(static_cast<size_t>(target_degree) * 2);

                // Sample distinct random neighbors, excluding self.
                std::uniform_int_distribution<MKL_INT> dist(0, rows - 2);

                while (static_cast<MKL_INT>(nbrs.size()) < target_degree) {
                    MKL_INT j = dist(rng);
                    if (j >= i) ++j;   // maps [0, rows-2] -> [0, rows-1] \ {i}
                    nbrs.insert(j);
                }

                for (MKL_INT j : nbrs) {
                    new_edges.emplace_back(i, j);
                }
            }
        }

        sparse_matrix_t add_mat = build_unit_csr_from_edges(new_edges, rows, cols);
        check_sparse(
            mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, graph, 1.0, add_mat, &connected_graph),
            "mkl_sparse_d_add failed");
        if (add_mat) mkl_sparse_destroy(add_mat);
    }

    check_sparse(mkl_sparse_order(connected_graph), "mkl_sparse_order failed on connected_graph");

    // Export connected graph and build row-normalized CSR.
    sparse_index_base_t indexing2;
    MKL_INT rows2, cols2;
    MKL_INT* row_start2 = nullptr;
    MKL_INT* row_end2   = nullptr;
    MKL_INT* col_indx2  = nullptr;
    double*  values2    = nullptr;

    check_sparse(
        mkl_sparse_d_export_csr(connected_graph, &indexing2, &rows2, &cols2,
                                &row_start2, &row_end2, &col_indx2, &values2),
        "mkl_sparse_d_export_csr failed on connected_graph");

    const MKL_INT base2 = (indexing2 == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    const MKL_INT connected_nnz = row_end2[rows2 - 1] - base2;

    std::vector<MKL_INT> out_row_start(rows2);
    std::vector<MKL_INT> out_row_end(rows2);
    std::vector<MKL_INT> out_col_indx(connected_nnz);
    std::vector<double>  out_values(connected_nnz);

    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < rows2; ++i) {
        MKL_INT begin = row_start2[i] - base2;
        MKL_INT end   = row_end2[i]   - base2;
        MKL_INT deg   = end - begin;

        out_row_start[i] = begin;
        out_row_end[i]   = end;

        double inv_deg = (deg > 0) ? (1.0 / static_cast<double>(deg)) : 0.0;

        for (MKL_INT p = begin; p < end; ++p) {
            out_col_indx[p] = col_indx2[p] - base2;
            out_values[p]   = inv_deg;
        }
    }

    sparse_matrix_t tmp_norm = nullptr;
    check_sparse(
        mkl_sparse_d_create_csr(&tmp_norm,
                                SPARSE_INDEX_BASE_ZERO,
                                rows2,
                                cols2,
                                out_row_start.data(),
                                out_row_end.data(),
                                out_col_indx.data(),
                                out_values.data()),
        "mkl_sparse_d_create_csr failed for normalized graph");

    sparse_matrix_t graph_row_normalized = deep_copy_sparse(tmp_norm);

    mkl_sparse_destroy(tmp_norm);
    mkl_sparse_destroy(connected_graph);

    check_sparse(mkl_sparse_order(graph_row_normalized), "mkl_sparse_order failed on graph_row_normalized");
    return graph_row_normalized;
}

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
    MKL_INT columns, MKL_INT ldb, double beta, double* C, MKL_INT ldc, MKL_INT block_size = 128 ) {
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


// struct PermutedMKLCSR {
//     sparse_matrix_t A = nullptr;

//     // Keep backing arrays alive for the lifetime of the handle.
//     // This is the safest choice.
//     std::vector<MKL_INT> rows_start;
//     std::vector<MKL_INT> rows_end;
//     std::vector<MKL_INT> col_indx;
//     std::vector<double> values;

//     // Useful metadata
//     std::vector<idx_t> part;          // METIS part for each old vertex
//     std::vector<MKL_INT> perm;        // new_index -> old_index
//     std::vector<MKL_INT> iperm;       // old_index -> new_index
//     std::vector<MKL_INT> block_ptr;   // block boundaries in permuted ordering

//     MKL_INT nrows = 0;
//     MKL_INT ncols = 0;

//     ~PermutedMKLCSR() {
//         if (A) {
//             mkl_sparse_destroy(A);
//             A = nullptr;
//         }
//     }

//     PermutedMKLCSR() = default;
//     PermutedMKLCSR(const PermutedMKLCSR&) = delete;
//     PermutedMKLCSR& operator=(const PermutedMKLCSR&) = delete;

//     PermutedMKLCSR(PermutedMKLCSR&& other) noexcept {
//         *this = std::move(other);
//     }

//     PermutedMKLCSR& operator=(PermutedMKLCSR&& other) noexcept {
//         if (this != &other) {
//             if (A) mkl_sparse_destroy(A);

//             A = other.A;
//             other.A = nullptr;

//             rows_start = std::move(other.rows_start);
//             rows_end   = std::move(other.rows_end);
//             col_indx   = std::move(other.col_indx);
//             values     = std::move(other.values);

//             part       = std::move(other.part);
//             perm       = std::move(other.perm);
//             iperm      = std::move(other.iperm);
//             block_ptr  = std::move(other.block_ptr);

//             nrows = other.nrows;
//             ncols = other.ncols;
//         }
//         return *this;
//     }
// };

// static inline void check_sparse_status(sparse_status_t st, const char* msg) {
//     if (st != SPARSE_STATUS_SUCCESS) {
//         throw std::runtime_error(msg);
//     }
// }

// static inline void check_metis_ok(int status, const char* msg) {
//     if (status != METIS_OK) {
//         throw std::runtime_error(msg);
//     }
// }

// static inline idx_t safe_to_idx_t(MKL_INT x, const char* msg) {
//     if (x < 0) throw std::runtime_error(msg);
//     if (static_cast<long long>(x) > static_cast<long long>(std::numeric_limits<idx_t>::max())) {
//         throw std::runtime_error(msg);
//     }
//     return static_cast<idx_t>(x);
// }

// static inline MKL_INT safe_to_mkl_int(idx_t x, const char* msg) {
//     if (x < 0) throw std::runtime_error(msg);
//     if (static_cast<long long>(x) > static_cast<long long>(std::numeric_limits<MKL_INT>::max())) {
//         throw std::runtime_error(msg);
//     }
//     return static_cast<MKL_INT>(x);
// }

// /*
// Assumptions:
// - input A_in is CSR and square
// - symmetric permutation A' = P A P^T is desired
// - METIS partitioning is based on the STRUCTURAL graph of A:
//     * diagonal entries are ignored
//     * graph is symmetrized (undirected)
// - original numerical values of A are preserved in the final reordered matrix
// */
// PermutedMKLCSR reorder_with_metis(const sparse_matrix_t A_in, idx_t nparts) {
//     if (!A_in) {
//         throw std::runtime_error("A_in is null");
//     }
//     if (nparts <= 0) {
//         throw std::runtime_error("nparts must be > 0");
//     }

//     // -------------------------------------------------------------------------
//     // 1) Export input MKL CSR
//     // -------------------------------------------------------------------------
//     sparse_index_base_t indexing;
//     MKL_INT rows, cols;
//     MKL_INT* in_rows_start = nullptr;
//     MKL_INT* in_rows_end   = nullptr;
//     MKL_INT* in_col_indx   = nullptr;
//     double*  in_values     = nullptr;

//     check_sparse_status(
//         mkl_sparse_d_export_csr(
//             A_in, &indexing, &rows, &cols,
//             &in_rows_start, &in_rows_end,
//             &in_col_indx, &in_values),
//         "mkl_sparse_d_export_csr failed"
//     );

//     if (rows != cols) {
//         throw std::runtime_error("reorder_with_metis currently assumes a square matrix");
//     }

//     const MKL_INT n = rows;
//     const MKL_INT base = (indexing == SPARSE_INDEX_BASE_ONE) ? 1 : 0;
//     const MKL_INT nnz  = in_rows_end[n - 1] - base;

//     if (nnz < 0) {
//         throw std::runtime_error("invalid CSR: negative nnz");
//     }

//     // -------------------------------------------------------------------------
//     // 2) Build undirected structural graph for METIS from A's sparsity pattern
//     //    Ignore diagonal; symmetrize pattern
//     // -------------------------------------------------------------------------
//     std::vector<std::unordered_set<MKL_INT>> nbrs(static_cast<size_t>(n));

//     for (MKL_INT i = 0; i < n; ++i) {
//         MKL_INT row_begin = in_rows_start[i] - base;
//         MKL_INT row_end   = in_rows_end[i]   - base;

//         for (MKL_INT p = row_begin; p < row_end; ++p) {
//             MKL_INT j = in_col_indx[p] - base;

//             if (j < 0 || j >= n) {
//                 throw std::runtime_error("column index out of range");
//             }
//             if (j == i) continue; // drop diagonal for graph partitioning

//             nbrs[static_cast<size_t>(i)].insert(j);
//             nbrs[static_cast<size_t>(j)].insert(i); // symmetrize
//         }
//     }

//     // Convert to METIS CSR
//     std::vector<idx_t> xadj(static_cast<size_t>(n) + 1, 0);
//     size_t total_adj = 0;
//     for (MKL_INT i = 0; i < n; ++i) {
//         xadj[static_cast<size_t>(i)] = static_cast<idx_t>(total_adj);
//         total_adj += nbrs[static_cast<size_t>(i)].size();
//     }
//     xadj[static_cast<size_t>(n)] = static_cast<idx_t>(total_adj);

//     std::vector<idx_t> adjncy;
//     adjncy.reserve(total_adj);

//     for (MKL_INT i = 0; i < n; ++i) {
//         // sorting neighbors is not required, but nice for determinism
//         std::vector<MKL_INT> row_nbrs(
//             nbrs[static_cast<size_t>(i)].begin(),
//             nbrs[static_cast<size_t>(i)].end()
//         );
//         std::sort(row_nbrs.begin(), row_nbrs.end());
//         for (MKL_INT j : row_nbrs) {
//             adjncy.push_back(safe_to_idx_t(j, "MKL_INT -> idx_t overflow in adjncy"));
//         }
//     }

//     // -------------------------------------------------------------------------
//     // 3) METIS partition
//     // -------------------------------------------------------------------------
//     idx_t nvtxs = safe_to_idx_t(n, "MKL_INT -> idx_t overflow in nvtxs");
//     idx_t ncon  = 1;
//     idx_t objval = 0;

//     if (nparts > nvtxs) {
//         nparts = nvtxs;
//     }

//     std::vector<idx_t> part(static_cast<size_t>(nvtxs), 0);

//     int metis_status = METIS_PartGraphKway(
//         &nvtxs,
//         &ncon,
//         xadj.data(),
//         adjncy.data(),
//         nullptr,   // vwgt
//         nullptr,   // vsize
//         nullptr,   // adjwgt
//         &nparts,
//         nullptr,   // tpwgts
//         nullptr,   // ubvec
//         nullptr,   // options
//         &objval,
//         part.data()
//     );
//     check_metis_ok(metis_status, "METIS_PartGraphKway failed");

//     // -------------------------------------------------------------------------
//     // 4) Build perm / iperm / block_ptr
//     //    perm[new] = old
//     //    iperm[old] = new
//     // -------------------------------------------------------------------------
//     std::vector<MKL_INT> counts(static_cast<size_t>(nparts), 0);
//     for (idx_t p : part) {
//         if (p < 0 || p >= nparts) {
//             throw std::runtime_error("invalid METIS partition id");
//         }
//         counts[static_cast<size_t>(p)]++;
//     }

//     std::vector<MKL_INT> block_ptr(static_cast<size_t>(nparts) + 1, 0);
//     for (idx_t p = 0; p < nparts; ++p) {
//         block_ptr[static_cast<size_t>(p + 1)] =
//             block_ptr[static_cast<size_t>(p)] + counts[static_cast<size_t>(p)];
//     }

//     std::vector<MKL_INT> perm(static_cast<size_t>(n), 0);
//     std::vector<MKL_INT> next = block_ptr;

//     for (MKL_INT old_i = 0; old_i < n; ++old_i) {
//         idx_t p = part[static_cast<size_t>(old_i)];
//         perm[static_cast<size_t>(next[static_cast<size_t>(p)]++)] = old_i;
//     }

//     std::vector<MKL_INT> iperm(static_cast<size_t>(n), 0);
//     for (MKL_INT new_i = 0; new_i < n; ++new_i) {
//         iperm[static_cast<size_t>(perm[static_cast<size_t>(new_i)])] = new_i;
//     }

//     // -------------------------------------------------------------------------
//     // 5) Build permuted CSR for A' = P A P^T
//     //
//     // For each new row new_i:
//     //   old_i = perm[new_i]
//     //   each old entry (old_i, old_j) goes to (new_i, iperm[old_j])
//     // -------------------------------------------------------------------------
//     std::vector<MKL_INT> out_rows_start(static_cast<size_t>(n), 0);
//     std::vector<MKL_INT> out_rows_end(static_cast<size_t>(n), 0);

//     // First pass: row lengths
//     for (MKL_INT new_i = 0; new_i < n; ++new_i) {
//         MKL_INT old_i = perm[static_cast<size_t>(new_i)];
//         MKL_INT row_begin = in_rows_start[old_i] - base;
//         MKL_INT row_end   = in_rows_end[old_i]   - base;

//         out_rows_start[static_cast<size_t>(new_i)] = row_begin; // temporary
//         out_rows_end[static_cast<size_t>(new_i)]   = row_end;   // temporary
//     }

//     // Because symmetric permutation preserves nnz count row-by-row under row reorder,
//     // row length of new row == row length of corresponding old row.
//     std::vector<MKL_INT> row_ptr(static_cast<size_t>(n) + 1, 0);
//     for (MKL_INT new_i = 0; new_i < n; ++new_i) {
//         MKL_INT len = out_rows_end[static_cast<size_t>(new_i)] -
//                       out_rows_start[static_cast<size_t>(new_i)];
//         row_ptr[static_cast<size_t>(new_i + 1)] =
//             row_ptr[static_cast<size_t>(new_i)] + len;
//     }

//     const MKL_INT out_nnz = row_ptr[static_cast<size_t>(n)];
//     std::vector<MKL_INT> out_col_indx(static_cast<size_t>(out_nnz), 0);
//     std::vector<double>  out_values(static_cast<size_t>(out_nnz), 0.0);

//     for (MKL_INT new_i = 0; new_i < n; ++new_i) {
//         MKL_INT old_i = perm[static_cast<size_t>(new_i)];
//         MKL_INT old_begin = in_rows_start[old_i] - base;
//         MKL_INT old_end   = in_rows_end[old_i]   - base;

//         MKL_INT out_begin = row_ptr[static_cast<size_t>(new_i)];
//         MKL_INT len = old_end - old_begin;

//         for (MKL_INT k = 0; k < len; ++k) {
//             MKL_INT old_j = in_col_indx[old_begin + k] - base;
//             double  val   = in_values[old_begin + k];

//             MKL_INT new_j = iperm[static_cast<size_t>(old_j)];

//             out_col_indx[static_cast<size_t>(out_begin + k)] = new_j;
//             out_values[static_cast<size_t>(out_begin + k)]   = val;
//         }

//         // Sort columns within this row
//         auto first_col = out_col_indx.begin() + out_begin;
//         auto last_col  = first_col + len;
//         auto first_val = out_values.begin() + out_begin;

//         std::vector<std::pair<MKL_INT,double>> tmp;
//         tmp.reserve(static_cast<size_t>(len));
//         for (MKL_INT k = 0; k < len; ++k) {
//             tmp.emplace_back(*(first_col + k), *(first_val + k));
//         }
//         std::sort(tmp.begin(), tmp.end(),
//                   [](const auto& a, const auto& b) { return a.first < b.first; });

//         for (MKL_INT k = 0; k < len; ++k) {
//             *(first_col + k) = tmp[static_cast<size_t>(k)].first;
//             *(first_val + k) = tmp[static_cast<size_t>(k)].second;
//         }
//     }

//     // Convert row_ptr into MKL rows_start / rows_end
//     for (MKL_INT i = 0; i < n; ++i) {
//         out_rows_start[static_cast<size_t>(i)] = row_ptr[static_cast<size_t>(i)];
//         out_rows_end[static_cast<size_t>(i)]   = row_ptr[static_cast<size_t>(i + 1)];
//     }

//     // Convert to one-based if input used one-based
//     if (indexing == SPARSE_INDEX_BASE_ONE) {
//         for (MKL_INT i = 0; i < n; ++i) {
//             out_rows_start[static_cast<size_t>(i)] += 1;
//             out_rows_end[static_cast<size_t>(i)]   += 1;
//         }
//         for (MKL_INT& c : out_col_indx) c += 1;
//     }

//     // -------------------------------------------------------------------------
//     // 6) Create MKL CSR handle
//     // -------------------------------------------------------------------------
//     PermutedMKLCSR result;
//     result.nrows = n;
//     result.ncols = n;
//     result.part = std::move(part);
//     result.perm = std::move(perm);
//     result.iperm = std::move(iperm);
//     result.block_ptr = std::move(block_ptr);
//     result.rows_start = std::move(out_rows_start);
//     result.rows_end   = std::move(out_rows_end);
//     result.col_indx   = std::move(out_col_indx);
//     result.values     = std::move(out_values);

//     check_sparse_status(
//         mkl_sparse_d_create_csr(
//             &result.A,
//             indexing,
//             n,
//             n,
//             result.rows_start.data(),
//             result.rows_end.data(),
//             result.col_indx.data(),
//             result.values.data()
//         ),
//         "mkl_sparse_d_create_csr failed for reordered matrix"
//     );

//     // Optional but good practice
//     check_sparse_status(
//         mkl_sparse_order(result.A),
//         "mkl_sparse_order failed on reordered matrix"
//     );

//     return result;
// }

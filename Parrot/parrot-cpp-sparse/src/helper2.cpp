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

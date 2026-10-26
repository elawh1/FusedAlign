#pragma once

#include <mkl.h>
#include <cmath>
#include <cinttypes>   
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <omp.h>
#include <sstream>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <type_traits>
#include "debug_log.h"

bool has_nan(const std::vector<double>& v);

bool has_nan_or_inf(const std::vector<double>& v) ;

double getMax(const std::vector<double>& v) ;

double getMin(const std::vector<double>& v) ;

void load_config(const std::string& data_path, MKL_INT& g1_num_nodes, MKL_INT& g2_num_nodes, bool& use_attr, bool& directed, int& inner_iters, int& outer_iters) ;

template <typename Real = double>
std::vector<Real> readFileToVector(const std::string &filename, bool directed) ;

void print_num_zeros(std::vector<double>  M, const char* name ) ;

void compare( const std::vector<double>& cpp_vec, const std::string& filename, const std::string& label) ;

void print_Cost_vals (const std::vector<double>& Cost) ;

double percentile(std::vector<double> S, double p) ;

void print_stats(const std::vector<double>& S) ;

template <typename Real = double>
void daxpy_chunked(size_t N, Real alpha, const Real* x, Real* y) ;

template <typename Real = double>
void dscal_chunked(size_t N, Real alpha, Real* x) ;

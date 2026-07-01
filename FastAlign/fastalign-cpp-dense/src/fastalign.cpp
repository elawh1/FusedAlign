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
#include <utility>
#include <type_traits>
#include "accuracy.h"
#include "helper.h"
#include "helper2.h"

constexpr bool kUseFloat = true;
using Real = std::conditional_t<kUseFloat, float, double>;

int main(int argc, char* argv[]){ 

    std::string data_path;
    MKL_INT g1_num_nodes, g2_num_nodes;
    bool use_attr = false, directed = false;
    int inner_iters = 5;
    int outer_iters = 10;
    bool sparsify = false;
    using milliseconds = std::chrono::milliseconds;
    std::vector<std::pair<std::string, long long>> phase_timings;
    
    if (argc < 3) {
        printf("provide input directory and sparsifying option \n");
        return 0;
    }
    else {
        data_path = argv[1];
        sparsify = std::stoi(argv[2]);
    }
    auto setup_start = std::chrono::high_resolution_clock::now();
    load_config(data_path, g1_num_nodes, g2_num_nodes, use_attr, directed, inner_iters, outer_iters);
    std::cout << "----- Start -----" << std::endl;
    printf("MKL threads: %d\n", mkl_get_max_threads());
    printf("MKL dynamic: %d\n", mkl_get_dynamic());
    printf("OMP threads: %d\n", omp_get_max_threads());
    DEBUG_ONLY(printf("OMP in parallel region? %d\n", omp_in_parallel()));
    DEBUG_ONLY(printf("graph1 #nodes %d -- graph2 #nodes %d -- directed %d -- use node attributes %d\n\n", g1_num_nodes, g2_num_nodes, directed, use_attr));

    
    sparse_matrix_t graph1_py, graph2_py;
    MKL_INT avg_nnz1_per_n1, avg_nnz2_per_n2;
    {
        std::vector<Real> g1_edgelist = readFileToVector<Real>(data_path + "/graph1.txt", directed);
        graph1_py = make_mat_from_edgelist<Real>(g1_edgelist, g1_num_nodes, g1_num_nodes);        //row major csr
        avg_nnz1_per_n1 = (int) g1_edgelist.size()/g1_num_nodes;
    }
    {
        std::vector<Real> g2_edgelist = readFileToVector<Real>(data_path + "/graph2.txt", directed);
        graph2_py = make_mat_from_edgelist<Real>(g2_edgelist, g2_num_nodes, g2_num_nodes);        //row major csr
        avg_nnz2_per_n2 = (int) g2_edgelist.size()/g2_num_nodes;
    }
    std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchor_pairs, anchor_pairs_test;
    {
        std::vector<Real> anchors = readFileToVector<Real>(data_path + "/anchor_links.txt", directed);
        anchor_pairs = get_anchor_pairs<Real>(anchors);
    } 
    {
        std::vector<Real> anchors_test = readFileToVector<Real>(data_path + "/test_pairs.txt", directed);
        anchor_pairs_test = get_anchor_pairs<Real>(anchors_test);
    }
    std::vector<Real> g1_feat, g2_feat;
    if(use_attr) {
        g1_feat = readFileToVector<Real>(data_path + "/graph1_feat.txt", directed);
        g2_feat = readFileToVector<Real>(data_path + "/graph2_feat.txt", directed);
        DEBUG_ONLY(printf("g1_feat size %ld  g2_feat size %ld\n", g1_feat.size(), g2_feat.size()));
    }
    auto setup_end = std::chrono::high_resolution_clock::now();
    // phase_timings.push_back({"input loading", std::chrono::duration_cast<milliseconds>(setup_end - setup_start).count()});
    int l = anchor_pairs.first.size();
    std::cout << "Dataset path " << data_path << std::endl;
    printf("graph1 #nodes %d -- graph2 #nodes %d -- directed %d -- avg nnz1/n1 %d -- avg nnz2/n2  %d -- use node attributes %d --- # of anchor links(l) %ld\n\n",
        g1_num_nodes, g2_num_nodes, directed, avg_nnz1_per_n1, avg_nnz2_per_n2, use_attr, anchor_pairs.first.size());

    //normalize graph adjacencies
    auto time_0 = std::chrono::high_resolution_clock::now();
    sparse_matrix_t graph1_row_normalized = row_normalize_graph<Real>(graph1_py, g1_num_nodes, avg_nnz1_per_n1);
    sparse_matrix_t graph2_row_normalized = row_normalize_graph<Real>(graph2_py, g2_num_nodes, avg_nnz2_per_n2);
    auto time_1 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"row normalizing graphs", std::chrono::duration_cast<milliseconds>(time_1 - time_0).count()});
    DEBUG_ONLY(std::cout << "row normalizing graphs : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() << " ms" << std::endl);

    std::vector<Real>  g1_rwr ((size_t) g1_num_nodes * (size_t) l);
    auto g1_rwr_start = std::chrono::high_resolution_clock::now();
    get_batch_rwr_scores_rowmjr<Real>(graph1_row_normalized, anchor_pairs.first, g1_num_nodes, avg_nnz1_per_n1, directed, Real(0.15), 100, true, g1_rwr);
    auto time_2 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"graph1 RWR", std::chrono::duration_cast<milliseconds>(time_2 - g1_rwr_start).count()});
    DEBUG_ONLY(std::cout << "graph1 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-g1_rwr_start).count() << " ms" << std::endl);
    std::vector<Real>  g2_rwr ((size_t) g2_num_nodes * (size_t) l);
    auto g2_rwr_start = std::chrono::high_resolution_clock::now();
    get_batch_rwr_scores_rowmjr<Real>(graph2_row_normalized, anchor_pairs.second, g2_num_nodes, avg_nnz2_per_n2, directed, Real(0.15), 100, true, g2_rwr);
    auto time_3 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"graph2 RWR", std::chrono::duration_cast<milliseconds>(time_3 - g2_rwr_start).count()});
    DEBUG_ONLY(std::cout << "graph2 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-g2_rwr_start).count() << " ms" << std::endl);

    //row normalize rwr for both graphs
    l2_norm_inplace<Real>(g1_rwr, g1_num_nodes, l);
    l2_norm_inplace<Real>(g2_rwr, g2_num_nodes, l);

    std::vector<Real> cross_rwr_dist((size_t) g1_num_nodes * (size_t) g2_num_nodes);
    auto cross_rwr_start = std::chrono::high_resolution_clock::now();
    get_normalized_neg_exp_dist<Real>(g1_rwr, g2_rwr, g1_num_nodes, g2_num_nodes, l, cross_rwr_dist);
    if (use_attr) {
        l2_norm_inplace<Real>(g1_feat, g1_num_nodes,  (size_t) g1_feat.size() / (size_t) g1_num_nodes);
        l2_norm_inplace<Real>(g2_feat, g2_num_nodes,  (size_t) g2_feat.size() / (size_t) g2_num_nodes);
        std::vector<Real> cross_attr_dist((size_t) g1_num_nodes * (size_t) g2_num_nodes);
        get_normalized_neg_exp_dist<Real>(g1_feat, g2_feat,  (size_t) g1_num_nodes,  (size_t) g2_num_nodes,  (size_t) g1_feat.size() / (size_t) g1_num_nodes, cross_attr_dist);
        daxpy_chunked<Real>(cross_attr_dist.size(), Real(0.5), cross_rwr_dist.data(), cross_attr_dist.data());
        cross_rwr_dist.swap(cross_attr_dist);
    } else {
        dscal_chunked<Real>(((size_t) g1_num_nodes * (size_t) g2_num_nodes), Real(1.5), cross_rwr_dist.data());
    }
    //make the anchor links cross rwr dist zero
    anchor_cost_zero<Real>(cross_rwr_dist, g1_num_nodes, g2_num_nodes, anchor_pairs, Real(0));
    auto time_4 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"cross RWR distance", std::chrono::duration_cast<milliseconds>(time_4 - cross_rwr_start).count()});
    DEBUG_ONLY(std::cout << "cross_rwr_dist computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-cross_rwr_start).count() << " ms" << std::endl);

    sparse_matrix_t intra_rwr1_masked_sddmm;
    auto time_7 = std::chrono::high_resolution_clock::now();
    if (use_attr) masked_sddmm_intra_cost<Real>(g1_feat, graph1_row_normalized, g1_num_nodes, (g1_feat.size()) / g1_num_nodes, intra_rwr1_masked_sddmm);
    else masked_sddmm_intra_cost<Real>(g1_rwr, graph1_row_normalized, g1_num_nodes, l, intra_rwr1_masked_sddmm);
    auto time_8 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"graph1 intra distance", std::chrono::duration_cast<milliseconds>(time_8 - time_7).count()});
    DEBUG_ONLY(std::cout << "g1 rwr dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_8-time_7).count() << " ms" << std::endl);
    sparse_matrix_t intra_rwr2_masked_sddmm;
    auto g2_intra_start = std::chrono::high_resolution_clock::now();
    if (use_attr) masked_sddmm_intra_cost<Real>(g2_feat, graph2_row_normalized, g2_num_nodes, (g2_feat.size()) / g2_num_nodes, intra_rwr2_masked_sddmm);
    else masked_sddmm_intra_cost<Real>(g2_rwr, graph2_row_normalized, g2_num_nodes, l, intra_rwr2_masked_sddmm);
    auto time_9 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"graph2 intra distance", std::chrono::duration_cast<milliseconds>(time_9 - g2_intra_start).count()});
    DEBUG_ONLY(std::cout << "g2 rwr dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_9-g2_intra_start).count() << " ms" << std::endl);
    //Free some memory 
    g1_feat.clear(); g1_feat.shrink_to_fit();
    g2_feat.clear(); g2_feat.shrink_to_fit();
    g1_rwr.clear(); g1_rwr.shrink_to_fit();
    g2_rwr.clear(); g2_rwr.shrink_to_fit();
    
    //Compute cross_dist
    std::vector<Real> cross_dist((size_t) g1_num_nodes * (size_t) g2_num_nodes);
    auto time_10 = std::chrono::high_resolution_clock::now();
    get_product_rwr_mat_new<Real>(graph1_row_normalized, graph2_row_normalized, cross_rwr_dist, g1_num_nodes, g2_num_nodes, Real(0.15), Real(0.75), cross_dist); 
    auto time_11 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"cross distance propagation", std::chrono::duration_cast<milliseconds>(time_11 - time_10).count()});
    DEBUG_ONLY(std::cout << "cross_dist computation completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_11-time_10).count() << " ms" << std::endl);
    cross_rwr_dist.clear(); cross_rwr_dist.shrink_to_fit();

    auto dense_conversion_start = std::chrono::high_resolution_clock::now();
    std::vector<Real> graph1_row_normalized_dense = sparse_to_dense_colmajor_typed<Real>(graph1_py, g1_num_nodes, g1_num_nodes);
    std::vector<Real> graph2_row_normalized_dense = sparse_to_dense_colmajor_typed<Real>(graph2_py, g2_num_nodes, g2_num_nodes);
    row_normalize_dense_with_isolated_ones<Real>(graph1_row_normalized_dense, g1_num_nodes);
    row_normalize_dense_with_isolated_ones<Real>(graph2_row_normalized_dense, g2_num_nodes);
    std::vector<Real> intra_rwr1_dense = sparse_to_dense_colmajor_typed<Real>(intra_rwr1_masked_sddmm, g1_num_nodes, g1_num_nodes);
    std::vector<Real> intra_rwr2_dense = sparse_to_dense_colmajor_typed<Real>(intra_rwr2_masked_sddmm, g2_num_nodes, g2_num_nodes);
    mkl_sparse_destroy(graph1_py); graph1_py = nullptr;
    mkl_sparse_destroy(graph2_py); graph2_py = nullptr;
    mkl_sparse_destroy(graph1_row_normalized); graph1_row_normalized = nullptr;
    mkl_sparse_destroy(graph2_row_normalized); graph2_row_normalized = nullptr;
    mkl_sparse_destroy(intra_rwr1_masked_sddmm); intra_rwr1_masked_sddmm = nullptr;
    mkl_sparse_destroy(intra_rwr2_masked_sddmm); intra_rwr2_masked_sddmm = nullptr;
    auto dense_conversion_end = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"dense conversion for optimization", std::chrono::duration_cast<milliseconds>(dense_conversion_end - dense_conversion_start).count()});
    DEBUG_ONLY(std::cout << "dense conversion for optimization completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(dense_conversion_end-dense_conversion_start).count() << " ms" << std::endl);

    auto optimization_start = std::chrono::high_resolution_clock::now();
    std::vector<Real> S = con_prox_pt_opt_dense<Real>(graph1_row_normalized_dense, graph2_row_normalized_dense, g1_num_nodes, g2_num_nodes, cross_dist, intra_rwr1_dense,
        intra_rwr2_dense, inner_iters, outer_iters, Real(5e-3), Real(5e-4), Real(5e-4), Real(1e-5), anchor_pairs);
    auto time_12 = std::chrono::high_resolution_clock::now();
    phase_timings.push_back({"optimization", std::chrono::duration_cast<milliseconds>(time_12 - optimization_start).count()});
    DEBUG_ONLY(std::cout << "con_prox_pt_opt_dense completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_12-optimization_start).count() << " ms" << std::endl);
    long long total_algorithm_ms = 0;
    for (const auto& phase : phase_timings) {
        total_algorithm_ms += phase.second;
    }
    std::cout << "total phases of algorithm completed in : " << total_algorithm_ms  << " ms" << std::endl;
    std::cout << "phase timings " << std::endl;
    for (const auto& phase : phase_timings) {
        std::cout << "  " << phase.first << ": " << phase.second << " ms" << std::endl;
    }

    std::cout << "Total time of algorithm including overheads : " << std::chrono::duration_cast<milliseconds>(time_12 - time_0).count() << " ms" << std::endl << std::endl;

    unsigned int numThreads = mkl_get_max_threads();
    omp_set_num_threads(numThreads);      
    std::unordered_map<int, double>  hits = hits_ks_mean_scores<Real>( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test, std::vector<int>{1, 10, 30, 50});

    for (const auto& pair : hits) {
        std::cout << "hit@ " << pair.first << " : " << pair.second << std::endl;
    }

    double mrr = mrr_mean_score_cpp<Real>( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test);
    std::cout << "mrr : " << mrr << std::endl;

    std::cout << "----- End -----" << std::endl;

    return 0;
}

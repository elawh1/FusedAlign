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
#include "accuracy.h"
#include "helper.h"
#include "helper2.h"

int main(int argc, char* argv[]){ 

    std::string data_path;
    MKL_INT g1_num_nodes, g2_num_nodes;
    bool use_attr = false, directed = false;
    int inner_iters = 5;
    int outer_iters = 10;
    bool sparsify = false;
    
    if (argc < 3) {
        printf("provide input directory and sparsifying option \n");
        return 0;
    }
    else {
        data_path = argv[1];
        sparsify = std::stoi(argv[2]);
    }
    load_config(data_path, g1_num_nodes, g2_num_nodes, use_attr, directed, inner_iters, outer_iters);
    printf("MKL threads: %d\n", mkl_get_max_threads());
    printf("MKL dynamic: %d\n", mkl_get_dynamic());
    // printf("MKL threading layer: %d\n", mkl_get_threading_layer());
    printf("OMP max threads: %d\n", omp_get_max_threads());
    printf("OMP in parallel region? %d\n", omp_in_parallel());
    printf("graph1 #nodes %d -- graph2 #nodes %d -- directed %d -- use node attributes %d\n\n", g1_num_nodes, g2_num_nodes, directed, use_attr);

    
    sparse_matrix_t graph1_py, graph2_py;
    MKL_INT avg_nnz1_per_n1, avg_nnz2_per_n2;
    {
        std::vector<double> g1_edgelist = readFileToVector(data_path + "/graph1.txt", directed);
        graph1_py = make_mat_from_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes);        //row major csr
        avg_nnz1_per_n1 = (int) g1_edgelist.size()/g1_num_nodes;
    }
    {
        std::vector<double> g2_edgelist = readFileToVector(data_path + "/graph2.txt", directed);
        graph2_py = make_mat_from_edgelist(g2_edgelist, g2_num_nodes, g2_num_nodes);        //row major csr
        avg_nnz2_per_n2 = (int) g2_edgelist.size()/g2_num_nodes;
    }
    std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchor_pairs, anchor_pairs_test;
    {
        std::vector<double> anchors = readFileToVector(data_path + "/anchor_links.txt", directed);
        anchor_pairs = get_anchor_pairs(anchors);
    } 
    {
        std::vector<double> anchors_test = readFileToVector(data_path + "/test_pairs.txt", directed);
        anchor_pairs_test = get_anchor_pairs(anchors_test);
    }
    std::vector<double> g1_feat, g2_feat;
    if(use_attr) {
        g1_feat = readFileToVector(data_path + "/graph1_feat.txt", directed);
        g2_feat = readFileToVector(data_path + "/graph2_feat.txt", directed);
        printf("g1_feat size %ld  g2_feat size %ld\n", g1_feat.size(), g2_feat.size());
    }
    int l = anchor_pairs.first.size();
    printf("graph1 #nodes %d -- graph2 #nodes %d -- avg nnz1/n1 %d -- avg nnz2/n2  %d -- use node attributes %d --- # of anchor links(l) %ld\n\n", 
    g1_num_nodes, g2_num_nodes, avg_nnz1_per_n1, avg_nnz2_per_n2, use_attr, anchor_pairs.first.size());

    auto time_0 = std::chrono::high_resolution_clock::now();
    //normalize graph adjacencies
    sparse_matrix_t graph1_row_normalized = row_normalize_graph(graph1_py, g1_num_nodes, avg_nnz1_per_n1);
    sparse_matrix_t graph2_row_normalized = row_normalize_graph(graph2_py, g2_num_nodes, avg_nnz2_per_n2);
    auto time_1 = std::chrono::high_resolution_clock::now();
    std::cout << "row normalizing graphs : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count()   << " ms" << std::endl;

    //Free graphs - we will use normalized graph from now on
    mkl_sparse_destroy(graph1_py);  graph1_py = nullptr;
    mkl_sparse_destroy(graph2_py);  graph2_py = nullptr;

    std::vector<double>  g1_rwr ((size_t) g1_num_nodes * (size_t) l);
    get_batch_rwr_scores(graph1_row_normalized, anchor_pairs.first, g1_num_nodes, avg_nnz1_per_n1, directed, 0.15, 100, true, g1_rwr);
    auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "graph1 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count()   << " ms" << std::endl;
    std::vector<double>  g2_rwr ((size_t) g2_num_nodes * (size_t) l);
    get_batch_rwr_scores(graph2_row_normalized, anchor_pairs.second, g2_num_nodes, avg_nnz2_per_n2, directed, 0.15, 100, true, g2_rwr);
    auto time_3 = std::chrono::high_resolution_clock::now();
    std::cout << "graph2 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-time_2).count()   << " ms" << std::endl;
    
    // std::vector<double> cross_rwr_dist (g1_num_nodes * g2_num_nodes);
    // if (use_attr) {
    //     get_normalized_neg_exp_dist (g1_rwr, g2_rwr, g1_num_nodes, g2_num_nodes, l, cross_rwr_dist);
    //     std::vector<double> cross_attr_dist (g1_num_nodes * g2_num_nodes);
    //     get_normalized_neg_exp_dist (g1_feat, g2_feat, g1_num_nodes, g2_num_nodes, g1_feat.size() / g1_num_nodes, cross_attr_dist);
    //     cblas_daxpy(cross_attr_dist.size(), 0.5, cross_rwr_dist.data(), 1, cross_rwr_dist.data(), 1);
    //     cross_attr_dist.clear(); cross_attr_dist.shrink_to_fit();
    // }
    // else {
    //     get_normalized_neg_exp_dist (g1_rwr, g2_rwr, g1_num_nodes, g2_num_nodes, l, cross_rwr_dist);
    //     cblas_dscal(g1_num_nodes * g2_num_nodes, (double) 1.5, cross_rwr_dist.data(), 1);
    //     // cross_attr_dist = cross_rwr_dist;
    //     // cblas_daxpy(cross_attr_dist.size(), 0.5, cross_rwr_dist.data(), 1, cross_attr_dist.data(), 1);
    // }

    //row normalize rwr for both graphs
    l2_norm_inplace(g1_rwr, g1_num_nodes, l);
    l2_norm_inplace(g2_rwr, g2_num_nodes, l);

    std::vector<double> cross_rwr_dist((size_t) g1_num_nodes * (size_t) g2_num_nodes);
    get_normalized_neg_exp_dist(g1_rwr, g2_rwr, g1_num_nodes, g2_num_nodes, l, cross_rwr_dist);
    if (use_attr) {
        l2_norm_inplace(g1_feat, g1_num_nodes,  (size_t) g1_feat.size() / (size_t) g1_num_nodes);
        l2_norm_inplace(g2_feat, g2_num_nodes,  (size_t) g2_feat.size() / (size_t) g2_num_nodes);
        std::vector<double> cross_attr_dist((size_t) g1_num_nodes * (size_t) g2_num_nodes);
        get_normalized_neg_exp_dist(g1_feat, g2_feat,  (size_t) g1_num_nodes,  (size_t) g2_num_nodes,  (size_t) g1_feat.size() / (size_t) g1_num_nodes, cross_attr_dist);
        cblas_daxpy((size_t) cross_attr_dist.size(), 0.5, cross_rwr_dist.data(), 1, cross_attr_dist.data(), 1);
        cross_rwr_dist.swap(cross_attr_dist);
    } else {
        cblas_dscal(((size_t) g1_num_nodes * (size_t) g2_num_nodes), 1.5, cross_rwr_dist.data(), 1);
    }
    //make the anchor links cross rwr dist zero
    anchor_cost_zero(cross_rwr_dist, g1_num_nodes, g2_num_nodes, anchor_pairs, 0.0);
    auto time_4 = std::chrono::high_resolution_clock::now();
    std::cout << "cross_rwr_dist computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-time_3).count() << " ms" << std::endl;

    auto time_7 = std::chrono::high_resolution_clock::now();
    sparse_matrix_t intra_rwr1_masked_sddmm;
    if (use_attr) masked_sddmm_intra_cost(g1_feat, graph1_row_normalized, g1_num_nodes, (g1_feat.size()) / g1_num_nodes, intra_rwr1_masked_sddmm);
    else masked_sddmm_intra_cost(g1_rwr, graph1_row_normalized, g1_num_nodes, l, intra_rwr1_masked_sddmm);
    auto time_8 = std::chrono::high_resolution_clock::now();
    std::cout << "g1 rwr dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_8-time_7).count() << " ms" << std::endl;

    sparse_matrix_t intra_rwr2_masked_sddmm;
    if (use_attr) masked_sddmm_intra_cost(g2_feat, graph2_row_normalized, g2_num_nodes, (g2_feat.size()) / g2_num_nodes, intra_rwr2_masked_sddmm);
    else masked_sddmm_intra_cost(g2_rwr, graph2_row_normalized, g2_num_nodes, l, intra_rwr2_masked_sddmm);
    auto time_9 = std::chrono::high_resolution_clock::now();
    std::cout << "g2 rwr dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_9-time_8).count()  << " ms" << std::endl;

    //Free some memory 
    g1_feat.clear(); g1_feat.shrink_to_fit();
    g2_feat.clear(); g2_feat.shrink_to_fit();
    g1_rwr.clear(); g1_rwr.shrink_to_fit();
    g2_rwr.clear(); g2_rwr.shrink_to_fit();

    //Compute cross_dist
    auto time_10 = std::chrono::high_resolution_clock::now();
    std::vector<double> cross_dist((size_t) g1_num_nodes * (size_t) g2_num_nodes);
    get_product_rwr_mat_new(graph1_row_normalized, graph2_row_normalized, cross_rwr_dist, g1_num_nodes, g2_num_nodes, 0.15, 0.75, cross_dist); 
    cross_rwr_dist.clear(); cross_rwr_dist.shrink_to_fit();
    auto time_11 = std::chrono::high_resolution_clock::now();
    std::cout << "cross_dist computation completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_11-time_10).count()   << " ms" << std::endl;
    std::vector<double> S = con_prox_pt_opt(graph1_row_normalized, graph2_row_normalized, g1_num_nodes, g2_num_nodes, avg_nnz1_per_n1, avg_nnz2_per_n2, cross_dist, intra_rwr1_masked_sddmm, 
        intra_rwr2_masked_sddmm, inner_iters, outer_iters, 5e-3, 5e-4, 5e-4, 1e-5, anchor_pairs, sparsify);
    auto time_12 = std::chrono::high_resolution_clock::now();
    std::cout << "con_prox_pt_opt completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_12-time_11).count() << " ms" << std::endl;
    std::cout << "algorithm completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_12-time_0).count()  << " ms" << std::endl;

    std::unordered_map<int, double>  hits = hits_ks_mean_scores( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test, std::vector<int>{1, 10, 30, 50});

    for (const auto& pair : hits) {
        std::cout << "hit@ " << pair.first << " : " << pair.second << std::endl;
    }

    double mrr = mrr_mean_score_cpp( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test);
    std::cout << "mrr : " << mrr << std::endl;

    return 0;
}


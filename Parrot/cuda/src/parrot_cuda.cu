#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <ctime> 
#include <chrono>
#include <utility>
#include "accuracy.h"
#include "cuda_helper.cuh"

int main(int argc, char* argv[]){ 

    std::string data_path;
    MKL_INT g1_num_nodes, g2_num_nodes;
    bool use_attr = false, directed = false;
    int inner_iters = 5;
    int outer_iters = 10;
    bool sparsify = false;
    using milliseconds = std::chrono::milliseconds;
    std::vector<std::pair<std::string, long long>> phase_timings;
    auto synced_now = []() {
        CHECK_CUDA(cudaDeviceSynchronize());
        return std::chrono::high_resolution_clock::now();
    };
    
    if (argc < 3) {
        printf("provide input directory and sparsifying option \n");
        return 0;
    }
    else {
        data_path = argv[1];
        sparsify = std::stoi(argv[2]);
    }
    load_config(data_path, g1_num_nodes, g2_num_nodes, use_attr, directed, inner_iters, outer_iters);
    DEBUG_ONLY(printf("graph1 #nodes %d -- graph2 #nodes %d -- directed %d -- use node attributes %d\n\n", g1_num_nodes, g2_num_nodes, directed, use_attr));

    CudaCsrMatrix graph1_py;
    CudaCsrMatrix graph2_py;
    int avg_nnz1_per_n1 = 0;
    int avg_nnz2_per_n2 = 0;
    {
        std::vector<double> g1_edgelist = readFileToVector(data_path + "/graph1.txt", directed);
        graph1_py = make_cuda_csr_from_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes);
        avg_nnz1_per_n1 = static_cast<int>(g1_edgelist.size()) / g1_num_nodes;
    }
    {
        std::vector<double> g2_edgelist = readFileToVector(data_path + "/graph2.txt", directed);
        graph2_py = make_cuda_csr_from_edgelist(g2_edgelist, g2_num_nodes, g2_num_nodes);
        avg_nnz2_per_n2 = static_cast<int>(g2_edgelist.size()) / g2_num_nodes;
    }
    std::pair<std::vector<int>, std::vector<int>> anchor_pairs, anchor_pairs_test;
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
        DEBUG_ONLY(printf("g1_feat size %ld  g2_feat size %ld\n", g1_feat.size(), g2_feat.size()));
    }
    int l = anchor_pairs.first.size();
    DEBUG_ONLY(printf("graph1 #nodes %d -- graph2 #nodes %d -- avg nnz1/n1 %d -- avg nnz2/n2  %d -- use node attributes %d --- # of anchor links(l) %ld\n\n",
        g1_num_nodes, g2_num_nodes, avg_nnz1_per_n1, avg_nnz2_per_n2, use_attr, anchor_pairs.first.size()));

    auto time_0 = synced_now();
    //normalize graph adjacencies
    CudaCsrMatrix graph1_row_normalized = row_normalize_graph_cuda(graph1_py, avg_nnz1_per_n1);
    CudaCsrMatrix graph2_row_normalized = row_normalize_graph_cuda(graph2_py, avg_nnz2_per_n2);
    auto time_1 = synced_now();
    phase_timings.push_back({"row normalizing graphs", std::chrono::duration_cast<milliseconds>(time_1 - time_0).count()});
    DEBUG_ONLY(std::cout << "row normalizing graphs : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() << " ms" << std::endl);

    //Free graphs - we will use normalized graph from now on
    destroy_cuda_csr(graph1_py);
    destroy_cuda_csr(graph2_py);

    double* d_g1_rwr = nullptr;
    double* d_g2_rwr = nullptr;
    double* d_g1_feat = nullptr;
    double* d_g2_feat = nullptr;
    double* d_cross_rwr_dist = nullptr;
    double* d_cross_attr_dist = nullptr;
    double* d_cross_dist = nullptr;
    BatchRwrContext rwr_context;
    const size_t g1_rwr_size = static_cast<size_t>(g1_num_nodes) * static_cast<size_t>(l);
    const size_t g2_rwr_size = static_cast<size_t>(g2_num_nodes) * static_cast<size_t>(l);
    const size_t cross_size = static_cast<size_t>(g1_num_nodes) * static_cast<size_t>(g2_num_nodes);

    CHECK_CUDA(cudaMalloc(&d_g1_rwr, g1_rwr_size * sizeof(double)));
    auto g1_rwr_start = synced_now();
    get_batch_rwr_scores_cusparse_device(graph1_row_normalized, anchor_pairs.first, g1_num_nodes, 0.15, 100, d_g1_rwr, rwr_context);
    auto time_2 = synced_now();
    phase_timings.push_back({"graph1 RWR", std::chrono::duration_cast<milliseconds>(time_2 - g1_rwr_start).count()});
    DEBUG_ONLY(std::cout << "graph1 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-g1_rwr_start).count() << " ms" << std::endl);
    CHECK_CUDA(cudaMalloc(&d_g2_rwr, g2_rwr_size * sizeof(double)));
    auto g2_rwr_start = synced_now();
    get_batch_rwr_scores_cusparse_device(graph2_row_normalized, anchor_pairs.second, g2_num_nodes, 0.15, 100, d_g2_rwr, rwr_context);
    auto time_3 = synced_now();
    phase_timings.push_back({"graph2 RWR", std::chrono::duration_cast<milliseconds>(time_3 - g2_rwr_start).count()});
    DEBUG_ONLY(std::cout << "graph2 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_3-g2_rwr_start).count() << " ms" << std::endl);
    destroy_batch_rwr_context(rwr_context);

    //row normalize rwr for both graphs
    l2_norm_inplace_device(d_g1_rwr, g1_num_nodes, l);
    l2_norm_inplace_device(d_g2_rwr, g2_num_nodes, l);

    CHECK_CUDA(cudaMalloc(&d_cross_rwr_dist, cross_size * sizeof(double)));
    auto cross_rwr_start = synced_now();
    get_normalized_neg_exp_dist_device(d_g1_rwr, d_g2_rwr, g1_num_nodes, g2_num_nodes, l, d_cross_rwr_dist);
    if (use_attr) {
        const size_t g1_feat_dim = g1_feat.size() / static_cast<size_t>(g1_num_nodes);
        const size_t g2_feat_dim = g2_feat.size() / static_cast<size_t>(g2_num_nodes);
        CHECK_CUDA(cudaMalloc(&d_g1_feat, g1_feat.size() * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_g2_feat, g2_feat.size() * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(d_g1_feat, g1_feat.data(), g1_feat.size() * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_g2_feat, g2_feat.data(), g2_feat.size() * sizeof(double), cudaMemcpyHostToDevice));
        l2_norm_inplace_device(d_g1_feat, g1_num_nodes, g1_feat_dim);
        l2_norm_inplace_device(d_g2_feat, g2_num_nodes, g2_feat_dim);

        CHECK_CUDA(cudaMalloc(&d_cross_attr_dist, cross_size * sizeof(double)));
        get_normalized_neg_exp_dist_device(d_g1_feat, d_g2_feat, g1_num_nodes, g2_num_nodes, static_cast<int>(g1_feat_dim), d_cross_attr_dist);

        cublasHandle_t blend_handle = nullptr;
        CHECK_CUBLAS(cublasCreate(&blend_handle));
        const double alpha = 0.5;
        CHECK_CUBLAS(cublasDaxpy(blend_handle, static_cast<int>(cross_size), &alpha, d_cross_rwr_dist, 1, d_cross_attr_dist, 1));
        CHECK_CUBLAS(cublasDestroy(blend_handle));

        CHECK_CUDA(cudaFree(d_cross_rwr_dist));
        d_cross_rwr_dist = d_cross_attr_dist;
        d_cross_attr_dist = nullptr;
    } else {
        cublasHandle_t scale_handle = nullptr;
        CHECK_CUBLAS(cublasCreate(&scale_handle));
        const double scale = 1.5;
        CHECK_CUBLAS(cublasDscal(scale_handle, static_cast<int>(cross_size), &scale, d_cross_rwr_dist, 1));
        CHECK_CUBLAS(cublasDestroy(scale_handle));
    }
    //make the anchor links cross rwr dist zero
    anchor_cost_zero_device(d_cross_rwr_dist, g1_num_nodes, g2_num_nodes, anchor_pairs, 0.0);
    auto time_4 = synced_now();
    phase_timings.push_back({"cross RWR distance", std::chrono::duration_cast<milliseconds>(time_4 - cross_rwr_start).count()});
    DEBUG_ONLY(std::cout << "cross_rwr_dist computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_4-cross_rwr_start).count() << " ms" << std::endl);

    CudaCsrMatrix intra_rwr1_masked_sddmm;
    auto time_7 = synced_now();
    if (use_attr) intra_rwr1_masked_sddmm = masked_sddmm_intra_cost_cusparse_device(d_g1_feat, graph1_row_normalized, g1_num_nodes, static_cast<int>(g1_feat.size() / static_cast<size_t>(g1_num_nodes)));
    else intra_rwr1_masked_sddmm = masked_sddmm_intra_cost_cusparse_device(d_g1_rwr, graph1_row_normalized, g1_num_nodes, l);
    auto time_8 = synced_now();
    phase_timings.push_back({"graph1 intra distance", std::chrono::duration_cast<milliseconds>(time_8 - time_7).count()});
    DEBUG_ONLY(std::cout << "g1 rwr dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_8-time_7).count() << " ms" << std::endl);

    CudaCsrMatrix intra_rwr2_masked_sddmm;
    auto g2_intra_start = synced_now();
    if (use_attr) intra_rwr2_masked_sddmm = masked_sddmm_intra_cost_cusparse_device(d_g2_feat, graph2_row_normalized, g2_num_nodes, static_cast<int>(g2_feat.size() / static_cast<size_t>(g2_num_nodes)));
    else intra_rwr2_masked_sddmm = masked_sddmm_intra_cost_cusparse_device(d_g2_rwr, graph2_row_normalized, g2_num_nodes, l);
    auto time_9 = synced_now();
    phase_timings.push_back({"graph2 intra distance", std::chrono::duration_cast<milliseconds>(time_9 - g2_intra_start).count()});
    DEBUG_ONLY(std::cout << "g2 rwr dist " << std::chrono::duration_cast<std::chrono::milliseconds>(time_9-g2_intra_start).count() << " ms" << std::endl);

    //Free some memory 
    g1_feat.clear(); g1_feat.shrink_to_fit();
    g2_feat.clear(); g2_feat.shrink_to_fit();
    CHECK_CUDA(cudaFree(d_g1_rwr));
    CHECK_CUDA(cudaFree(d_g2_rwr));
    d_g1_rwr = nullptr;
    d_g2_rwr = nullptr;
    if (d_g1_feat != nullptr) CHECK_CUDA(cudaFree(d_g1_feat));
    if (d_g2_feat != nullptr) CHECK_CUDA(cudaFree(d_g2_feat));
    d_g1_feat = nullptr;
    d_g2_feat = nullptr;

    //Compute cross_dist
    CHECK_CUDA(cudaMalloc(&d_cross_dist, cross_size * sizeof(double)));
    auto time_10 = synced_now();
    get_product_rwr_mat_cusparse_device(graph1_row_normalized, graph2_row_normalized, d_cross_rwr_dist, g1_num_nodes, g2_num_nodes, 0.15, 0.75, d_cross_dist);
    auto time_11 = synced_now();
    phase_timings.push_back({"cross distance propagation", std::chrono::duration_cast<milliseconds>(time_11 - time_10).count()});
    DEBUG_ONLY(std::cout << "cross_dist computation completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_11-time_10).count() << " ms" << std::endl);
    CHECK_CUDA(cudaFree(d_cross_rwr_dist));
    d_cross_rwr_dist = nullptr;
    auto optimization_start = synced_now();
    std::vector<double> S = con_prox_pt_opt_device(graph1_row_normalized, graph2_row_normalized, g1_num_nodes, g2_num_nodes, avg_nnz1_per_n1, avg_nnz2_per_n2, d_cross_dist, intra_rwr1_masked_sddmm, 
        intra_rwr2_masked_sddmm, inner_iters, outer_iters, 5e-3, 5e-4, 5e-4, 1e-5, anchor_pairs, sparsify);
    auto time_12 = synced_now();
    phase_timings.push_back({"optimization", std::chrono::duration_cast<milliseconds>(time_12 - optimization_start).count()});
    DEBUG_ONLY(std::cout << "con_prox_pt_opt completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_12-optimization_start).count() << " ms" << std::endl);
    CHECK_CUDA(cudaFree(d_cross_dist));
    d_cross_dist = nullptr;
    long long total_algorithm_ms = 0;
    for (const auto& phase : phase_timings) {
        total_algorithm_ms += phase.second;
    }
    std::cout << "algorithm completed in " << total_algorithm_ms << " ms" << std::endl;
    std::cout << "phase timings:" << std::endl;
    for (const auto& phase : phase_timings) {
        std::cout << "  " << phase.first << ": " << phase.second << " ms" << std::endl;
    }

    std::unordered_map<int, double>  hits = hits_ks_mean_scores( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test, std::vector<int>{1, 10, 30, 50});

    for (const auto& pair : hits) {
        std::cout << "hit@ " << pair.first << " : " << pair.second << std::endl;
    }

    double mrr = mrr_mean_score_cpp( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test);
    std::cout << "mrr : " << mrr << std::endl;

    destroy_cuda_csr(intra_rwr2_masked_sddmm);
    destroy_cuda_csr(intra_rwr1_masked_sddmm);
    destroy_cuda_csr(graph2_row_normalized);
    destroy_cuda_csr(graph1_row_normalized);

    return 0;
}

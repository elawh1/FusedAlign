#include <iostream>
#include <vector>
#ifdef MKL_AVAILABLE
#include <mkl.h>
#include <mkl_vml.h>
#include <omp.h>
#endif
#include <string>
#include <sstream>
// #include <mat.h>
// #include <cmath>
#include <algorithm>
#include <fstream>
#include "helper.h"
#include <ctime> 
#include <chrono>

void load_config(const std::string& data_path, MKL_INT& g1_num_nodes, MKL_INT& g2_num_nodes, bool& use_attr, int& inner_iters, int& outer_iters) {
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
            else if (key == "outer_iter") outer_iters = std::stoi(eq);
            else if (key == "inner_iter") inner_iters = std::stoi(eq);
        }
    }
}


int main(int argc, char* argv[]){ 

    std::string data_path;
    MKL_INT g1_num_nodes, g2_num_nodes;
    bool use_attr = false;
    int inner_iters = 5;
    int outer_iters = 10;
    
    if (argc < 2) {
        printf("provide input directory \n");
        return 0;
    }
    else {
        data_path = argv[1];
    }
    load_config(data_path, g1_num_nodes, g2_num_nodes, use_attr, inner_iters, outer_iters);
    printf("MKL threads: %d\n", mkl_get_max_threads());
    printf("MKL dynamic: %d\n", mkl_get_dynamic());
    // printf("MKL threading layer: %d\n", mkl_get_threading_layer());
    printf("OMP max threads: %d\n", omp_get_max_threads());
    printf("OMP in parallel region? %d\n", omp_in_parallel());

    std::vector<double> g1_edgelist = readFileToVector(data_path + "/graph1.txt");
    sparse_matrix_t graph1_py = make_mat_from_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes);        //row major csr
    std::vector<double> g2_edgelist = readFileToVector(data_path + "/graph2.txt");
    sparse_matrix_t graph2_py = make_mat_from_edgelist(g2_edgelist, g2_num_nodes, g2_num_nodes);        //row major csr
    std::vector<double> anchors = readFileToVector(data_path + "/anchor_links.txt");
    std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchor_pairs = get_anchor_pairs(anchors);
    std::vector<double> anchors_test = readFileToVector(data_path + "/test_pairs.txt");
    std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchor_pairs_test = get_anchor_pairs(anchors_test);
    std::vector<double> g1_feat, g2_feat;
    if(use_attr) {
        g1_feat = readFileToVector(data_path + "/graph1_feat.txt");
        g2_feat = readFileToVector(data_path + "/graph2_feat.txt");
        printf("g1_feat size %ld  g2_feat size %ld\n", g1_feat.size(), g2_feat.size());
    }
    int l = anchor_pairs.first.size();
    printf("graph1 #nodes %d -- graph2 #nodes %d -- use node attributes %d --- # of anchor links %ld\n\n", g1_num_nodes, g2_num_nodes, use_attr, anchor_pairs.first.size());

    auto time_0 = std::chrono::high_resolution_clock::now();
    std::vector<double>  g1_rwr = get_batch_rwr_scores(graph1_py, anchor_pairs.first, g1_num_nodes, 0.15, 100, true );
    auto time_1 = std::chrono::high_resolution_clock::now();
    std::cout << "graph1 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() << " ms" << std::endl;
    std::vector<double>  g2_rwr = get_batch_rwr_scores(graph2_py, anchor_pairs.second, g2_num_nodes, 0.15, 100, true );
    auto time_2 = std::chrono::high_resolution_clock::now();
    std::cout << "graph2 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << " ms" << std::endl;

    // compare(g1_rwr, data_path + "/rwr_emb1.txt", "g1_rwr");
    // compare(g2_rwr, data_path + "/rwr_emb2.txt", "g2_rwr");
    auto time_3 = std::chrono::high_resolution_clock::now();
    std::vector<double> cross_rwr_dist = get_normalized_neg_exp_dist (g1_rwr, g2_rwr, g1_num_nodes, g2_num_nodes, l);

    //copy rwr dist to attr dist because we dont have attributes
    std::vector<double> cross_attr_dist;
    if (use_attr) {
        cross_attr_dist = get_normalized_neg_exp_dist (g1_feat, g2_feat, g1_num_nodes, g2_num_nodes, g1_feat.size() / g1_num_nodes);
    }
    else {
        cross_attr_dist = cross_rwr_dist;
    }
    
    cblas_daxpy(cross_attr_dist.size(), 0.5, cross_rwr_dist.data(), 1, cross_attr_dist.data(), 1);
    //make the anchor links cross rwr dist zero
    anchor_cost_zero(cross_attr_dist, g1_num_nodes, g2_num_nodes, anchor_pairs, 0.0);
    auto time_5 = std::chrono::high_resolution_clock::now();
    std::cout << "cross_rwr_dist computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_5-time_3).count() << " ms" << std::endl;
    // printf("cross_rwr_dist size is %ld\n", cross_rwr_dist.size());

    //compute intra network embedding and mask it with graph using spmm
    std::vector<double> intra_rwr_dist1, intra_rwr_dist2;
    if (use_attr) {
        intra_rwr_dist1 = get_normalized_neg_exp_dist (g1_feat, g1_feat, g1_num_nodes, g1_num_nodes, (g1_feat.size()) / g1_num_nodes);
    }
    else {
        intra_rwr_dist1 = get_normalized_neg_exp_dist (g1_rwr, g1_rwr, g1_num_nodes, g1_num_nodes, l);
    }
    
    std::vector<double> intra_rwr_dist1_masked(g1_num_nodes * g1_num_nodes);                // intra_rwr_dist1_masked = graph1_adj * intra_rwr_dist1
    {
        std::vector<double> graph1_py_dense(g1_num_nodes * g1_num_nodes, 0.0);
        sparse_index_base_t indexing;
        MKL_INT rows, cols, *row_ptr_st, *row_ptr_end, *col_ind;
        double *values;
        mkl_sparse_d_export_csr(graph1_py, &indexing, &rows, &cols, &row_ptr_st, &row_ptr_end, &col_ind, &values);
        for (MKL_INT i = 0; i < rows; ++i) {
            for (MKL_INT p = row_ptr_st[i]; p < row_ptr_end[i]; ++p) {
                MKL_INT j = col_ind[p];
                if (indexing == SPARSE_INDEX_BASE_ONE) j -= 1;  // adjust to 0-based
                graph1_py_dense[i * cols + j] = 1.0;                    // mask pattern = 1
            }
        }
        vdMul(g1_num_nodes*g1_num_nodes, graph1_py_dense.data(), intra_rwr_dist1.data(), intra_rwr_dist1_masked.data());
    }
    auto time_6 = std::chrono::high_resolution_clock::now();
    std::cout << "intra_rwr_dist graph1 completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_6-time_5).count() << " ms" << std::endl;

    // printf("intra_rwr_dist1_masked computed\n");
    // compare(intra_rwr_dist1_masked, data_path + "/intra_dist1.txt", "intra_rwr_dist1_masked");

        
    if (use_attr) {
        intra_rwr_dist2 = get_normalized_neg_exp_dist (g2_feat, g2_feat, g2_num_nodes, g2_num_nodes,  (g2_feat.size()) / g2_num_nodes);
    }
    else {
        intra_rwr_dist2 = get_normalized_neg_exp_dist (g2_rwr, g2_rwr, g2_num_nodes, g2_num_nodes, l);
    }
    std::vector<double> intra_rwr_dist2_masked(g2_num_nodes * g2_num_nodes);                // intra_rwr_dist1_masked = graph1_adj * intra_rwr_dist1
    {
        std::vector<double> graph2_py_dense(g2_num_nodes * g2_num_nodes, 0.0);
        sparse_index_base_t indexing;
        MKL_INT rows, cols, *row_ptr_st, *row_ptr_end, *col_ind;
        double *values;
        mkl_sparse_d_export_csr(graph2_py, &indexing, &rows, &cols, &row_ptr_st, &row_ptr_end, &col_ind, &values);
        for (MKL_INT i = 0; i < rows; ++i) {
            for (MKL_INT p = row_ptr_st[i]; p < row_ptr_end[i]; ++p) {
                MKL_INT j = col_ind[p];
                if (indexing == SPARSE_INDEX_BASE_ONE) j -= 1;  // adjust to 0-based
                graph2_py_dense[i * cols + j] = 1.0;                    // mask pattern = 1
            }
        }
        vdMul(g2_num_nodes * g2_num_nodes, graph2_py_dense.data(), intra_rwr_dist2.data(), intra_rwr_dist2_masked.data());
    }
    auto time_7 = std::chrono::high_resolution_clock::now();
    std::cout << "intra_rwr_dist graph2 completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_7-time_6).count() << " ms" << std::endl;

    // printf("intra_rwr_dist2_masked computed\n");
    // compare(intra_rwr_dist2_masked, data_path + "/intra_dist2.txt", "intra_rwr_dist2_masked");

    //Compute cross_dist
    // std::vector<double> cross_dist = get_product_rwr_mat(graph1_py, graph2_py, cross_rwr_dist, g1_num_nodes, g2_num_nodes, 0.15, 0.75); 
    // print_sparse_matrix_info(graph1_py, "graph1_py");
    // print_sparse_matrix_info(graph2_py, "graph2_py");
    std::vector<double> cross_dist = get_product_rwr_mat(graph1_py, graph2_py, cross_attr_dist, g1_num_nodes, g2_num_nodes, 0.15, 0.75); 
    auto time_8 = std::chrono::high_resolution_clock::now();
    std::cout << "cross_dist computation completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_8-time_7).count() << " ms" << std::endl;
    // printf("cross_dist computed\n");
    // compare(cross_dist, data_path + "/cross_dist.txt", "cross_dist");
    std::vector<double> S = con_prox_pt_opt(data_path, graph1_py, graph2_py, g1_num_nodes, g2_num_nodes, cross_dist, intra_rwr_dist1_masked, intra_rwr_dist2_masked, inner_iters, outer_iters, 5e-3, 5e-4, 5e-4, 1e-5, anchor_pairs, anchor_pairs_test, anchors_test);
    auto time_9 = std::chrono::high_resolution_clock::now();
    std::cout << "con_prox_pt_opt completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_9-time_8).count() << " ms" << std::endl;
    std::cout << "algorithm completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_9-time_0).count() << " ms" << std::endl;

    compare(S, data_path + "/S.txt", "S");

    std::unordered_map<int, double>  hits = hits_ks_mean_scores( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test, std::vector<int>{1, 10, 30, 50});

    for (const auto& pair : hits) {
        std::cout << "hit@ " << pair.first << " : " << pair.second << std::endl;
    }

    double mrr = mrr_mean_score_cpp( S, g1_num_nodes, g2_num_nodes, anchor_pairs_test);
    std::cout << "mrr : " << mrr << std::endl;

    return 0;
}


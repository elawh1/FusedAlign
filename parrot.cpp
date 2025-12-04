#include <iostream>
#include <vector>
#include <mkl.h>
#include <mat.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "helper.h"
#include <ctime> 
#include <chrono>


int main(){	

    /*
	const char* matFileName = "phone-email.mat";
	const char* variableName = "email";
	MATFile* pmat = matOpen(matFileName, "r");
	if (!pmat) {
		std::cerr << "Error opening mat file \n";
		return 0;
	}
	mxArray* email = matGetVariable(pmat, "email");
	mxArray* phone = matGetVariable(pmat, "phone");
	mxArray* gnd = matGetVariable(pmat, "gnd");
	mxArray* H = matGetVariable(pmat, "H");

	if (!mxIsDouble(email) || mxIsComplex(email) || !mxIsDouble(phone) || mxIsComplex(phone) ||   !mxIsDouble(gnd) || mxIsComplex(gnd) || !mxIsDouble(H) || mxIsComplex(H) ) {
		std::cerr << "Variable is not a real double-precision array." << std::endl;
		mxDestroyArray(email); // Destroy the mxArray
		mxDestroyArray(phone);
		mxDestroyArray(H);
		mxDestroyArray(gnd);
	}
	mwSize numRowsG1 = mxGetM(email); // Number of rows
	mwSize numColsG1 = mxGetN(email); // Number of columns	
	mwSize numRowsG2 = mxGetM(phone); // Number of rows
	mwSize numColsG2 = mxGetN(phone); // Number of columns	


	double* data = mxGetPr(email);
	mwSize num_elements = mxGetNumberOfElements(email);
	std::vector<double> graph1(data, data+num_elements);
	data = mxGetPr(phone);
	num_elements = mxGetNumberOfElements(phone);
	std::vector<double> graph2(data, data+num_elements);
	data = mxGetPr(gnd);
	num_elements = mxGetNumberOfElements(gnd);
	std::vector<double> gndTruth(data, data+num_elements);
	data = mxGetPr(H);
	num_elements = mxGetNumberOfElements(H);
	std::vector<double> anchors(data, data+num_elements);

	MKL_INT g1_num_nodes = std::sqrt(graph1.size());
 	MKL_INT g2_num_nodes = std::sqrt(graph2.size());

	printf("graph1 size : %d   graph2 size : %d   gndTruth size  %d   anchors size %d\n", graph1.size(), graph2.size(), gndTruth.size(), anchors.size());
	printf("graph1 num nodes: %d   graph2 num nodes : %d \n", g1_num_nodes, g2_num_nodes);
	
		
    MKLCSC g1_csc = make_csc_from_vec(graph1, g1_num_nodes, g1_num_nodes);
    MKLCSC g2_csc = make_csc_from_vec(graph2, g2_num_nodes, g2_num_nodes);
        

    */
    MKL_INT g1_num_nodes = 1000;
    MKL_INT g2_num_nodes = 1003;
    
    std::vector<double> g1_edgelist = readFileToVector("graph1.txt");
    sparse_matrix_t graph1_py = make_mat_from_edgelist(g1_edgelist, g1_num_nodes, g1_num_nodes);        //row major csr
    std::vector<double> g2_edgelist = readFileToVector("graph2.txt");
    sparse_matrix_t graph2_py = make_mat_from_edgelist(g2_edgelist, g2_num_nodes, g2_num_nodes);        //row major csr

    std::vector<double> anchors = readFileToVector("anchor_links-2.txt");
    std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchor_pairs = get_anchor_pairs(anchors);
    // for(int i=0;i<100;i++) {
    //     printf("%d %d \n", anchor_pairs.first[i], anchor_pairs.second[i]);
    // }
    printf("num of anchor links : %d \n", anchor_pairs.first.size());

    int l = anchor_pairs.first.size();

    std::vector<double> anchors_test = readFileToVector("test_pairs.txt");
    std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchor_pairs_test = get_anchor_pairs(anchors_test);
 
    printf("num of  anchor links in test_pairs : %d \n", anchor_pairs.first.size());
    
    // std::vector<double>  g1_rwr = get_batch_rwr_scores(g1_csc.spMat, anchor_pairs.first, g1_num_nodes, 0.15, 100, true );
    // std::vector<double>  g2_rwr = get_batch_rwr_scores(g2_csc.spMat, anchor_pairs.second, g2_num_nodes, 0.15, 100, true );
    time_t t0 = std::time(NULL);
    auto time_0 = std::chrono::high_resolution_clock::now();
    std::vector<double>  g1_rwr = get_batch_rwr_scores(graph1_py, anchor_pairs.first, g1_num_nodes, 0.15, 100, true );
    time_t t1 = std::time(NULL);
    auto time_1 = std::chrono::high_resolution_clock::now();
    std::vector<double>  g2_rwr = get_batch_rwr_scores(graph2_py, anchor_pairs.second, g2_num_nodes, 0.15, 100, true );
    time_t t2 = std::time(NULL);
    auto time_2 = std::chrono::high_resolution_clock::now();
    printf("graph1 rwr computer in %d sec and graph2 rwr computed in %d sec\n", t1-t0, t2-t1);
    std::cout << "graph1 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_1-time_0).count() << " ms" << std::endl;
    std::cout << "graph2 rwr computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(time_2-time_1).count() << " ms" << std::endl;


	
    // read planetalign imp output (column-major, one value per line)
    std::vector<double> py(g1_rwr.size());
    std::ifstream fin("rwr_emb1.txt");
    for (size_t i = 0; i < g1_rwr.size() && fin; ++i) fin >> py[i];
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < g1_rwr.size(); ++i) {
        double a = py[i], b = g1_rwr[i];
        // if( i < 100)
        //     std::cout << py[i] << " " << g1_rwr[i] << "\n";
        double da = std::abs(a - b);
        max_abs = std::max(max_abs, da);
        double denom = std::max({std::abs(a), std::abs(b), 1e-12});
        max_rel = std::max(max_rel, da / denom);
    }
    std::cout << "max_abs = " << max_abs << "  max_rel = " << max_rel << "\n";

    std::vector<double> py2(g2_rwr.size());
    std::ifstream fin2("rwr_emb2.txt");
    for (size_t i = 0; i < g2_rwr.size() && fin2; ++i) fin2 >> py2[i];
    for (size_t i = 0; i < g2_rwr.size(); ++i) {
        double a = py2[i], b = g2_rwr[i];
        // if( i < 100)
        //     std::cout << py2[i] << " " << g2_rwr[i] << "\n";
        double da = std::abs(a - b);
        max_abs = std::max(max_abs, da);
        double denom = std::max({std::abs(a), std::abs(b), 1e-12});
        max_rel = std::max(max_rel, da / denom);
    }
    std::cout << "max_abs = " << max_abs << "  max_rel = " << max_rel << "\n";


    
    std::vector<double> cross_rwr_dist = get_normalized_neg_exp_dist (g1_rwr, g2_rwr, g1_num_nodes, g2_num_nodes, l);
    printf("cross_rwr_dist   computed size of it %d\n", cross_rwr_dist.size());

    //copy rwr dist to attr dist because we dont have attributes
    std::vector<double> cross_attr_dist = cross_rwr_dist;


    // cross_rwr_dist is the same as cross_node_dist in parrot implementation
    // crossCost  = alpha * crossCost  + (1 - alpha) * C1 * C1
    cblas_daxpy(cross_rwr_dist.size(), 0.5, cross_attr_dist.data(), 1, cross_rwr_dist.data(), 1);
    //make the anchor links cross rwr dist zero
    // crossCost  = crossCost .* (~H)
    anchor_cost_zero(cross_rwr_dist, g1_num_nodes, g2_num_nodes, anchor_pairs, 0.0);

    //compute intra network embedding and mask it with graph using spmm
    matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    std::vector<double> intra_rwr_dist1 = get_normalized_neg_exp_dist (g1_rwr, g1_rwr, g1_num_nodes, g1_num_nodes, l);
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
    printf("intra_rwr_dist1_masked computed\n");

    // auto st = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph1_py, descr, SPARSE_LAYOUT_ROW_MAJOR, intra_rwr_dist1.data(), g1_num_nodes, l, 0.0, intra_rwr_dist1.data(), l);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm failed");
    // for(int i=0; i<g1_num_nodes*g1_num_nodes; i++) {
    //     if(intra_rwr_dist1_masked[i] != 0.0) {       printf(" --intra_rwr_dist1_masked[i]  %.2f\n", intra_rwr_dist1_masked[i]);}
    // }
        
    std::vector<double> intra_rwr_dist2 = get_normalized_neg_exp_dist (g2_rwr, g2_rwr, g2_num_nodes, g2_num_nodes, l);
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
    printf("intra_rwr_dist2_masked computed\n");


    std::vector<double> py4(intra_rwr_dist1_masked.size());
    std::ifstream fin4("intra_dist1.txt");
    for (size_t i = 0; i < intra_rwr_dist1_masked.size() && fin4; ++i) fin4 >> py4[i];
    max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < intra_rwr_dist1_masked.size(); ++i) {
        double a = py4[i], b = intra_rwr_dist1_masked[i];
        // if((a != 0.0 && b == 0.0)  || (a == 0.0 && b != 0.0))
        //     printf(" intra_rwr_dist1[i]  %.2f   py4[i]   %.2f\n", intra_rwr_dist1_masked[i], py4[i]);
        // std::cout << "i " << i << " --  graph1_py_dense[i] " << graph1_py_dense[i] << "   -  " << py4[i] << " " << intra_rwr_dist1[i] << "\n";
        double da = std::abs(a - b);
        max_abs = std::max(max_abs, da);
        double denom = std::max({std::abs(a), std::abs(b), 1e-12});
        max_rel = std::max(max_rel, da / denom);
    }
    std::cout << "intra_rwr_dist1_masked max_abs = " << max_abs << "  max_rel = " << max_rel << "\n";


    
    std::vector<double> py5(intra_rwr_dist2_masked.size());
    std::ifstream fin5("intra_dist2.txt");
    for (size_t i = 0; i < intra_rwr_dist2_masked.size() && fin5; ++i) fin5 >> py5[i];
    max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < intra_rwr_dist2_masked.size(); ++i) {
        double a = py5[i], b = intra_rwr_dist2_masked[i];
        // if( i < 100)
        //     std::cout << py3[i] << " " << g1_rwr[i] << "\n";
        double da = std::abs(a - b);
        max_abs = std::max(max_abs, da);
        double denom = std::max({std::abs(a), std::abs(b), 1e-12});
        max_rel = std::max(max_rel, da / denom);
    }
    std::cout << "intra_rwr_dist2_masked  max_abs = " << max_abs << "  max_rel = " << max_rel << "\n";
    

    // st = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, graph2_py, descr, SPARSE_LAYOUT_COLUMN_MAJOR, intra_rwr_dist2.data(), g2_num_nodes, l, 0.0, intra_rwr_dist2.data(), l);
    // if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl_sparse_d_mm failed");

    //make intra_rwr_dist1 sparse but in dense format
    // std::vector<double> graph1_py_dense(g1_num_nodes * g1_num_nodes, 0.0);
    // sparse_index_base_t indexing;
    // MKL_INT rows, cols, *row_ptr_st, *row_ptr_end, *col_ind;
    // double *values;
    // mkl_sparse_d_export_csr(graph1_py, &indexing, &rows, &cols, &row_ptr_st, &row_ptr_end, &col_ind, &values);
    // for (MKL_INT i = 0; i < rows; ++i) {
    //     for (MKL_INT p = row_ptr_st[i]; p < row_ptr_end[i]; ++p) {
    //         MKL_INT j = col_ind[p];
    //         if (indexing == SPARSE_INDEX_BASE_ONE) j -= 1;  // adjust to 0-based
    //         graph1_py_dense[i * cols + j] = 1.0;                    // mask pattern = 1
    //     }
    // }
    // vdMul(g1_num_nodes*g1_num_nodes, graph1_py_dense.data(), intra_rwr_dist1.data(), intra_rwr_dist1.data());

    //make intra_rwr_dist1 sparse but in dense format
    // std::vector<double> graph2_py_dense(g2_num_nodes * g2_num_nodes, 0.0);
    // // sparse_index_base_t indexing;
    // // MKL_INT rows, cols, *row_ptr_st, *row_ptr_end, *col_ind;
    // // double *values;
    // mkl_sparse_d_export_csr(graph2_py, &indexing, &rows, &cols, &row_ptr_st, &row_ptr_end, &col_ind, &values);
    // for (MKL_INT i = 0; i < rows; ++i) {
    //     for (MKL_INT p = row_ptr_st[i]; p < row_ptr_end[i]; ++p) {
    //         MKL_INT j = col_ind[p];
    //         if (indexing == SPARSE_INDEX_BASE_ONE) j -= 1;  // adjust to 0-based
    //         graph2_py_dense[i * cols + j] = 1.0;                    // mask pattern = 1
    //     }
    // }
    // vdMul(g2_num_nodes*g2_num_nodes, graph2_py_dense.data(), intra_rwr_dist2.data(), intra_rwr_dist2.data());
    
    //Compute cross_dist
    std::vector<double> cross_dist = get_product_rwr_mat(graph1_py, graph2_py, cross_rwr_dist, g1_num_nodes, g2_num_nodes, 0.15, 0.75); 
    printf("cross_dist computed\n");

    
    //check dist matrices
    std::vector<double> py6(cross_dist.size());
    std::ifstream fin6("cross_dist.txt");
    for (size_t i = 0; i < cross_dist.size() && fin6; ++i) fin6 >> py6[i];
    max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < cross_dist.size(); ++i) {
        double a = py6[i], b = cross_dist[i];
        // if( i < 100)
        //     std::cout << py6[i] << " " << cross_dist[i] << "\n";
        double da = std::abs(a - b);
        max_abs = std::max(max_abs, da);
        double denom = std::max({std::abs(a), std::abs(b), 1e-12});
        max_rel = std::max(max_rel, da / denom);
    }
    std::cout << "cross_dist max_abs = " << max_abs << "  max_rel = " << max_rel << "\n";
    

    
    
    int inner_iters = 5;
    int outer_iters = 10;
    std::vector<double> S = con_prox_pt_opt(graph1_py, graph2_py, g1_num_nodes, g2_num_nodes, cross_dist, intra_rwr_dist1_masked, intra_rwr_dist2, inner_iters, outer_iters, 5e-3, 5e-4, 5e-4, 1e-5, anchor_pairs, anchor_pairs_test, anchors_test);

    std::vector<double> py7(S.size());
    std::ifstream fin7("intra_dist2.txt");
    for (size_t i = 0; i < S.size() && fin7; ++i) fin7 >> py7[i];
    max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < S.size(); ++i) {
        double a = py7[i], b = S[i];
        // if( i < 1000 && S[i]  != 0.0)
        //     std::cout << py7[i] << " " << S[i] << "\n";
        double da = std::abs(a - b);
        max_abs = std::max(max_abs, da);
        double denom = std::max({std::abs(a), std::abs(b), 1e-12});
        max_rel = std::max(max_rel, da / denom);
        // if(da > 1e-2)
        //     std::cout <<"diff " << da << " vals " << a << " " << b << "\n";
        
    }
    std::cout << "S max_abs  = " << max_abs << "  max_rel = " << max_rel << "\n";

    double maxRes = py7[0], minRes = py7[0];
    for (int i = 1; i < g2_num_nodes * g1_num_nodes; ++i) {
        if (py7[i] > maxRes) maxRes = py7[i];
        if (py7[i] < minRes) minRes = py7[i];
    }
    std::cerr << "minRes = " << minRes << ", maxRes = " << maxRes << "\n";
    
	// mxDestroyArray(email);
	// mxDestroyArray(phone);
	// mxDestroyArray(gnd);
	// mxDestroyArray(H);
	// matClose(pmat);

	return 0;
}

#include <iostream>
#include <vector>
#include <mkl.h>
#include <mat.h>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>


struct MKLCSC {
    sparse_matrix_t spMat = nullptr;
    std::vector<MKL_INT> col_start, col_end, row_ind;
    std::vector<double>  vals;
    ~MKLCSC() { if (spMat) mkl_sparse_destroy(spMat); }
};


std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> get_anchor_pairs(std::vector<double>& H_vec, MKL_INT m, MKL_INT n) {
    std::vector<MKL_INT> anchor_g1;
    std::vector<MKL_INT> anchor_g2;

    for(MKL_INT i = 0; i < m; i++) {
        for(MKL_INT j = 0; j < n; j++) {
            if(H_vec[(size_t)j * m + i] != (double) 0.0) {
                anchor_g1.push_back(i);
                anchor_g2.push_back(j);
            }
        }
    }
    return {anchor_g1, anchor_g2};
}

void check_cuda_err( cudaError_t err, char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
    else    return;
}


int main(){	
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

	int g1_num_nodes = std::sqrt(graph1.size());
 	int g2_num_nodes = std::sqrt(graph2.size());
    

	printf("graph1 size : %d   graph2 size : %d   gndTruth size  %d   anchors size %d\n", graph1.size(), graph2.size(), gndTruth.size(), anchors.size());
	printf("graph1 num nodes: %d   graph2 num nodes : %d \n", g1_num_nodes, g2_num_nodes);
	
		
    // MKLCSC g1_csc = make_csc_from_vec(graph1, g1_num_nodes, g1_num_nodes);
    // MKLCSC g2_csc = make_csc_from_vec(graph2, g2_num_nodes, g2_num_nodes);

    std::pair<std::vector<MKL_INT>, std::vector<MKL_INT>> anchor_pairs = get_anchor_pairs(anchors, g1_num_nodes, g2_num_nodes);
    int l = anchor_pairs.first.size();
    printf("num of anchor links : %d \n", l);

    //Check gpu size
    size_t freeB, totalB; cudaMemGetInfo(&freeB, &totalB);
    printf("free=%zu, total=%zu\n", freeB, totalB);

    //Copy data to cuda
    double *d_graph1 = nullptr, *d_graph2 = nullptr, *d_anchors = nullptr;
    check_cuda_err( cudaMalloc( (void**)&d_graph1, graph1.size() * sizeof(double)), "cudaMalloc d_graph1 failed");
    check_cuda_err( cudaMalloc( (void**)&d_graph2, graph2.size() * sizeof(double)), "cudaMalloc d_graph2 failed");
    check_cuda_err( cudaMalloc( (void**)&d_anchors, anchors.size() * sizeof(double)), "cudaMalloc anchors failed");
    check_cuda_err( cudaMemcpy( d_graph1, graph1.data(), graph1.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy graph1 failed");
    check_cuda_err( cudaMemcpy( d_graph2, graph2.data(), graph2.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy graph2 failed");
    check_cuda_err( cudaMemcpy( d_anchors, anchors.data(), anchors.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy anchors failed");
    printf("copy data to gpu successful\n");


    


    /*
    std::vector<double>  g1_rwr = get_batch_rwr_scores(g1_csc.spMat, anchor_pairs.first, g1_num_nodes, 0.15, 100, true );
    std::vector<double>  g2_rwr = get_batch_rwr_scores(g2_csc.spMat, anchor_pairs.second, g2_num_nodes, 0.15, 100, true );
    printf("graph1 and graph2 rwr computed \n");

    std::vector<double> cross_rwr_dist = get_normalized_neg_exp_dist (g1_rwr, g2_rwr, g1_num_nodes, g2_num_nodes, l);
    printf("cross_rwr_dist computed \n");

    //copy rwr dist to attr dist because we dont have attributes
    std::vector<double> cross_attr_dist = cross_rwr_dist;

    // cross_rwr_dist is the same as cross_node_dist in parrot implementation
    cblas_daxpy(n1 * n2, alpha, cross_attr_dist.data(), 1, cross_rwr_dist.data(), 1);

    
    */

	mxDestroyArray(email);
	mxDestroyArray(phone);
	mxDestroyArray(gnd);
	mxDestroyArray(H);
	matClose(pmat);


	return 0;
}

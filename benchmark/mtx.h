#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>


int read_mtx (std::string mtx_path, std::vector<double>& g1_edgelist, MKL_INT& g1_num_nodes) {
    std::ifstream mtx_file(mtx_path);
    if (!mtx_file.is_open()) {
        std::cerr << "Error: Cannot open file " << mtx_path << std::endl;
        return 1;
    }

    std::string line;
    bool isDirected = true; // Default: directed



    // // Skip comments and header lines
    // while (std::getline(mtx_file, line)) {
    //     if (line.empty())
    //         continue;
    //     if (line[0] == '%')
    //         continue;
    //     // Found the size line (e.g., "36692 36692 183831")
    //     break;
    // }

    // // Now line contains the size info
    // int rows = 0, cols = 0, nnz = 0;
    // {
    //     std::istringstream iss(line);
    //     iss >> rows >> cols >> nnz;
    // }
    // printf("rows %d cols %d nnz %d\n", rows, cols, nnz);
    // g1_num_nodes = rows;
    // g1_edgelist.reserve(2 * nnz); // preallocate for speed

    // // Read edge entries
    // while (std::getline(mtx_file, line)) {
    //     if (line.empty()) continue;
    //     if (line[0] == '%') continue;

    //     std::istringstream iss(line);
    //     int n1, n2;
    //     if (!(iss >> n1 >> n2)) continue;
    //     // Matrix Market format uses 1-based indices, so if you want 0-based, uncomment below:
    //     n1--; n2--;
    //     g1_edgelist.push_back(static_cast<double>(n1));
    //     g1_edgelist.push_back(static_cast<double>(n2));
    // }
    // printf("g1_edgelist size %d last el %e %e first %e %e\n", g1_edgelist.size(), g1_edgelist[2 * nnz - 1], g1_edgelist[2 * nnz - 2], g1_edgelist[0], g1_edgelist[1] );
    // mtx_file.close();

    bool size_line_found = false;
    int rows = 0, cols = 0, nnz = 0;

    while (std::getline(mtx_file, line)) {
        // Lowercase for easier comparison
        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        if (lower_line.find("% kind:") != std::string::npos) {
            if (lower_line.find("undirected") != std::string::npos) {
                isDirected = false;
            } else if (lower_line.find("directed") != std::string::npos) {
                isDirected = true;
            }
            printf("is directed %d\n", isDirected );
        }
        
        if (line.empty() || line[0] == '%') continue;

        // Matrix size line, looks like "36692 36692 183831"
        if (!size_line_found) {
            std::istringstream iss(line);
            iss >> rows >> cols >> nnz;
            size_line_found = true;
            g1_num_nodes = rows;
            printf("rows %d cols %d nnz %d\n", rows, cols, nnz);
            continue;
        }

        // Edge lines
        std::istringstream iss(line);
        int n1, n2;
        if (!(iss >> n1 >> n2)) continue;
        n1--; n2--;
        // Add edge(s)
        g1_edgelist.push_back(static_cast<double>(n1));
        g1_edgelist.push_back(static_cast<double>(n2));

        if (!isDirected && n1 != n2) { // undirected, add reverse edge
            g1_edgelist.push_back(static_cast<double>(n2));
            g1_edgelist.push_back(static_cast<double>(n1));
        }
    }

    return 1;
}

// int main() {
//     std::string mtx_path;
//     std::cout << "Enter .mtx file path: ";
//     std::getline(std::cin, mtx_path);

//     std::ifstream mtx_file(mtx_path);
//     if (!mtx_file.is_open()) {
//         std::cerr << "Error: Cannot open file " << mtx_path << std::endl;
//         return 1;
//     }

//     std::string line;
//     // Skip comments and header lines
//     while (std::getline(mtx_file, line)) {
//         if (line.empty())
//             continue;
//         if (line[0] == '%')
//             continue;
//         // Found the size line (e.g., "36692 36692 183831")
//         break;
//     }

//     // Now line contains the size info
//     int rows = 0, cols = 0, nnz = 0;
//     {
//         std::istringstream iss(line);
//         iss >> rows >> cols >> nnz;
//     }

//     std::vector<double> g1_edgelist;
//     g1_edgelist.reserve(2 * nnz); // preallocate for speed

//     // Read edge entries
//     while (std::getline(mtx_file, line)) {
//         if (line.empty()) continue;
//         if (line[0] == '%') continue;

//         std::istringstream iss(line);
//         int n1, n2;
//         if (!(iss >> n1 >> n2)) continue;
//         // Matrix Market format uses 1-based indices, so if you want 0-based, uncomment below:
//         // n1--; n2--;
//         g1_edgelist.push_back(static_cast<double>(n1));
//         g1_edgelist.push_back(static_cast<double>(n2));
//     }

//     mtx_file.close();

//     std::cout << "g1_edgelist size: " << g1_edgelist.size() << std::endl;
//     // Print first 10 edges as example
//     for (size_t i = 0; i < g1_edgelist.size() && i < 20; i += 2) {
//         std::cout << g1_edgelist[i] << " " << g1_edgelist[i + 1] << std::endl;
//     }

//     // You now have the edge list in g1_edgelist.
//     return 0;
// }
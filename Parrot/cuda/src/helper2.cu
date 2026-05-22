#include "cuda_helper.cuh"

#include <sstream>

void load_config(const std::string& data_path,
                 int& g1_num_nodes,
                 int& g2_num_nodes,
                 bool& use_attr,
                 bool& directed,
                 int& inner_iters,
                 int& outer_iters) {
    std::ifstream fin(data_path + "/config.txt");
    if (!fin) {
        throw std::runtime_error("cannot open config.txt");
    }
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::string key;
        std::string value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            if (key == "g1_num_nodes") g1_num_nodes = std::stoi(value);
            else if (key == "g2_num_nodes") g2_num_nodes = std::stoi(value);
            else if (key == "use_attr") use_attr = (value == "1");
            else if (key == "directed") directed = (value == "1");
            else if (key == "outer_iter") outer_iters = std::stoi(value);
            else if (key == "inner_iter") inner_iters = std::stoi(value);
        }
    }
}

std::vector<element_type> readFileToVector(const std::string& filename, bool directed) {
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("cannot open input file: " + filename);
    }

    std::vector<element_type> data;
    if (directed) {
        element_type val = static_cast<element_type>(0.0);
        while (fin >> val) {
            data.push_back(val);
        }
        return data;
    }

    element_type u = static_cast<element_type>(0.0);
    element_type v = static_cast<element_type>(0.0);
    while (fin >> u >> v) {
        data.push_back(u);
        data.push_back(v);
        if (u != v) {
            data.push_back(v);
            data.push_back(u);
        }
    }
    return data;
}

std::pair<std::vector<int>, std::vector<int>> get_anchor_pairs(std::vector<element_type>& H_vec) {
    if (H_vec.size() % 2 != 0) {
        throw std::invalid_argument("anchor vector must contain pairs");
    }
    std::pair<std::vector<int>, std::vector<int>> anchors;
    const size_t count = H_vec.size() / 2;
    anchors.first.resize(count);
    anchors.second.resize(count);
    for (size_t i = 0; i < count; ++i) {
        anchors.first[i] = static_cast<int>(H_vec[2 * i]);
        anchors.second[i] = static_cast<int>(H_vec[2 * i + 1]);
    }
    return anchors;
}

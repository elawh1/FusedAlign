# PARROT-Imp (C++ / CUDA)

This repository contains a C++ (and optional CUDA) implementation of **PARROT**
(Position-Aware Regularized Optimal Transport for Network Alignment), adapted from the original
Python implementation.

⚠️ **Status:** C++ implementation works and accuracy matches the python version.

---

# Datasets for PARROT

This directory contains datasets used to run the PARROT graph alignment
algorithm.

Each dataset should be placed in its own subdirectory.

---

## Dataset 

Each dataset directory should contain:

- `config.txt` : contains configuration to run parrot for that dataset
- `graph1.txt` : adjacency matrix of graph 1
- `graph2.txt` : adjacency matrix of graph 2
- `anchor_links.txt` : known anchor correspondences
- `graph1_feat.txt` (optional): node features for graph 1
- `graph2_feat.txt` (optional): node features for graph 2
- `test_pairs.txt` : test pairs of nodes
- `S.txt` : expected result of similarity matrix computed by NA algorithm

---

## How to run

```bash
cd Parrot
make
./parrot [path_to_dataset_directory]

## Example
```bash
./parrot ../data/phone-email
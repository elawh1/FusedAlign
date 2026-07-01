# FastAlign

FastAlign is a scalable, sparsity-aware framework for optimal-transport (OT)-based
network alignment. It preserves the original OT alignment formulation while
reinterpreting its computation as a set of recurring sparse-dense operations,
enabling substantially faster and more memory-efficient alignment on large,
sparse real-world graphs.

FastAlign achieves comparable alignment accuracy to state-of-the-art OT-based
methods while running significantly faster — up to ~9x on CPU and ~33x on GPU
compared to the strongest OT-based baseline.

## Requirements

- C++17 compiler (g++)
- BLAS backend: OpenBLAS or Intel MKL
- OpenMP
- CUDA toolkit + cuSPARSE (GPU only)

## Building and Running

### CPU

```bash
cd FastAlign
make
./fastalign <data_path> <sparsify_flag>
```

### GPU

```bash
cd FastAlign/cuda
make
./cuda_alignment <data_path> <sparsify_flag>
```

- `data_path` — path to a dataset directory
- `sparsify_flag` — `1` to enable sparsity-aware optimizations, `0` to disable

### Output

Running FastAlign prints alignment accuracy metrics (Hits@1/10/30/50 and MRR)
along with a per-phase runtime breakdown to stdout.

## Included Datasets

Real-world and synthetic benchmark datasets are included under `data/` and
`ER-Graph/`.

## Ablation Studies

`CPP-baseline`, `CUDA-baseline`, `cpp-dense`, `cuda-dense`, `FA-CPP-dense`,
`FA-CPP-sparsity`, `FA-CPP-fusion-sparsity`, `FA-CUDA-sparsity`,
`fastalign-cpp-dense`, `fastalign-cpp-sparse` are standalone ablation variants
used to isolate the contribution of individual optimizations, reported in the
paper's ablation study. Each builds and runs the same way as above:

```bash
cd FastAlign/<variant_name>
make
./fastalign <data_path> <sparsify_flag>       # or ./cuda_alignment for CUDA variants
```

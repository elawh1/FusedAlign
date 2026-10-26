# FusedAlign

FusedAlign is a scalable, sparsity-aware framework for optimal-transport-based network alignment. It accelerates alignment through fused sparse–dense computations, reduced intermediate data movement, and optimized CPU and GPU execution.

## Requirements

- C++17 compiler (g++)
- BLAS backend: OpenBLAS or Intel MKL
- OpenMP
- CUDA toolkit + cuSPARSE (GPU only)

## Building and Running

### CPU

```bash
cd FusedAlign
make
./fusedalign <data_path> <sparsify_flag>
```

### GPU

```bash
cd FusedAlign/cuda
make
./cuda_alignment <data_path> <sparsify_flag>
```

- `data_path` — path to a dataset directory
- `sparsify_flag` — `1` to enable sparsity-aware optimizations, `0` to disable

### Output

Running FusedAlign prints alignment accuracy metrics (Hits@1/10/30/50 and MRR)
along with a per-phase runtime breakdown to stdout.

## Included Datasets

Real-world and synthetic benchmark datasets are included under `data/` and
`ER-Graph/`.

## Ablation Studies

`CPP-baseline`, `CUDA-baseline`, `cpp-dense`, `cuda-dense`, `FA-CPP-dense`,
`FA-CPP-sparsity`, `FA-CPP-fusion-sparsity`, `FA-CUDA-sparsity`,
`fusedalign-cpp-dense`, `fusedalign-cpp-sparse` are standalone ablation variants
used to isolate the contribution of individual optimizations, reported in the
paper's ablation study. Each builds and runs the same way as above:

```bash
cd FusedAlign/<variant_name>
make
./fusedalign <data_path> <sparsify_flag>       # or ./cuda_alignment for CUDA variants
```

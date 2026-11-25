
# Parallel Reduction in CUDA

## Project Overview
This project implements parallel reduction (sum of an array) using CUDA. It compares CPU reduction, a naive CUDA implementation, and an optimised CUDA implementation using warp shuffle operations.

## Objective
To understand how parallel reduction improves computation performance on a GPU and compare different CUDA implementation strategies.

## Hardware and Software
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA Toolkit: 13.5
- Platform: Ubuntu on WSL2
- Language: C++ and CUDA
- Python libraries: NumPy, Pandas, Matplotlib

## Input
- Number of elements: 2^24 (16,777,216)
- Data type: float
- Input size: 64 MiB

## Implementations
1. CPU Reduction: Sequential summation using C++.
2. Naive CUDA Reduction: Interleaved addressing, shared memory and synchronisation.
3. Optimised CUDA Reduction: Processes multiple elements per thread and uses warp shuffle operations.

## Performance Results

| Implementation | Block Size | Average Time (ms) |
|---|---:|---:|
| CPU | - | 6.267350 |
| Naive CUDA | 256 | 0.313508 |
| Optimised CUDA | 64 | 0.191210 |
| Optimised CUDA | 128 | 0.170462 |
| Optimised CUDA | 256 | 0.171179 |
| Optimised CUDA | 512 | 0.204147 |
| Optimised CUDA | 1024 | 0.171749 |

The best measured configuration was the optimised CUDA implementation with 128 threads per block.

## Performance Analysis
- The optimised 128-thread configuration achieved approximately 1.84x speedup over the naive CUDA implementation.
- The measured CPU-to-GPU timing ratio was approximately 36.77x. This compares CPU computation time with CUDA kernel timing; host-device transfers are excluded.
- The results show that block size affects GPU performance.

## Timing Methodology
- 5 warm-up runs.
- 30 measured runs.
- CUDA events used for GPU timing.
- Output reset is included in the measured GPU interval.
- Host-to-device input transfer is excluded.

## Validation
GPU sums are compared with a CPU reference sum using an absolute error tolerance of 5.0.

Run the validation script:

```bash
python scripts/validate_results.py
```

## Project Structure

```text
cuda-parallel-reduction/
├── docs/
├── graphs/
├── results/
├── scripts/
├── src/
├── .gitignore
└── README.md
```

## How to Compile and Run

Compile the CPU implementation:

```bash
g++ -O2 -std=c++17 src/cpu_reduction.cpp -o bin/cpu_reduction
./bin/cpu_reduction
```

Compile the naive CUDA implementation:

```bash
nvcc -O2 src/naive_reduction.cu -o bin/naive_reduction
./bin/naive_reduction
```

Compile and run the optimised implementation with 128 threads:

```bash
nvcc -O2 -DBLOCK_SIZE=128 src/optimized_reduction.cu -o bin/optimized_128
./bin/optimized_128
```

Change `BLOCK_SIZE` to 64, 256, 512 or 1024 to test other configurations.

Create the binary directory before compiling:

```bash
mkdir -p bin
```

## Applications
Parallel reduction is useful in:
- Machine learning loss calculations
- Vector norms and normalisation
- Statistical computations
- Scientific computing

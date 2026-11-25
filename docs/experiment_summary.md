# Parallel Reduction in CUDA

## 1. Experiment Objective

To implement and compare CPU sequential reduction,
naive CUDA reduction, and optimised CUDA reduction.

## 2. Hardware Configuration

- GPU: NVIDIA RTX 4500 Ada Generation
- GPU Memory: 24570 MiB
- CUDA Version: 13.5
- Environment: Ubuntu WSL 2

## 3. Input Configuration

- Number of elements: 2^24
- Total elements: 16,777,216
- Data type: float
- Input size: 64 MiB

## 4. Implementations

### CPU Reduction
Sequential summation using a C++ loop.
Accumulation uses double precision.

### Naive CUDA Reduction
- Interleaved addressing
- Shared memory
- Thread synchronisation
- Atomic addition of block results

### Optimised CUDA Reduction
- First add during load
- Warp shuffle using __shfl_down_sync
- Shared memory for warp partial sums
- Atomic addition of block results

## 5. Timing Methodology

- 5 warm-up iterations
- 30 measured iterations
- CUDA events used for GPU timing
- Average execution time calculated

Note: GPU timing includes output reset using
cudaMemset before each kernel launch. It excludes
host-to-device and device-to-host data transfers.

## 6. Validation

CPU reference sum: 8472487.359375

All six CUDA configurations passed validation
with an absolute error tolerance of 5.0.

## 7. Best Measured Configuration

Block size: 128 threads
Number of blocks: 65536
Average measured time: 0.170462 ms
Effective bandwidth estimate: 393.69 GB/s

## 8. Performance Improvement

Naive CUDA time: 0.313508 ms
Optimised CUDA time: 0.170462 ms

Naive-to-optimised speedup: 1.84x

CPU-to-optimised kernel-time ratio: 36.77x

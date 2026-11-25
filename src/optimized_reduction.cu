#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <cstdlib>
#include <cmath>

using namespace std;

#define N (1 << 24)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define WARMUP 5
#define ITERATIONS 30

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Warp-level reduction using shuffle instructions
__device__ float warpReduce(float value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    return value;
}

// Optimised parallel reduction kernel
__global__ void optimizedReduction(
    const float *input,
    float *output,
    int n)
{
    __shared__ float shared[BLOCK_SIZE / 32];

    unsigned int tid = threadIdx.x;

    // First add during load
    unsigned int index =
        blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    if (index < n)
        sum = input[index];

    if (index + blockDim.x < n)
        sum += input[index + blockDim.x];

    // Warp-level reduction
    sum = warpReduce(sum);

    // Store one partial sum per warp
    if ((tid % 32) == 0)
    {
        shared[tid / 32] = sum;
    }

    __syncthreads();

    // First warp reduces the warp partial sums
    if (tid < 32)
    {
        sum = (tid < BLOCK_SIZE / 32)
                  ? shared[tid]
                  : 0.0f;

        sum = warpReduce(sum);
    }

    // One final result per block
    if (tid == 0)
    {
        atomicAdd(output, sum);
    }
}

int main()
{
    float *h_input, *d_input, *d_output;

    size_t bytes = N * sizeof(float);

    h_input = new float[N];

    // Generate the same input data as the CPU program
    for (int i = 0; i < N; i++)
    {
        h_input[i] =
            static_cast<float>((i % 100) + 1) / 100.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + (BLOCK_SIZE * 2) - 1)
                 / (BLOCK_SIZE * 2);

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < WARMUP; i++)
    {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

        optimizedReduction<<<blocks, BLOCK_SIZE>>>(
            d_input, d_output, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed execution
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

        optimizedReduction<<<blocks, BLOCK_SIZE>>>(
            d_input, d_output, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_time = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_time, start, stop));

    float average_time = total_time / ITERATIONS;

    float result;

    CUDA_CHECK(cudaMemcpy(
        &result, d_output, sizeof(float),
        cudaMemcpyDeviceToHost));

    cout << fixed << setprecision(6);

    cout << "\nOptimised CUDA Reduction Results\n";
    cout << "--------------------------------\n";
    cout << "Input elements: " << N << endl;
    cout << "Number of blocks: " << blocks << endl;
    cout << "Block size: " << BLOCK_SIZE << endl;
    cout << "GPU sum: " << result << endl;
    cout << "Warm-up runs: " << WARMUP << endl;
    cout << "Measured runs: " << ITERATIONS << endl;
    cout << "Average measured time: "
         << average_time << " ms" << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_input;

    return 0;
}

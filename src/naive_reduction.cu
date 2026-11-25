#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace std;

#define N (1 << 24)
#define BLOCK_SIZE 256
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

// Naive interleaved addressing reduction
__global__ void naiveReduction(float *input, float *output, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (index < n) ? input[index] : 0.0f;

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(output, sdata[0]);
    }
}

int main()
{
    float *h_input, *d_input, *d_output;

    size_t bytes = N * sizeof(float);

    h_input = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_input[i] = static_cast<float>((i % 100) + 1) / 100.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes,
                          cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up runs
    for (int i = 0; i < WARMUP; i++)
    {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

        naiveReduction<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

        naiveReduction<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));

    float average_time = total_time / ITERATIONS;

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float),
                          cudaMemcpyDeviceToHost));

    cout << fixed << setprecision(6);

    cout << "\nNaive CUDA Reduction Results\n";
    cout << "----------------------------\n";
    cout << "Input elements: " << N << endl;
    cout << "Number of blocks: " << blocks << endl;
    cout << "Block size: " << BLOCK_SIZE << endl;
    cout << "GPU sum: " << result << endl;
    cout << "Warm-up runs: " << WARMUP << endl;
    cout << "Measured runs: " << ITERATIONS << endl;
    cout << "Average kernel time: " << average_time << " ms" << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_input;

    return 0;
}

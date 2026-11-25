
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPU()
{
    printf("Hello from CUDA GPU! Thread ID: %d\n", threadIdx.x);
}

int main()
{
    helloGPU<<<1, 5>>>();

    cudaDeviceSynchronize();

    printf("CUDA GPU test completed successfully!\n");

    return 0;
}

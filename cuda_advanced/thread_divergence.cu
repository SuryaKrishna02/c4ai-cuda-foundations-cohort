#include <stdio.h>

__device__ float PathA(float *src)
{
    // Dummy computation for PathA
    return src[threadIdx.x] * 2.0f;
}

__device__ float PathB(float *src)
{
    // Dummy computation for PathB
    return src[threadIdx.x] + 1.0f;
}

__global__ void TestDivergence(float *dst, float *src)
{
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    float value = 0.0f;

    if (threadIdx.x % 2 == 0)
    {
        // Threads executing PathA are active while threads
        // executing PathB are inactive.
        value = PathA(src);
    }
    else
    {
        // Threads executing PathB are active while threads
        // executing PathA are inactive.
        value = PathB(src);
    }
    // Threads converge here again and execute in parallel.
    dst[index] = value;
}

int main()
{
    const int arraySize = 256;
    const int arrayBytes = arraySize * sizeof(float);

    float h_src[arraySize];
    float h_dst[arraySize];

    // Initialize input data
    for (int i = 0; i < arraySize; i++)
    {
        h_src[i] = static_cast<float>(i);
    }

    // Device memory allocation
    float *d_src;
    float *d_dst;
    cudaMalloc((void **)&d_src, arrayBytes);
    cudaMalloc((void **)&d_dst, arrayBytes);

    // Copy data from host to device
    cudaMemcpy(d_src, h_src, arrayBytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256; // or any appropriate value
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // Launch the kernel
    TestDivergence<<<gridSize, blockSize>>>(d_dst, d_src);

    // Copy data from device to host
    cudaMemcpy(h_dst, d_dst, arrayBytes, cudaMemcpyDeviceToHost);

    // Print some of the results
    for (int i = 0; i < 10; i++)
    {
        printf("h_dst[%d] = %f\n", i, h_dst[i]);
    }

    // Clean up device memory
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}

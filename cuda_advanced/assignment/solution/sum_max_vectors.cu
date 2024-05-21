#include <cuda_runtime.h>
#include <iostream>

__global__ void findMax(float *d_vec, float *d_max, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared memory
    if (i < n)
    {
        sdata[tid] = d_vec[i];
    }
    else
    {
        sdata[tid] = -FLT_MAX;
    }
    __syncthreads();

    // Perform parallel reduction to find max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && i + s < n)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        d_max[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int N = 2048;
    const int bytes = N * sizeof(float);
    const int numThreads = 1024;
    const int numBlocks = (N + numThreads - 1) / numThreads;

    float *h_vec1 = new float[N];
    float *h_vec2 = new float[N];

    // Initialize vectors with some values
    for (int i = 0; i < N; ++i)
    {
        h_vec1[i] = static_cast<float>(i);
        h_vec2[i] = static_cast<float>(i + 1);
    }

    float *d_vec1, *d_vec2, *d_max1, *d_max2;
    cudaMalloc(&d_vec1, bytes);
    cudaMalloc(&d_vec2, bytes);
    cudaMalloc(&d_max1, sizeof(float) * N);
    cudaMalloc(&d_max2, sizeof(float) * N);

    cudaMemcpy(d_vec1, h_vec1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, bytes, cudaMemcpyHostToDevice);

    findMax<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(d_vec1, d_max1, N);
    findMax<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(d_vec2, d_max2, N);

    cudaDeviceSynchronize();

    float *h_max1 = new float[numBlocks];
    float *h_max2 = new float[numBlocks];
    cudaMemcpy(h_max1, d_max1, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max2, d_max2, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);

    float max1 = -FLT_MAX, max2 = -FLT_MAX;
    for (int i = 0; i < numBlocks; ++i)
    {
        max1 = max(max1, h_max1[i]);
        max2 = max(max2, h_max2[i]);
    }

    float sum_max = max1 + max2;
    std::cout << "Sum of maximum elements: " << sum_max << std::endl;

    free(h_vec1);
    free(h_vec2);
    free(h_max1);
    free(h_max2);

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_max1);
    cudaFree(d_max2);

    return 0;
}
#include <cuda_runtime.h>
#include <iostream>

__global__ void findMaxMultipleThread(float *d_vec1, float *d_vec2, float *d_max1, float *d_max2, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float max1 = -FLT_MAX;
        float max2 = -FLT_MAX;

        for (int i = 0; i < n; ++i)
        {
            max1 = max(max1, d_vec1[i]);
            max2 = max(max2, d_vec2[i]);
        }

        d_max1[0] = max1;
        d_max2[0] = max2;
    }
}

int main()
{
    const int N = 2048;
    const int bytes = N * sizeof(float);
    const int numThreads = 1024;
    const int numBlocks = 1;

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

    findMaxMultipleThread<<<numBlocks, numThreads>>>(d_vec1, d_vec2, d_max1, d_max2, N);

    float h_max1, h_max2;
    cudaMemcpy(&h_max1, d_max1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max2, d_max2, sizeof(float), cudaMemcpyDeviceToHost);

    float sum_max = h_max1 + h_max2;
    std::cout << "Sum of maximum elements: " << sum_max << std::endl;

    free(h_vec1);
    free(h_vec2);

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_max1);
    cudaFree(d_max2);

    return 0;
}
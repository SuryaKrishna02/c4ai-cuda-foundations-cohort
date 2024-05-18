#include <cuda_runtime.h>

#define N 1024
#define nStreams 4
#define nThreads 256

__global__ void kernel(float *a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = a[idx] * 2.0f;
}

int main()
{
    float *a_h, *a_d;
    size_t size = N * sizeof(float);
    cudaStream_t stream[nStreams];

    // Allocate memory on the host
    a_h = (float *)malloc(size);
    // Allocate memory on the device
    cudaMalloc((void **)&a_d, size);

    // Initialize host array and create streams
    for (int i = 0; i < N; i++)
    {
        a_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    size_t chunkSize = N * sizeof(float) / nStreams;
    for (int i = 0; i < nStreams; i++)
    {
        size_t offset = i * N / nStreams;
        cudaMemcpyAsync(a_d + offset, a_h + offset, chunkSize, cudaMemcpyHostToDevice, stream[i]);
        kernel<<<N / (nThreads * nStreams), nThreads, 0, stream[i]>>>(a_d + offset);
    }

    // Copy the results back to the host
    for (int i = 0; i < nStreams; i++)
    {
        size_t offset = i * N / nStreams;
        cudaMemcpyAsync(a_h + offset, a_d + offset, chunkSize, cudaMemcpyDeviceToHost, stream[i]);
    }

    // Wait for all streams to complete
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    // Free memory
    cudaFree(a_d);
    free(a_h);

    return 0;
}

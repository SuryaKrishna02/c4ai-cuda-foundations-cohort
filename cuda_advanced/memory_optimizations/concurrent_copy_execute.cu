#include <cuda_runtime.h>
#include <iostream>

// Error checking macro
#define cudaCheckError()                                                                                                  \
    {                                                                                                                     \
        cudaError_t e = cudaGetLastError();                                                                               \
        if (e != cudaSuccess)                                                                                             \
        {                                                                                                                 \
            std::cerr << "CUDA Error " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(e) << std::endl; \
            exit(EXIT_FAILURE);                                                                                           \
        }                                                                                                                 \
    }

__global__ void kernel(float *data)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2.0f; // operation: double the value
}

int main()
{
    float *a_h, *a_d;
    float *otherData_d;
    size_t size = 1024 * sizeof(float); // Example size
    dim3 grid(1);
    dim3 block(1024);

    // Allocate host memory
    a_h = (float *)malloc(size);
    if (!a_h)
    {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate device memory
    cudaMalloc((void **)&a_d, size);
    cudaCheckError();
    cudaMalloc((void **)&otherData_d, size);
    cudaCheckError();

    // Initialize host data
    for (int i = 0; i < 1024; ++i)
    {
        a_h[i] = static_cast<float>(i);
    }

    cudaStream_t stream1, stream2;

    // Create streams
    cudaStreamCreate(&stream1);
    cudaCheckError();
    cudaStreamCreate(&stream2);
    cudaCheckError();

    // Asynchronously copy data to the device on stream1
    cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream1);
    cudaCheckError();

    // Launch kernel on stream2
    kernel<<<grid, block, 0, stream2>>>(otherData_d);
    cudaCheckError();

    // Synchronize streams to ensure all operations are complete
    cudaStreamSynchronize(stream1);
    cudaCheckError();
    cudaStreamSynchronize(stream2);
    cudaCheckError();

    // Clean up
    cudaStreamDestroy(stream1);
    cudaCheckError();
    cudaStreamDestroy(stream2);
    cudaCheckError();
    cudaFree(a_d);
    cudaCheckError();
    cudaFree(otherData_d);
    cudaCheckError();
    free(a_h);

    return 0;
}
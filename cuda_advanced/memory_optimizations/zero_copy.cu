#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(float *a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[idx] = idx;
}

int main()
{
    float *a_h, *a_map;
    cudaDeviceProp prop;
    int nBytes = 1024 * sizeof(float);
    int gridSize = 1;
    int blockSize = 1024;

    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory)
        exit(0);

    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&a_map, a_h, 0);

    kernel<<<gridSize, blockSize>>>(a_map);
    cudaDeviceSynchronize();

    for (int i = 0; i < 1024; i++)
    {
        std::cout << a_h[i] << " ";
    }
    std::cout << std::endl;

    cudaFreeHost(a_h);
    return 0;
}

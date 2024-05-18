#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to process data
__global__ void kernel(float *a_d, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        a_d[idx] *= 2.0f; // Example operation: doubling each element
    }
}

// CPU function to perform some operation (placeholder)
void cpuFunction()
{
    printf("CPU function executed.\n");
}

// Function to initialize host data
void initializeHostData(float *a_h, int size)
{
    for (int i = 0; i < size; ++i)
    {
        a_h[i] = static_cast<float>(i);
    }
}

int main()
{
    int size = 1024; // Size of the array
    int grid = 4;    // Number of blocks (1024 elements / 256 threads per block)
    int block = 256; // Number of threads per block

    // Allocate memory on the device
    float *a_d;
    cudaMalloc((void **)&a_d, size * sizeof(float));

    // Allocate memory on the host
    float *a_h = (float *)malloc(size * sizeof(float));

    // Initialize host data
    initializeHostData(a_h, size);

    // Copy data from host to device asynchronously
    cudaMemcpyAsync(a_d, a_h, size * sizeof(float), cudaMemcpyHostToDevice, 0);

    // Launch the kernel
    kernel<<<grid, block>>>(a_d, size);

    // Execute CPU function
    cpuFunction();

    // Synchronize to ensure the kernel has completed
    cudaDeviceSynchronize();

    // Copy data back from device to host to check results
    cudaMemcpy(a_h, a_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the results
    for (int i = 0; i < size; ++i)
    {
        if (a_h[i] != static_cast<float>(i) * 2.0f)
        {
            printf("Error at index %d: expected %f, got %f\n", i, static_cast<float>(i) * 2.0f, a_h[i]);
            break;
        }
    }
    printf("Verification completed.\n");

    // Cleanup: Free device and host memory
    cudaFree(a_d);
    free(a_h);

    // Error handling
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
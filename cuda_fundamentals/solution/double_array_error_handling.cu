#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define CHECK_CUDA_ERROR(err)                                          \
    {                                                                  \
        if (err != cudaSuccess)                                        \
        {                                                              \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    }

__global__ void initialize_array(int *array, int size)
{
    // Calculate the index for the current thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        array[index] = index;
    }
    // COMPLETE THIS
}

int main()
{
    const int array_size = 10;
    int *d_array;

    // Allocate memory on GPU
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_array, array_size * sizeof(int)));

    // Launch the CUDA kernel to initialize the array
    initialize_array<<<1, array_size>>>(d_array, array_size);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy data from device to host
    int h_array[array_size];
    CHECK_CUDA_ERROR(cudaMemcpy(h_array, d_array, array_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the initialized array
    cout << "Initialized Array:" << endl;
    for (int i = 0; i < array_size; ++i)
    {
        cout << h_array[i] << " ";
    }
    cout << endl;

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_array));

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return 0;
}
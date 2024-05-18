#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void kernel(float *d_odata, float *d_idata, int size_x, int size_y, int NUM_REPS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_x * size_y)
    {
        for (int i = 0; i < NUM_REPS; ++i)
        {
            d_odata[idx] = d_idata[idx] + 1.0f;
        }
    }
}

int main()
{
    cudaEvent_t start, stop;
    float time;

    // Error handling
    cudaError_t err;

    // Create CUDA events
    err = cudaEventCreate(&start);
    if (err != cudaSuccess)
    {
        cerr << "Failed to create start event: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess)
    {
        cerr << "Failed to create stop event: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Example data size and kernel launch configuration
    int size_x = 1024;
    int size_y = 1024;
    int NUM_REPS = 10;
    int grid = (size_x * size_y + 255) / 256;
    int threads = 256;

    // Allocate device memory
    float *d_idata, *d_odata;
    err = cudaMalloc((void **)&d_idata, size_x * size_y * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "Failed to allocate device memory for d_idata: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc((void **)&d_odata, size_x * size_y * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "Failed to allocate device memory for d_odata: " << cudaGetErrorString(err) << endl;
        cudaFree(d_idata);
        return -1;
    }

    // Record the start event
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess)
    {
        cerr << "Failed to record start event: " << cudaGetErrorString(err) << endl;
        cudaFree(d_idata);
        cudaFree(d_odata);
        return -1;
    }

    // Launch the kernel
    kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, NUM_REPS);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "Failed to launch kernel: " << cudaGetErrorString(err) << endl;
        cudaFree(d_idata);
        cudaFree(d_odata);
        return -1;
    }

    // Record the stop event
    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess)
    {
        cerr << "Failed to record stop event: " << cudaGetErrorString(err) << endl;
        cudaFree(d_idata);
        cudaFree(d_odata);
        return -1;
    }

    // Synchronize the stop event
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
    {
        cerr << "Failed to synchronize stop event: " << cudaGetErrorString(err) << endl;
        cudaFree(d_idata);
        cudaFree(d_odata);
        return -1;
    }

    // Calculate the elapsed time
    err = cudaEventElapsedTime(&time, start, stop);
    if (err != cudaSuccess)
    {
        cerr << "Failed to calculate elapsed time: " << cudaGetErrorString(err) << endl;
        cudaFree(d_idata);
        cudaFree(d_odata);
        return -1;
    }

    cout << "Elapsed time: " << time << " ms" << endl;

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
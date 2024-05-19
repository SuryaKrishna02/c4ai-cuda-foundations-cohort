#include <cuda_runtime.h>
#include <iostream>

#define TILE_DIM 32 // Define TILE_DIM as the warp size, typically 32 for current GPUs

__global__ void bankConflictMultiply(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float transposedTile[TILE_DIM][TILE_DIM + 1];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x * blockDim.x + threadIdx.y) * TILE_DIM +
          threadIdx.x];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++)
    {
        sum += aTile[threadIdx.y][i] * transposedTile[i][threadIdx.x];
    }
    c[row * M + col] = sum;
}

int main()
{
    // Define matrix dimensions
    int M = 1024; // Example size for matrix A (Mxw)
    int w = 32;   // Warp size, also width of matrix A

    // Allocate host memory
    float *h_a = (float *)malloc(M * w * sizeof(float));
    float *h_c = (float *)malloc(M * M * sizeof(float));

    // Initialize host matrices
    for (int i = 0; i < M * w; i++)
    {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_c;
    cudaMalloc(&d_a, M * w * sizeof(float));
    cudaMalloc(&d_c, M * M * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, M * w * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((M + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
    bankConflictMultiply<<<dimGrid, dimBlock>>>(d_a, d_c, M);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, M * M * sizeof(float), cudaMemcpyDeviceToHost);

    // Check the results
    // (Checking code here)

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_c);

    return 0;
}

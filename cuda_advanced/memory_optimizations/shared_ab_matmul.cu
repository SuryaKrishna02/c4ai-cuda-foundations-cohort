#include <cuda_runtime.h>
#include <iostream>

#define TILE_DIM 32 // Define TILE_DIM as the warp size, typically 32 for current GPUs

__global__ void coalescedMultiply(float *a, float *b, float *c, int M,
                                  int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
        bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];
    __syncthreads();
    if (row < M && col < N)
    {
        for (int i = 0; i < TILE_DIM; i++)
        {
            sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    // Define matrix dimensions
    int M = 1024; // Example size for matrix A (Mxw)
    int w = 32;   // Warp size, also width of matrix A and height of matrix B
    int N = 1024; // Example size for matrix B (wxN)

    // Allocate host memory
    float *h_a = (float *)malloc(M * w * sizeof(float));
    float *h_b = (float *)malloc(w * N * sizeof(float));
    float *h_c = (float *)malloc(M * N * sizeof(float));

    // Initialize host matrices
    for (int i = 0; i < M * w; i++)
    {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < w * N; i++)
    {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * w * sizeof(float));
    cudaMalloc(&d_b, w * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, M * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, w * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
    coalescedMultiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check the results
    // (Checking code here)

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

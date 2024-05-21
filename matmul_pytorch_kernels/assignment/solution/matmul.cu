#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int m, int n, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p)
    {
        float sum = 0.0;
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

int main()
{
    const int m = 2, n = 2, p = 2;
    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * p * sizeof(float);
    size_t sizeC = m * p * sizeof(float);

    // Allocate host memory
    float h_A[] = {5, 2, 1, 6};

    float h_B[] = {2, 8, 4, 7};

    float *h_C = (float *)malloc(sizeC);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    // Allocate memory on the device for matrix A
    cudaMalloc((void **)&d_A, sizeA);
    // Allocate memory on the device for matrix B
    cudaMalloc((void **)&d_B, sizeB);
    // Allocate memory on the device for matrix C
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 threadsPerBlock(2, 2); // Adjusted for the 2x2 matrix
    dim3 blocksPerGrid(1, 1);   // Only one block is needed

    // Launch the kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize(); // Ensure all threads have finished

    // Copy the result back to the host
    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Resulting Matrix C:\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            printf("%f ", h_C[i * p + j]);
        }
        printf("\n");
    }

    // Free device memory
    // Free memory allocated for matrix A on the device
    cudaFree(d_A);
    // Free memory allocated for matrix B on the device
    cudaFree(d_B);
    // Free memory allocated for matrix C on the device
    cudaFree(d_C);

    // Free host memory
    // Free memory allocated for matrix C on the host
    free(h_C);

    return 0;
}
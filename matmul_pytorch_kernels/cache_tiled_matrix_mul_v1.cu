#include <iostream>
#include <cassert>
using namespace std;

// Static shmem calculation for convenience (int 16x16 matrix)
#define SHMEM_SIZE 16 * 16 * 4

__global__ void tiledMatrixMul(int *a, int *b, int *c, int n, int tile_size)
{
    // Two statically-size pieces of shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Shorten these parameters for clean re-use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate global row and column positions for this thread
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // Intermediate sum for element being written
    int temp_val = 0;

    // Sweep tiles over entire matrix
    for (int i = 0; i < (n / tile_size); i++)
    {
        /*
            Every thread in a threadblock loads one element into shared memory
            The element location in shared memory corresponds to the thread's
            position in the threadblock (e.g. thread [0, 0] loads for
            A[0 * tile_size + 0], and B[0*tile_size+0].)

            Explanation of indexing parameters
            For A:
                        row*n: Indexes the global row for this thread (loop-invariant)
                  i*tile_size: Indexes the next set of columns each iteration
                           tx: Indexes the column within that set

            For B:
                  i*tile_size: Indexes the next set of rows each iteration
                         ty*n: Indexes the row within that set
                          col: Indexes the global column (loop-invariant)
        */
        A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
        B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

        // Ensure all threads have loaded their data before proceeding
        __syncthreads();

        // Calculate all temp values for this title
        for (int j = 0; j < tile_size; j++)
        {
            temp_val += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
        }

        // Ensure some threads don't progress and stomp current shared memory values
        __syncthreads();
    }
    c[(row * n) + col] = temp_val;
}

// Check result
void verify_result(int *a, int *b, int *c, int n)
{
    int *verify_c;
    verify_c = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                verify_c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            assert(c[i * n + j] == verify_c[i * n + j]);
        }
    }
}

// Initialization function for matrices;
void init_matrices(int *a, int *b, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
}

int main()
{
    // Matrix size of 1024 x 1024;
    int n = 1 << 10;
    int tile_size = 16;

    // Size (in bytes) of matrix
    size_t bytes = n * n * sizeof(int);

    // Host pointers
    int *h_a, *h_b, *h_c;

    // Allocate host memory
    h_a = (int *)malloc(bytes);
    h_b = (int *)malloc(bytes);
    h_c = (int *)malloc(bytes);

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocated device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize matrices
    init_matrices(h_a, h_b, n);

    // Copy data to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int BLOCK_SIZE = 16;

    // Blocks in each dimension
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    // use dim3 objects
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Launch Kernel
    tiledMatrixMul<<<grid, threads>>>(d_a, d_b, d_c, n, tile_size);

    // Wait for GPU to complete execution
    cudaDeviceSynchronize();

    // Copy back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check the result
    // verify_result(h_a, h_b, h_c, n);

    // Free the GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free the CPU memory
    free(h_a);
    free(h_b);
    free(h_c);

    cout << "COMPLETED SUCCESSFULLY\n"
         << endl;

    return 0;
}
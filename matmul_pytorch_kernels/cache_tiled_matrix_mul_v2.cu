#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

// Static shmem calculation for convenience (int 16x16 matrix)
#define SHMEM_SIZE 16 * 16 * 4

__global__ void tiledMatrixMul(int *a, int *b, int *c, int N, int tile_size)
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
    for (int i = 0; i < (N / tile_size); i++)
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
        A[(ty * tile_size) + tx] = a[row * N + (i * tile_size + tx)];
        B[(ty * tile_size) + tx] = b[(i * tile_size * N + ty * N) + col];

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
    c[(row * N) + col] = temp_val;
}

// Initializes a square matrix with random numbers between 0-100;
void init_matrix(int *m, int N)
{
    for (int i = 0; i < N * N; i++)
    {
        m[i] = rand() % 100;
    }
}

// verify the result on the CPU
void verify_result(int *a, int *b, int *c, int N)
{
    int tmp;
    // For every row...
    for (int i = 0; i < N; i++)
    {
        // For every col...
        for (int j = 0; j < N; j++)
        {
            tmp = 0;
            // For every element in the row-col pari.
            for (int k = 0; k < N; k++)
            {
                tmp += a[i * N + k] * b[k * N + j];
            }
            // Check each result
            assert(tmp == c[i * N + j]);
        }
    }
}

int main()
{
    // Set our square matrix dimension (2^10 x 2^10 default)
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);
    int tile_size = 16;

    // Allocate memory for our matrices
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set our CTA and Grid dimensions
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // Setup our kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Launch our kernel
    tiledMatrixMul<<<BLOCKS, THREADS>>>(a, b, c, N, tile_size);
    cudaDeviceSynchronize();

    // Verify the result
    // verify_result(a, b, c, N);

    cout << "PROGRAM COMPLETED SUCCESSFULLY!" << endl;

    return 0;
}
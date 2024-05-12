#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void add_basic(int n, int *x)
{
    for (int i = 0; i < n; i++)
    {
        x[i] *= 2;
    }
}

int main()
{
    int N = 10;
    int *x;

    cudaMallocManaged(&x, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        x[i] = i + 1;
    }

    cout << "Before Doubling = First 2 elements of the array:" << endl;
    for (int i = 0; i < 2; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;

    add_basic<<<1, 1>>>(N, x);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cout << "After Doubling = First 2 elements of the array:" << endl;
    for (int i = 0; i < 2; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;

    cudaFree(x);

    return 0;
}
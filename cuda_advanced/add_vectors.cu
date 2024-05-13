#include <iostream>
using namespace std;

#define N 10
#define HANDLE_ERROR(err)                                              \
    {                                                                  \
        if (err != cudaSuccess)                                        \
        {                                                              \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    }

__global__ void add(int *a, int *b, int *c)
{
    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int),
                            cudaMemcpyHostToDevice));

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int),
                            cudaMemcpyDeviceToHost));
    // display the results
    for (int i = 0; i < N; i++)
    {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }
    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
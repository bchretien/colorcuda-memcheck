#include <cuda.h>

__global__ void bug_kernel(const int* in, int* out, int num)
{
    const int threadNum = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < num; i += threadNum)
        // Out of bounds access here
        out[i] = in[i] + in[i + 1];

    return;
}

int main()
{
    // Allocate memory
    const int num   = 100;
    int* d_in       = NULL;
    int* d_out      = NULL;
    const int size  = num * sizeof(*d_in);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Compute
    const int blocksPerGrid     = 128;
    const int threadsPerBlock   = 128;

    bug_kernel<<< blocksPerGrid, threadsPerBlock >>>(d_in, d_out, num);

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}


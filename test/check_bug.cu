// See: http://choorucode.com/2011/03/page/2/

#include <cuda.h>

__global__ void fooKernel( const int* inArr, int num, int* outArr )
{
    const int threadNum     = gridDim.x * blockDim.x;
    const int curThreadIdx  = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    for ( int i = curThreadIdx; i < num; i += threadNum )
        outArr[ i ] = inArr[ i ] + inArr[ i + 1 ];

    return;
}

int main()
{
    // Allocate memory
    const int num   = 100;
    int* dInArr     = NULL;
    int* dOutArr    = NULL;
    const int size  = num * sizeof( *dInArr );

    cudaMalloc( &dInArr, size );
    cudaMalloc( &dOutArr, size );

    // Compute
    const int blocksPerGrid     = 128;
    const int threadsPerBlock   = 128;

    fooKernel<<< blocksPerGrid, threadsPerBlock >>>( dInArr, num, dOutArr );

    // Free memory
    cudaFree( dInArr );
    cudaFree( dOutArr );

    return 0;
}


#include "ops/transpose.h"

__global__ void d_transpose(float *d_A, float *d_T, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        d_T[col * M + row] = d_A[row * N + col];
    }
}

void transpose(float *d_A, float *d_T, int M, int N)
{
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    d_transpose<<<gridDim, blockDim>>>(d_A, d_T, M, N);
}

void transpose_test()
{

}

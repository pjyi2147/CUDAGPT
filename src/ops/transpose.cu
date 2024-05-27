#include <cstdio>
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
    int row = 4, col = 3;
    float A[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    float *d_A, *d_T;
    cudaMalloc(&d_A, row * col * sizeof(float));
    cudaMalloc(&d_T, col * row * sizeof(float));

    cudaMemcpy(d_A, A, row * col * sizeof(float), cudaMemcpyHostToDevice);

    transpose(d_A, d_T, row, col);

    float T[col * row];
    cudaMemcpy(T, d_T, col * row * sizeof(float), cudaMemcpyDeviceToHost);

    printf("transpose test, T: \n");
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            printf("%f ", T[i * row + j]);
        }
        printf("\n");
    }
    printf("transpose test done\n\n");
}

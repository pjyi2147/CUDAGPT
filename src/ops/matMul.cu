#include "ops/matMul.h"

__global__ void d_matmul(float *d_A, float *d_B, float *d_C, int M, int N) {
    // TODO: Implement matmul
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += d_A[row * N + i] * d_B[i * N + col];
        }
        d_C[row * N + col] = sum;
    }
}

void matMul(float *d_A, float *d_B, float *d_C, int M, int N) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    d_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);
}

void matMul_test()
{

}

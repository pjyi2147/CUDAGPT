#include "ops/bmm.h"

__global__ void d_bmm(float *d_a, float *d_b, float *d_c, int M, int N, int K) {
// TODO: Implement bmm
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += d_a[row * N + i] * d_b[i * K + col];
        }
        d_c[row * K + col] = sum;
    }
}

void bmm(float *d_a, float *d_b, float *d_c, int M, int N, int K) {
    dim3 blockDim(32, 32);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    d_bmm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
}

void bmm_test()
{

}

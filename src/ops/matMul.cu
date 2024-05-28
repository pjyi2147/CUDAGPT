#include <cstdio>
#include "ops/matMul.h"

__global__ void d_matmul(const float *d_A, const float *d_B, float *d_C, int M, int N) {
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

void matMul(const float *d_A, const float *d_B, float *d_C, int M, int N) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    d_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);
}

void matMul_test() {
    int rowa = 2, cola = 3, rowb = 3, colb = 2;
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {1, 2, 3, 4, 5, 6};
    float c[4] = {0, 0, 0, 0};

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, rowa * cola * sizeof(float));
    cudaMalloc(&d_b, rowb * colb * sizeof(float));
    cudaMalloc(&d_c, rowa * colb * sizeof(float));

    cudaMemcpy(d_a, a, rowa * cola * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, rowb * colb * sizeof(float), cudaMemcpyHostToDevice);

    matMul(d_a, d_b, d_c, rowa, colb);

    cudaMemcpy(c, d_c, rowa * colb * sizeof(float), cudaMemcpyDeviceToHost);

    printf("matMul test, c: \n");
    for (int i = 0; i < rowa; i++) {
        for (int j = 0; j < colb; j++) {
            printf("%f ", c[i * colb + j]);
        }
        printf("\n");
    }
    printf("matMul test done\n\n");
}

#include <cstdio>
#include "ops/matScale.h"

__global__ void d_matScale(float *d_a, float *d_b, float alpha, int N) {
    //TODO: Implement matScale
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // apply scaling
    if (col < N) {
        d_b[col] = alpha * d_a[col];
    }
}

void matScale(float *d_a, float *d_b, float alpha, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_matScale<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, alpha, N);
}

void matScale_test() {
    int row = 5, col = 5;
    float a[25] = {1, 2, 3, 4, 5,
                   6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25};
    float scale = 0.5;

    float *d_a, *d_b;
    cudaMalloc(&d_a, row * col * sizeof(float));
    cudaMalloc(&d_b, row * col * sizeof(float));

    cudaMemcpy(d_a, a, row * col * sizeof(float), cudaMemcpyHostToDevice);

    matScale(d_a, d_b, scale, row * col);

    float b[25];
    cudaMemcpy(b, d_b, row * col * sizeof(float), cudaMemcpyDeviceToHost);

    printf("matScale test, b: \n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", b[i * col + j]);
        }
        printf("\n");
    }
    printf("matScale test done\n\n");
}

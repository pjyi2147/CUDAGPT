#include <cstdio>
#include "ops/matAdd.h"

__global__ void d_matAdd(float *d_a, float *d_b, float *d_c, int rows, int cols) {
    // TODO: Implement matAdd
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int i = 0; i < cols; ++i) {
            d_c[row * cols + i] = d_a[row * cols + i] + d_b[row * cols + i];
        }
    }
}

void matAdd(float *d_a, float *d_b, float *d_c, int rows, int cols) {
    int numBlocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_matAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, rows, cols);
}

void matAdd_test() {
    int row = 3, col = 3;
    float a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float b[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, row * col * sizeof(float));
    cudaMalloc(&d_b, row * col * sizeof(float));

    cudaMemcpy(d_a, a, row * col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, row * col * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_c, row * col * sizeof(float));

    matAdd(d_a, d_b, d_c, row, col);

    float c[row * col];
    cudaMemcpy(c, d_c, row * col * sizeof(float), cudaMemcpyDeviceToHost);

    printf("matAdd test, c: \n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", c[i * col + j]);
        }
        printf("\n");
    }
    printf("matAdd test done\n\n");
}

#include <cstdio>
#include "ops/vectAdd.h"

__global__ void d_vectAdd(float *d_a, float *d_b, float *d_c, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        d_c[col] = d_a[col] + d_b[col];
    }
}

void vectAdd(float *d_a, float *d_b, float *d_c, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_vectAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
}

void vectAdd_test() {
    float a[5] = {1, 2, 3, 4, 5};
    float b[5] = {1, 2, 3, 4, 5};

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 5 * sizeof(float));
    cudaMalloc(&d_b, 5 * sizeof(float));
    cudaMalloc(&d_c, 5 * sizeof(float));

    cudaMemcpy(d_a, a, 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 5 * sizeof(float), cudaMemcpyHostToDevice);

    vectAdd(d_a, d_b, d_c, 5);

    float c[5];
    cudaMemcpy(c, d_c, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("vectAdd test, c: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", c[i]);
    }
    printf("\nvectAdd test done\n\n");
}

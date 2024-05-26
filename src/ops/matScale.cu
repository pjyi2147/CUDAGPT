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

}

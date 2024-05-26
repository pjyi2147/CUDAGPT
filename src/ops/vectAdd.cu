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
    
}

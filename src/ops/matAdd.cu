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

}

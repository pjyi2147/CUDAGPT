#include "ops/layernorm.h"

__global__ void d_layernorm(float *d_a, float *d_b, float *d_mean, float *d_var, float *d_gamma, float *d_beta, int N, int D) {
    // TODO: Implement layernorm
}

void layernorm(float *d_a, float *d_b, float *d_mean, float *d_var, float *d_gamma, float *d_beta, int N, int D) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_layernorm<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_mean, d_var, d_gamma, d_beta, N, D);
}

void layernorm_test() {

}

#include "ops/relu.h"

__global__ void d_relu(float *d_a, float *d_b, float alpha, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// apply leaky ReLU
	if (col < N) {
		d_b[col] = fmaxf(alpha * d_a[col], d_a[col]);
	}
}

void relu(float *d_a, float *d_b, float alpha, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_relu<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, alpha, N);
}

void relu_test() {

}

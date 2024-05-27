#include <cstdio>
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
    float a[10] = {-1, -2, -3, -4, -5, 1, 2, 3, 4, 5};

    float *d_a, *d_b;
    cudaMalloc(&d_a, 10 * sizeof(float));
    cudaMalloc(&d_b, 10 * sizeof(float));

    cudaMemcpy(d_a, a, 10 * sizeof(float), cudaMemcpyHostToDevice);

    relu(d_a, d_b, 0.1, 10);

    float b[10];
    cudaMemcpy(b, d_b, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("relu test, b: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", b[i]);
    }
    printf("\nrelu test done\n\n");
}

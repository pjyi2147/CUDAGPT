#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "ops/softmax.h"

__global__ void compute_row_max(const float* d_in, float* d_out, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = d_in[row * cols];
        for (int i = 1; i < cols; ++i) {
            max_val = fmaxf(max_val, d_in[row * cols + i]);
        }
        d_out[row] = max_val;
    }
}

__global__ void compute_exps_and_sum(const float* d_in, const float* d_row_max,
                                     float* d_exps, float* d_row_sum,
                                     int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            d_exps[row * cols + i] = expf(d_in[row * cols + i] - d_row_max[row]);
            sum += d_exps[row * cols + i];
        }
        d_row_sum[row] = sum;
    }
}

__global__ void compute_softmax(const float* d_exp, const float* d_row_sum,
                                float* d_out, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int i = 0; i < cols; ++i) {
            d_out[row * cols + i] = d_exp[row * cols + i] / d_row_sum[row];
        }
    }
}

void softmax(const float* d_in, float* d_out, int rows, int cols)
{
    int numBlocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_row_max;
    float* d_exps;
    float* d_row_sum;

    cudaMalloc(&d_row_max, rows * sizeof(float));
    cudaMalloc(&d_exps, rows * cols * sizeof(float));
    cudaMalloc(&d_row_sum, rows * sizeof(float));

    compute_row_max<<<numBlocks, BLOCK_SIZE>>>(d_in, d_row_max, rows, cols);

    compute_exps_and_sum<<<numBlocks, BLOCK_SIZE>>>(d_in, d_row_max, d_exps, d_row_sum, rows, cols);

    compute_softmax<<<numBlocks, BLOCK_SIZE>>>(d_exps, d_row_sum, d_out, rows, cols);

    cudaFree(d_row_max);
    cudaFree(d_exps);
    cudaFree(d_row_sum);
}

void softmax_test()
{
    const int rows = 2;
    const int cols = 3;
    float input[rows * cols] = {3.0f, 2.0f, 3.0f, 3.0f, 5.0f, 6.0f};
    float* output = (float*)malloc(rows * cols * sizeof(float));

    float* d_in;
    float* d_out;

    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));

    cudaMemcpy(d_in, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    softmax(d_in, d_out, rows, cols);

    cudaMemcpy(output, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", output[i * cols + j]);
        }
        printf("\n");
    }
}

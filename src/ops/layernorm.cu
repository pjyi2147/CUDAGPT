#include <cstdio>
#include <cmath>
#include "ops/layernorm.h"

__global__ void d_mean(float *d_a, float *d_mean, float *redArr1, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int padM = (M + blockDim.x - 1) / blockDim.x * blockDim.x;
    int col = idx % padM;
    int row = idx / padM;

    if (row < N && col < M) {
        redArr1[idx] = d_a[idx];
    }
    __syncthreads();

    for (int s = 1; s < M; s *= 2) {
        if (col % (2 * s) == 0) {
            redArr1[idx] += redArr1[idx + s];
        }
        __syncthreads();
    }

    if (col % padM == 0 && row < N) {
        d_mean[row] = redArr1[idx] / M;
    }
}

__global__ void d_var(float *d_a, float *d_var, float *d_mean, float *redArr1,
                      int N, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int padM = (M + blockDim.x - 1) / blockDim.x * blockDim.x;
    int col = idx % padM;
    int row = idx / padM;
    if (row < N && col < M) {
        redArr1[idx] = (d_a[idx] - d_mean[row]) * (d_a[idx] - d_mean[row]);
    }
    __syncthreads();

    for (int s = 1; s < M; s *= 2) {
        if (col % (2 * s) == 0) {
            redArr1[idx] += redArr1[idx + s];
        }
        __syncthreads();
    }

    if (col % padM == 0 && row < N) {
        d_var[row] = redArr1[idx] / M;
    }
}

__global__ void d_layernorm(float *d_a, float *d_norm, float *d_mean, float *d_var,
                            int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int padM = (M + blockDim.x - 1) / blockDim.x * blockDim.x;
    int col = idx % padM;
    int row = idx / padM;

    if (row < N && col < M) {
        d_norm[idx] = (d_a[idx] - d_mean[row]) / sqrtf(d_var[row] + 1e-6);
    }
}


void layernorm(float *d_a, float *d_norm, int N, int M) {
    int padM = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int numBlocks = (N * padM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_padA, *d_padN, *d_m, *d_v;
    float *redArr1;

    cudaMalloc(&d_padA, N * padM * sizeof(float));
    cudaMalloc(&d_padN, N * padM * sizeof(float));
    cudaMalloc(&d_m, N * sizeof(float));
    cudaMalloc(&d_v, N * sizeof(float));

    cudaMalloc(&redArr1, 2 * N * padM * sizeof(float));

    for (int i = 0; i < N; i++) {
        cudaMemcpy(d_padA + i * padM, d_a + i * M, M * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    d_mean<<<numBlocks, BLOCK_SIZE>>>(d_padA, d_m, redArr1, N, M);
    cudaDeviceSynchronize();

    d_var<<<numBlocks, BLOCK_SIZE>>>(d_padA, d_v, d_m, redArr1, N, M);
    cudaDeviceSynchronize();

    d_layernorm<<<numBlocks, BLOCK_SIZE>>>(d_padA, d_padN, d_m, d_v, N, M);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        cudaMemcpy(d_norm + i * M, d_padN + i * padM, M * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_padA);
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(redArr1);
}

void layernorm_test() {
    float a[5] = {1, 2, 3, 4, 5};

    float *d_a, *d_norm;
    cudaMalloc(&d_a, 5 * sizeof(float));
    cudaMalloc(&d_norm, 5 * sizeof(float));

    cudaMemcpy(d_a, a, 5 * sizeof(float), cudaMemcpyHostToDevice);

    layernorm(d_a, d_norm, 1, 5);

    float norm[5];
    cudaMemcpy(norm, d_norm, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("layernorm test, norm: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", norm[i]);
    }
    printf("\nlayernorm test done\n\n");
}

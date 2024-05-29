#include <cstdio>
#include <random>
#include "ops/matMul.h"
#include "ops/transpose.h"

__global__ void d_matmul(const float *d_A, const float *d_B, float *d_C, int M, int N, int O) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int p = 0; p < (N - 1) / TILE_SIZE + 1; ++p)
    {
        if (row < M && p * TILE_SIZE + threadIdx.x < N)
        {
            s_A[threadIdx.y][threadIdx.x] = d_A[row * N + p * TILE_SIZE + threadIdx.x];
        }
        else
        {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (p * TILE_SIZE + threadIdx.y < N && col < O)
        {
            s_B[threadIdx.y][threadIdx.x] = d_B[(p * TILE_SIZE + threadIdx.y) * O + col];
        }
        else
        {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
        {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < O)
    {
        d_C[row * O + col] = sum;
    }
}

void matMul(const float *d_A, const float *d_B, float *d_C, int M, int N, int O) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    d_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, O);
}

void matMul_test() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    int rowa = 200, cola = 300, rowb = 300, colb = 200;
    float a[rowa * cola], b[rowb * colb], c[rowa * colb];

    for (int i = 0; i < rowa * cola; i++) {
        a[i] = dist(gen);
    }

    for (int i = 0; i < rowb * colb; i++) {
        b[i] = dist(gen);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, rowa * cola * sizeof(float));
    cudaMalloc(&d_b, rowb * colb * sizeof(float));
    cudaMalloc(&d_c, rowa * colb * sizeof(float));

    cudaMemcpy(d_a, a, rowa * cola * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, rowb * colb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, rowa * colb * sizeof(float));

    matMul(d_a, d_b, d_c, rowa, cola, colb);

    cudaMemcpy(c, d_c, rowa * colb * sizeof(float), cudaMemcpyDeviceToHost);

    float ans_c[rowa * colb];
    for (int i = 0; i < rowa; i++) {
        for (int j = 0; j < colb; j++) {
            ans_c[i * colb + j] = 0;
            for (int k = 0; k < cola; k++) {
                ans_c[i * colb + j] += a[i * cola + k] * b[k * colb + j];
            }
        }
    }

    // verify
    bool pass = true;
    for (int i = 0; i < rowa * colb; i++) {
        if (abs(c[i] - ans_c[i]) > 5e-4) {
            pass = false;
            break;
        }
    }

    printf("matMul test: %s\n", pass ? "pass" : "fail");
    printf("matMul test done\n\n");
}

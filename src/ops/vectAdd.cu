#include <cstdio>
#include <random>
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

__global__ void d_vectBatchAdd(float *d_in, float *d_add, float *d_out, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    if (idx < M * N) {
        d_out[row * N + col] = d_in[row * N + col] + d_add[col];
    }
}

void vectBatchAdd(float *d_in, float *d_add, float *d_out, int M, int N) {
    int numBlocks = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_vectBatchAdd<<<numBlocks, BLOCK_SIZE>>>(d_in, d_add, d_out, M, N);
}

void vectAdd_test() {
    int M = 100;
    float a[M];
    float b[M];

    // init random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    for (int i = 0; i < M; i++) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * sizeof(float));
    cudaMalloc(&d_b, M * sizeof(float));
    cudaMalloc(&d_c, M * sizeof(float));

    cudaMemcpy(d_a, a, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, M * sizeof(float), cudaMemcpyHostToDevice);

    vectAdd(d_a, d_b, d_c, M);

    float c[M];
    cudaMemcpy(c, d_c, M * sizeof(float), cudaMemcpyDeviceToHost);

    float ans_c[M];
    for (int i = 0; i < M; i++) {
        ans_c[i] = a[i] + b[i];
    }

    // verify
    bool pass = true;
    for (int i = 0; i < M; i++) {
        if (c[i] != ans_c[i]) {
            pass = false;
            break;
        }
    }

    printf("vectAdd test: %s\n", pass ? "pass" : "fail");
    printf("vectAdd test done\n\n");
}

void vectBatchAdd_test() {
    int M = 200;
    int N = 5;
    float a[M * N];
    for (int i = 0; i < M * N; i++) {
        a[i] = i;
    }
    float b[N] = {1, 2, 3, 4, 5};

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    vectBatchAdd(d_a, d_b, d_c, M, N);
    cudaDeviceSynchronize();

    float c[M * N];
    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float ans_c[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            ans_c[i * N + j] = a[i * N + j] + b[j];
        }
    }

    // verify
    bool pass = true;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != ans_c[i]) {
            pass = false;
            break;
        }
    }

    printf("vectBatchAdd test: %s\n", pass ? "pass" : "fail");
    printf("vectBatchAdd test done\n\n");
}
